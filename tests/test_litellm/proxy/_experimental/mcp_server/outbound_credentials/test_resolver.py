"""Tests for the resolver dispatch: live arms produce auth, stubbed arms fail closed.

`none`, `api_key` (shared-key source), `authorization_code`, `token_exchange`, and
`client_credentials` are implemented; every other arm, plus the `api_key` BYOK source, returns a
typed `not_implemented` error until its mode lands. Parametrizing the stubs over one config each
also guards reachability: a dropped `case` would hit `assert_never` and raise instead of
returning the stub.
"""

import httpx
import pytest
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials import (
    ApiKeyConfig,
    AuthorizationCodeConfig,
    AwsSigV4Config,
    Byok,
    ClientCredentialsConfig,
    CredError,
    Error,
    NoneConfig,
    NoOpAuth,
    Ok,
    PassthroughConfig,
    Result,
    ServerSpec,
    SharedKey,
    StaticHeaderAuth,
    Subject,
    TokenExchangeConfig,
    UpstreamCredentialProvider,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
    TokenStoreUnavailable,
)

_SUBJECT = Subject(tenant_id="", subject_id="")


def _spec(config):
    return ServerSpec(server_id="s", resource="https://upstream.example.com", config=config)


def _emitted(auth: httpx.Auth) -> httpx.Headers:
    request = httpx.Request("GET", "https://upstream.example.com/mcp")
    flow = auth.auth_flow(request)
    next(flow)
    flow.close()
    return request.headers


@pytest.mark.asyncio
async def test_none_mode_yields_a_no_op_auth():
    result = await UpstreamCredentialProvider().resolve_credentials(_SUBJECT, _spec(NoneConfig()))
    assert isinstance(result, Ok)
    assert isinstance(result.ok, NoOpAuth)


@pytest.mark.asyncio
async def test_api_key_shared_emits_the_configured_header():
    config = ApiKeyConfig(
        header_name="X-API-Key",
        value_prefix="",
        key_source=SharedKey(value=SecretStr("secret-key")),
    )
    result = await UpstreamCredentialProvider().resolve_credentials(_SUBJECT, _spec(config))
    assert isinstance(result, Ok)
    assert isinstance(result.ok, StaticHeaderAuth)
    assert _emitted(result.ok)["X-API-Key"] == "secret-key"


@pytest.mark.asyncio
async def test_api_key_shared_honors_authorization_scheme():
    config = ApiKeyConfig(
        header_name="Authorization",
        value_prefix="Bearer",
        key_source=SharedKey(value=SecretStr("tok")),
    )
    result = await UpstreamCredentialProvider().resolve_credentials(_SUBJECT, _spec(config))
    assert isinstance(result, Ok)
    assert _emitted(result.ok)["Authorization"] == "Bearer tok"


class _FakeTokenStore:
    """An OAuthTokenStore returning a canned per-user token (None == not authorized)."""

    def __init__(self, by_user: dict) -> None:
        self._by_user = by_user

    async def fetch(self, user_id: str, server_id: str):
        return self._by_user.get((user_id, server_id))


@pytest.mark.asyncio
async def test_authorization_code_emits_bearer_for_a_stored_token():
    store = _FakeTokenStore({("alice", "s"): OAuthToken(access_token="at-alice")})
    result = await UpstreamCredentialProvider(oauth_token_store=store).resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(AuthorizationCodeConfig())
    )
    assert isinstance(result, Ok)
    assert _emitted(result.ok)["Authorization"] == "Bearer at-alice"


@pytest.mark.asyncio
async def test_authorization_code_without_token_is_semantically_unauthorized():
    result = await UpstreamCredentialProvider(oauth_token_store=_FakeTokenStore({})).resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(AuthorizationCodeConfig())
    )
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"
    # Semantic only: the per-server challenge is built at the edge, not in the request-free arm.
    assert "Authorization required" in result.error.unauthorized.detail
    assert result.error.unauthorized.www_authenticate is None
    assert result.error.unauthorized.body is None


@pytest.mark.asyncio
async def test_authorization_code_store_unavailable_is_unauthorized():
    class _Unavailable:
        async def fetch(self, user_id: str, server_id: str):
            raise TokenStoreUnavailable("down")

    result = await UpstreamCredentialProvider(oauth_token_store=_Unavailable()).resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(AuthorizationCodeConfig())
    )
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"


@pytest.mark.asyncio
async def test_authorization_code_with_no_store_wired_is_unauthorized():
    result = await UpstreamCredentialProvider().resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(AuthorizationCodeConfig())
    )
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"


@pytest.mark.asyncio
async def test_authorization_code_isolates_by_subject():
    store = _FakeTokenStore({("alice", "s"): OAuthToken(access_token="at-alice")})
    provider = UpstreamCredentialProvider(oauth_token_store=store)
    alice = await provider.resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(AuthorizationCodeConfig())
    )
    bob = await provider.resolve_credentials(Subject(tenant_id="", subject_id="bob"), _spec(AuthorizationCodeConfig()))
    assert isinstance(alice, Ok) and _emitted(alice.ok)["Authorization"] == "Bearer at-alice"
    assert isinstance(bob, Error) and bob.error.tag == "unauthorized"


@pytest.mark.asyncio
async def test_has_user_token_reflects_the_stored_token():
    present = UpstreamCredentialProvider(
        oauth_token_store=_FakeTokenStore({("alice", "s"): OAuthToken(access_token="at")})
    )
    absent = UpstreamCredentialProvider(oauth_token_store=_FakeTokenStore({}))
    spec = _spec(AuthorizationCodeConfig())
    subject = Subject(tenant_id="", subject_id="alice")
    assert await present.has_user_token(subject, spec) is True
    assert await absent.has_user_token(subject, spec) is False


@pytest.mark.asyncio
async def test_has_user_token_false_for_a_non_per_user_mode():
    # A none-mode server has no per-user token to check.
    provider = UpstreamCredentialProvider()
    spec = _spec(NoneConfig())
    assert await provider.has_user_token(Subject(tenant_id="", subject_id="a"), spec) is False


class _FakeExchanger:
    def __init__(self, result: Result[OAuthToken, CredError]) -> None:
        self._result = result
        self.calls: list[tuple[str, str, str]] = []
        self.invalidations: list[tuple[str, str, str]] = []

    async def exchange(self, subject_token, server, config, *, tenant_id=""):
        self.calls.append((subject_token, tenant_id, server.server_id))
        return self._result

    async def invalidate(self, subject_token, server, config, *, tenant_id=""):
        self.invalidations.append((subject_token, tenant_id, server.server_id))


_OBO = TokenExchangeConfig(
    token_exchange_endpoint="https://idp.example.com/token",
    client_id="cid",
    client_secret=SecretStr("csec"),
)


@pytest.mark.asyncio
async def test_token_exchange_emits_the_exchanged_bearer():
    exchanger = _FakeExchanger(Ok(OAuthToken(access_token="exchanged-at")))
    subject = Subject(tenant_id="acme", subject_id="alice", inbound_token=SecretStr("caller-jwt"))
    result = await UpstreamCredentialProvider(token_exchanger=exchanger).resolve_credentials(subject, _spec(_OBO))
    assert isinstance(result, Ok)
    assert _emitted(result.ok)["Authorization"] == "Bearer exchanged-at"
    # The arm hands the unwrapped caller token AND the tenant to the exchanger, never the upstream.
    assert exchanger.calls == [("caller-jwt", "acme", "s")]


@pytest.mark.asyncio
async def test_invalidate_credentials_drops_the_exchanged_token_for_the_subject_and_tenant():
    exchanger = _FakeExchanger(Ok(OAuthToken(access_token="exchanged-at")))
    provider = UpstreamCredentialProvider(token_exchanger=exchanger)
    subject = Subject(tenant_id="acme", subject_id="alice", inbound_token=SecretStr("caller-jwt"))
    await provider.invalidate_credentials(subject, _spec(_OBO))
    assert exchanger.invalidations == [("caller-jwt", "acme", "s")]


@pytest.mark.asyncio
async def test_invalidate_credentials_is_a_noop_without_a_caller_token():
    exchanger = _FakeExchanger(Ok(OAuthToken(access_token="never")))
    provider = UpstreamCredentialProvider(token_exchanger=exchanger)
    await provider.invalidate_credentials(Subject(tenant_id="acme", subject_id="alice"), _spec(_OBO))
    assert exchanger.invalidations == []


@pytest.mark.asyncio
async def test_token_exchange_without_caller_token_is_unauthorized():
    exchanger = _FakeExchanger(Ok(OAuthToken(access_token="never")))
    result = await UpstreamCredentialProvider(token_exchanger=exchanger).resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(_OBO)
    )
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"
    assert result.error.unauthorized.www_authenticate == 'Bearer error="invalid_request"'
    # No caller token means nothing to exchange: the IdP is never hit.
    assert exchanger.calls == []


@pytest.mark.asyncio
async def test_token_exchange_propagates_the_exchanger_error():
    err = CredError.of_upstream_unavailable("idp down")
    result = await UpstreamCredentialProvider(token_exchanger=_FakeExchanger(Error(err))).resolve_credentials(
        Subject(tenant_id="", subject_id="alice", inbound_token=SecretStr("jwt")),
        _spec(_OBO),
    )
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


@pytest.mark.asyncio
async def test_token_exchange_without_an_exchanger_fails_closed():
    # The fail-closed default (no exchanger wired) must not produce a credential.
    result = await UpstreamCredentialProvider().resolve_credentials(
        Subject(tenant_id="", subject_id="alice", inbound_token=SecretStr("jwt")),
        _spec(_OBO),
    )
    assert isinstance(result, Error)
    assert result.error.tag == "misconfigured"


class _FakeM2MSource:
    """A ClientCredentialsTokenSource returning a canned result and recording refetches."""

    def __init__(self, result) -> None:
        self._result = result
        self.gets: list[str] = []
        self.refetches: list[tuple[str, str]] = []

    async def get(self, server_id: str, config):
        self.gets.append(server_id)
        return self._result

    async def refetch(self, server_id: str, config, failed_access_token: str):
        self.refetches.append((server_id, failed_access_token))
        return "fresh-m2m"


_M2M = ClientCredentialsConfig(
    client_id="cid",
    client_secret=SecretStr("csec"),
    token_url="https://idp.example.com/token",
)


async def _emitted_async(auth: httpx.Auth, respond=None) -> tuple[httpx.Headers, list[httpx.Request]]:
    """Drive the async auth flow one request at a time, replying via ``respond`` when given."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return respond(request) if respond else httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), auth=auth) as client:
        await client.get("https://upstream.example.com/mcp")
    return seen[-1].headers, seen


@pytest.mark.asyncio
async def test_client_credentials_emits_the_minted_bearer():
    source = _FakeM2MSource(Ok(OAuthToken(access_token="m2m-at")))
    result = await UpstreamCredentialProvider(client_credentials_source=source).resolve_credentials(
        _SUBJECT, _spec(_M2M)
    )
    assert isinstance(result, Ok)
    headers, _ = await _emitted_async(result.ok)
    assert headers["Authorization"] == "Bearer m2m-at"
    assert source.gets == ["s"]


@pytest.mark.asyncio
async def test_client_credentials_ignores_the_subject():
    # The contract's no-user-context clause: every caller shares the one client identity.
    source = _FakeM2MSource(Ok(OAuthToken(access_token="m2m-at")))
    provider = UpstreamCredentialProvider(client_credentials_source=source)
    alice = await provider.resolve_credentials(Subject(tenant_id="t1", subject_id="alice"), _spec(_M2M))
    bob = await provider.resolve_credentials(Subject(tenant_id="t2", subject_id="bob"), _spec(_M2M))
    assert isinstance(alice, Ok) and isinstance(bob, Ok)
    alice_headers, _ = await _emitted_async(alice.ok)
    bob_headers, _ = await _emitted_async(bob.ok)
    assert alice_headers["Authorization"] == bob_headers["Authorization"] == "Bearer m2m-at"


@pytest.mark.asyncio
async def test_client_credentials_auth_retries_a_401_through_the_source():
    source = _FakeM2MSource(Ok(OAuthToken(access_token="stale-at")))
    result = await UpstreamCredentialProvider(client_credentials_source=source).resolve_credentials(
        _SUBJECT, _spec(_M2M)
    )
    assert isinstance(result, Ok)

    def respond(request: httpx.Request) -> httpx.Response:
        is_stale = request.headers["Authorization"] == "Bearer stale-at"
        return httpx.Response(401) if is_stale else httpx.Response(200)

    headers, seen = await _emitted_async(result.ok, respond)
    assert headers["Authorization"] == "Bearer fresh-m2m"
    assert len(seen) == 2
    assert source.refetches == [("s", "stale-at")]


@pytest.mark.asyncio
async def test_client_credentials_propagates_the_source_error():
    source = _FakeM2MSource(Error(CredError.of_upstream_unavailable("idp down")))
    result = await UpstreamCredentialProvider(client_credentials_source=source).resolve_credentials(
        _SUBJECT, _spec(_M2M)
    )
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


@pytest.mark.asyncio
async def test_client_credentials_with_no_source_wired_fails_closed_on_missing_config():
    # The default source validates the grant fields before any network is touched.
    result = await UpstreamCredentialProvider().resolve_credentials(_SUBJECT, _spec(ClientCredentialsConfig()))
    assert isinstance(result, Error)
    assert result.error.tag == "misconfigured"


_STUBBED = [
    ("api_key_byok", ApiKeyConfig(key_source=Byok())),
    ("passthrough", PassthroughConfig()),
    ("aws_sigv4", AwsSigV4Config(region="us-east-1")),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("label, config", _STUBBED)
async def test_unbuilt_arms_fail_closed_with_not_implemented(label, config):
    result = await UpstreamCredentialProvider().resolve_credentials(_SUBJECT, _spec(config))
    assert isinstance(result, Error)
    assert result.error.tag == "not_implemented"
