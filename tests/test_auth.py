import json

from backend.api import create_app
from backend.auth import AuthConfig, TokenAuthMiddleware, authorize_request


TOKENS = {
    "dispatcher": "DISPATCHER_TEST_SENTINEL_TOKEN",
    "researcher": "RESEARCHER_TEST_SENTINEL_TOKEN",
    "admin": "ADMIN_TEST_SENTINEL_TOKEN",
}


def _authorization(token=None):
    return b"Bearer " + token.encode("utf-8") if token is not None else b""


def _token_config():
    return AuthConfig.from_mapping("token", TOKENS)


def test_default_local_mode_preserves_unauthenticated_behavior():
    config = AuthConfig.from_mapping("local")
    assert authorize_request(config, "GET", "/api/v1/health", b"") is None
    assert authorize_request(config, "POST", "/api/v1/scenarios/plan", b"") is None

    app = create_app(auth_config=config)
    assert any(layer.cls is TokenAuthMiddleware for layer in app.user_middleware)


def test_token_mode_missing_invalid_and_unsupported_driver_credentials_are_401():
    config = _token_config()
    missing = authorize_request(config, "POST", "/api/v1/scenarios/validate", b"")
    assert missing == (401, "authentication_required", "bearer token required", True)

    sentinel = "UNKNOWN_TOKEN_SENTINEL"
    invalid = authorize_request(config, "POST", "/api/v1/scenarios/validate", _authorization(sentinel))
    assert invalid == (401, "authentication_failed", "bearer token is invalid", True)
    assert sentinel not in json.dumps(invalid)

    driver = authorize_request(config, "POST", "/api/v1/driver/events",
                               _authorization("DRIVER_TEST_SENTINEL_TOKEN"))
    assert driver[0:2] == (401, "authentication_failed")


def test_token_mode_health_is_public_only_when_configuration_is_valid():
    assert authorize_request(_token_config(), "GET", "/api/v1/health", b"") is None
    broken = AuthConfig.from_mapping("token", {"dispatcher": TOKENS["dispatcher"]})
    assert authorize_request(broken, "GET", "/api/v1/health", b"") == (
        503, "auth_misconfigured", "token authorization is not configured", False)


def test_explicit_role_matrix_allows_expected_work_and_denies_escalation_or_unknown_paths():
    config = _token_config()
    assert authorize_request(config, "POST", "/api/v1/scenarios/validate",
                             _authorization(TOKENS["researcher"])) is None
    assert authorize_request(config, "GET", "/api/v1/jobs/job-1",
                             _authorization(TOKENS["researcher"])) is None
    assert authorize_request(config, "POST", "/api/v1/recovery/plan",
                             _authorization(TOKENS["researcher"]))[0:2] == (403, "authorization_denied")
    assert authorize_request(config, "POST", "/api/v1/scenarios/financial-plan",
                             _authorization(TOKENS["dispatcher"])) is None
    assert authorize_request(config, "GET", "/api/v1/jobs/job-1/cancel",
                             _authorization(TOKENS["dispatcher"]))[0:2] == (403, "authorization_denied")
    assert authorize_request(config, "POST", "/api/v1/jobs/job-1/cancel",
                             _authorization(TOKENS["dispatcher"])) is None
    assert authorize_request(config, "GET", "/api/v1/recovery/audits/audit-1",
                             _authorization(TOKENS["researcher"])) is None
    assert authorize_request(config, "GET", "/api/v1/unknown/future",
                             _authorization(TOKENS["admin"]))[0:2] == (403, "authorization_denied")


def test_token_configuration_fails_closed_for_missing_duplicate_or_unknown_mode():
    configs = [AuthConfig.from_mapping("token", {"dispatcher": TOKENS["dispatcher"]}),
               AuthConfig.from_mapping("token", {"dispatcher": "same", "researcher": "same", "admin": "same"}),
               AuthConfig.from_mapping("unexpected", TOKENS)]
    for config in configs:
        result = authorize_request(config, "GET", "/api/v1/health", b"")
        assert result == (503, "auth_misconfigured", "token authorization is not configured", False)
        assert all(token not in json.dumps(result) for token in TOKENS.values())


def test_local_environment_loader_does_not_read_token_values():
    class LocalOnlyEnvironment(dict):
        def get(self, key, default=None):
            if key in {"DEBUGLEAF_DISPATCHER_TOKEN", "DEBUGLEAF_RESEARCHER_TOKEN", "DEBUGLEAF_ADMIN_TOKEN"}:
                raise AssertionError("local mode must not read token environment variables")
            return super().get(key, default)

    config = AuthConfig.from_environment(LocalOnlyEnvironment())
    assert config.mode == "local" and config.ready
