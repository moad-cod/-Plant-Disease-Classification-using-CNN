from backend.app.api.health import health_check


def test_health_check_returns_artifact_status():
    response = health_check()

    assert response.status == "ok"
    assert response.model_available is True
    assert response.anomaly_params_available is True
