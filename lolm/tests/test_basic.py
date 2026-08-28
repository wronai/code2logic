"""Basic tests for lolm package."""


def test_import_lolm():
    """Test that lolm can be imported."""
    import lolm

    assert hasattr(lolm, "get_client")
    assert hasattr(lolm, "LLMManager")
    assert hasattr(lolm, "LLMConfig")


def test_import_config():
    """Test config module imports."""
    from lolm import (
        DEFAULT_MODELS,
        DEFAULT_PROVIDER_PRIORITIES,
        LLMConfig,
    )

    assert LLMConfig is not None
    assert isinstance(DEFAULT_MODELS, dict)
    assert isinstance(DEFAULT_PROVIDER_PRIORITIES, dict)


def test_import_clients():
    """Test client classes can be imported."""
    from lolm import (
        BaseLLMClient,
        OllamaClient,
        OpenRouterClient,
    )

    assert BaseLLMClient is not None
    assert OpenRouterClient is not None
    assert OllamaClient is not None


def test_config_defaults():
    """Test LLMConfig has sensible defaults."""
    from lolm import LLMConfig

    config = LLMConfig()
    assert config.default_provider in (None, "auto", "openrouter", "ollama")
    assert isinstance(config.provider_priorities, dict)
    assert isinstance(config.provider_models, dict)


def test_manager_init():
    """Test LLMManager can be instantiated."""
    from lolm import LLMManager

    manager = LLMManager()
    assert manager is not None
    assert hasattr(manager, "initialize")
    assert hasattr(manager, "generate")


def test_recommended_models():
    """Test recommended models are defined."""
    from lolm import RECOMMENDED_MODELS

    assert "openrouter" in RECOMMENDED_MODELS
    assert "ollama" in RECOMMENDED_MODELS
    assert len(RECOMMENDED_MODELS["openrouter"]) > 0


def test_openrouter_client_uses_central_subllm(monkeypatch):
    from lolm import clients

    captured = {}

    def fake_complete(application, function, messages, **kwargs):
        captured.update(
            application=application,
            function=function,
            messages=messages,
            kwargs=kwargs,
        )
        return type("Response", (), {"content": "central result"})()

    monkeypatch.setattr(clients, "subllm_complete", fake_complete)
    client = clients.OpenRouterClient(api_key="test-key")

    assert client.generate("inspect", system="be precise") == "central result"
    assert captured["application"] == "semcod-code2logic"
    assert captured["function"] == "analyze"
    assert captured["kwargs"]["credentials"] == {"OPENROUTER_API_KEY": "test-key"}
