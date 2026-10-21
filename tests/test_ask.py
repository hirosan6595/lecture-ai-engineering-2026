from types import SimpleNamespace

from app import ask as ask_module


def test_main_displays_token_usage_after_model(monkeypatch, capsys):
    response = SimpleNamespace(
        text="answer",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=7,
        ),
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", lambda api_key: client)
    monkeypatch.setattr("sys.argv", ["ask", "test question"])

    ask_module.main()

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"[model] {ask_module.MODEL}"
    assert lines[1] == "[tokens] input=12 output=7"
    assert lines[-1] == "answer"
