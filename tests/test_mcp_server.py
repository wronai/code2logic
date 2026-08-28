from pathlib import Path

import pytest


def test_mcp_project_path_is_confined(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from code2logic.mcp_server import _require_project_path

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("CODE2LOGIC_MCP_PROJECT_ROOT", str(allowed))

    assert _require_project_path(str(allowed / "project")) == str(allowed / "project")
    with pytest.raises(PermissionError, match="CODE2LOGIC_MCP_PROJECT_ROOT"):
        _require_project_path(str(tmp_path / "outside"))


def test_mcp_project_path_must_be_present() -> None:
    from code2logic.mcp_server import _require_project_path

    with pytest.raises(ValueError, match="non-empty"):
        _require_project_path(None)
