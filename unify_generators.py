import re
from pathlib import Path

content = Path("code2logic/generators.py").read_text()

content = content.replace(
    "def generate(self, project: ProjectInfo, detail_level: str = 'standard') -> str:",
    "def generate(self, project: ProjectInfo, **kwargs) -> str:\n        detail = kwargs.get('detail', kwargs.get('detail_level', 'standard'))"
)
content = content.replace(
    "def generate(self, project: ProjectInfo) -> str:",
    "def generate(self, project: ProjectInfo, **kwargs) -> str:"
)
content = content.replace(
    "def generate(self, project: ProjectInfo, flat: bool = False,\n                 detail: str = 'standard') -> str:",
    "def generate(self, project: ProjectInfo, **kwargs) -> str:\n        flat = kwargs.get('flat', False)\n        detail = kwargs.get('detail', 'standard')"
)
content = content.replace(
    "def generate(self, project: ProjectInfo, flat: bool = False,\n                 detail: str = 'standard', compact: bool = True) -> str:",
    "def generate(self, project: ProjectInfo, **kwargs) -> str:\n        flat = kwargs.get('flat', False)\n        detail = kwargs.get('detail', 'standard')\n        compact = kwargs.get('compact', True)"
)
content = content.replace(
    "def generate(self, project: ProjectInfo, detail: str = 'standard') -> str:",
    "def generate(self, project: ProjectInfo, **kwargs) -> str:\n        detail = kwargs.get('detail', 'standard')"
)

Path("code2logic/generators.py").write_text(content)
