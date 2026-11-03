## Prerequisites

- **Git**: https://git-scm.com/downloads
- **uv** (Python package manager): https://docs.astral.sh/uv/getting-started/installation/

Install uv on Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install uv on macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Check it worked (reopen the terminal first):

```bash
uv --version
```