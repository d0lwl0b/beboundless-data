## Project Information
- The `/auto_patchright/hubstudio_api/` module helps with Hubstudio client's API interaction to control the fingerprint browser.
- The python `sqlmodel` module in this project refers to github.com/fastapi/sqlmodel, not tiangolo/sqlmodel. `sqlmodel` is `fastapi/sqlmodel`, not `tiangolo/sqlmodel`.
- All SQLModel/AsyncSession usage in `auto_patchright` must use dependency injection for session management. Never create sessions inside functions. All ORM object lifecycle must be managed by the caller. This follows deepwiki best practices and prevents DetachedInstanceError.

## USE MCP

### PRELUDE
lowercase(all);strip_code_fences;collapse_spaces
repo:
- python|cpython=python/cpython;
- pdm=pdm-project/pdm;
- langchain=langchain/langchain;
- sqlmodel=fastapi/sqlmodel;
- pydantic=pydantic/pydantic;
- streamlit=streamlit/streamlit;
- playwright=microsoft/playwright-python;

### RULES (case-insensitive, whole-word)
on_error(msg): for k,r in repo if /\b(k)\b/i in msg -> CALL mcp.deepwiki(repo=r)
on_code(src):  for k,r in repo if /\b(k)\b/i in src -> CALL mcp.context7(repo=r)

### FALLBACKS
if tool_missing(deepwiki|context7): reply "tool unavailable: <name>"

### RULES

- If the error mentions `python`, ask MCP `deepwiki` about the `python/cpython` repo.
- If the error mentions `langchain`, ask MCP `deepwiki` about the `langchain/langchain` repo.
- If the error mentions `sqlmodel`, ask MCP `deepwiki` about the `fastapi/sqlmodel` repo.
- If the error mentions `pydantic`, ask MCP `deepwiki` about the `pydantic/pydantic` repo.
- If the error mentions `streamlit`, ask MCP `deepwiki` about the `streamlit/streamlit` repo.
- If the error mentions `playwright`, ask MCP `deepwiki` about the `microsoft/playwright-python` repo.

- If the code uses `python`, consult MCP `context7` docs for `python/cpython`.
- If the code uses `langchain`, consult MCP `context7` docs for `langchain/langchain`.
- If the code uses `sqlmodel`, consult MCP `context7` docs for `fastapi/sqlmodel`.
- If the code uses `pydantic`, consult MCP `context7` docs for `pydantic/pydantic`.
- If the code uses `streamlit`, consult MCP `context7` docs for `streamlit/streamlit`.
- If the code uses `playwright`, consult MCP `context7` docs for `microsoft/playwright-python`.
