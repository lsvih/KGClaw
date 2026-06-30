# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.x     | :white_check_mark: |

## Reporting a Vulnerability

**Do not open public issues for security vulnerabilities.**

Please report security vulnerabilities privately to the project maintainers.

We aim to acknowledge reports within 48 hours and provide an initial assessment within 7 days.

## Security Considerations for KGClaw Users

### API Keys

KGClaw stores API keys in `~/.kgclaw/config.yaml`. Ensure this file has restricted permissions (`chmod 600`). Never commit this file to version control.

### Sandbox Execution

KGClaw's `run_python` tool executes agent-generated Python code in a sandboxed
environment. The sandbox uses AST-based static analysis to block dangerous
operations before execution.

**Security boundaries:**
- The AST safety checker (`check_code_safety`) blocks imports of dangerous modules (`os`, `subprocess`, `socket`, etc.) and calls to dangerous functions (`eval`, `exec`, `compile`, etc.)
- Execution runs in an isolated subprocess with a 30-second timeout
- Output is truncated to 100,000 characters

**Limitations (known):**
- AST-based analysis is a best-effort defense, not a formal security guarantee
- Sophisticated Python constructs (C extensions, codecs, encoding tricks) may bypass static analysis
- The sandbox is designed for **agent-generated code**, not untrusted user input
- If you allow untrusted users to inject code into the agent's generation pipeline, additional isolation (e.g., container/VM) is recommended

### Prompt Injection

KGClaw processes natural language prompts from users and passes them to LLM APIs.
Be aware that:
- User-provided ontology definitions and document content are included in LLM prompts
- Agent tool call results may contain sensitive document data
- Trace mode (`--trace`) writes full prompts and responses to disk

### Trace Mode

When `--trace` is enabled, full LLM prompts and responses are written to
`.kgclaw/traces/`. These files may contain:
- API keys embedded in URLs (masked when detected)
- PII from processed documents
- Proprietary data from workflow inputs

Keep trace files secure and delete them when no longer needed.

## Disclosure Policy

We follow responsible disclosure. Once a fix is available, we will:
1. Release a patch version
2. Publish a security advisory
3. Credit the reporter (unless they wish to remain anonymous)
