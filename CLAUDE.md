## Language & Tools
- Rust (edition 2024)
- cargo, clippy, rustfmt
- Tests: cargo test / cargo nextest
- All code, comments, PRs, and commit messages in English

## Project Structure
- `agua-core` — core library crate
- `agua-cli` — CLI binary crate

## Code Standards
- Run `cargo fmt` before commit
- No clippy warnings allowed
- All public APIs must have doc comments
- Error handling with thiserror/anyhow, no unwrap() in production code
- No `unsafe` code without explicit justification and review
- Follow Rust API guidelines (RFC 430): snake_case modules, CamelCase types
- Write unit tests for all new logic

## Dependencies
- Minimize dependencies; prefer well-maintained crates
- Audit and justify new dependency additions
- Be mindful of compile time and binary size impact

## Logging
- Use `tracing` for structured logging
- Log levels: error (failures), warn (recoverable issues), info (key events), debug/trace (development)

## Security
- Never commit secrets, credentials, `.env` files, or infrastructure config
- Keep `.gitignore` up to date (target/, .env, *.pem, editor files)

## Git Conventions
- Conventional commits: feat:, fix:, refactor:, test:, docs:, chore:
- One commit per logical change

## CI
- All of fmt, clippy, and test must pass before merge

## Teamwork
- Architecture changes require plan approval
- Code review required before merge
- All tests must pass before a task is marked complete
