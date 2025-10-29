# Project Context

## Purpose
This repository (workspace: `hw3f`) holds a small spec-driven project using the OpenSpec workflow. The immediate goal is to capture project conventions and make it straightforward to propose, review, and implement change proposals using the `openspec/` tooling and directory layout.

Primary goals
- Provide a clear, machine- and human-readable project context for AI-assisted development
- Make it easy to create and validate OpenSpec change proposals
- Keep the code and specs small and easy to iterate on for learning and demo purposes

> NOTE: I made a few reasonable assumptions about the stack and conventions below — please review and correct them where needed.

## Tech Stack
- Primary language: TypeScript (assumed; change to Python/Go if different)
- Runtime: Node.js (LTS)
- Build tools: npm / pnpm (choose one; currently unspecified)
- Testing: Jest for unit tests (if TypeScript/Node). If this repo uses Python, replace with pytest.
- Linting / Formatting: ESLint + Prettier (or flake8/black for Python)
- CI: GitHub Actions (recommended; adjust if another CI is used)

If your actual stack differs, replace the entries above. Keep the spec files language-agnostic where possible.

## Project Conventions

### Code Style
- Use a linter and formatter (ESLint + Prettier recommended for TypeScript). Commit hooks (husky) are optional but encouraged.
- Naming: kebab-case for file names and directories, camelCase for variables, PascalCase for exported classes/React components.
- Public APIs and spec text: use SHALL / MUST for normative requirements.

### Architecture Patterns
- Keep capabilities small and focused as described in `openspec/specs/` (one capability per folder).
- Prefer simple, modular services or modules. Avoid premature abstraction.
- If adding a new service or major pattern, include `design.md` inside the change proposal.

### Testing Strategy
- Unit tests: Target core logic with fast-running unit tests (Jest recommended).
- Integration tests: Add minimally for cross-module behavior when needed.
- Spec-driven tests: Map at least one test to each critical requirement scenario from specs.

### Git Workflow
- Branching: `main` (protected) + feature branches named `feat/<short-desc>` or `changes/<change-id>`.
- Commits: Prefer Conventional Commits (e.g., `feat: add widget`, `fix: correct parsing`), and include a reference to the change id when implementing a proposal (e.g., `implement: add-project-context (#add-project-context)`).
- PRs: Reference `openspec/changes/<change-id>` in the description and include a checklist from `tasks.md`.

## Domain Context
- This workspace appears to be focused on practicing OpenSpec-driven changes and agent-assisted coding. There are no application-specific domain models present in the repository yet.
- If your project has domain rules (billing, authentication, privacy), list them here so the assistant can reason correctly.

## Important Constraints
- Keep implementations small and reversible; prefer non-breaking, incremental changes where possible.
- Avoid introducing heavyweight dependencies without justification.
- Security: treat credentials and secrets as out-of-repo (use env and secret stores). Do not hard-code keys.

## External Dependencies
- Typical external services you might integrate: GitHub (CI), npm registry, and any APIs you plan to consume. Document endpoints and auth requirements here when they exist.

## Assumptions (please confirm)
1. Language: TypeScript / Node.js.
2. Package manager: npm or pnpm (unspecified).
3. CI: GitHub Actions is acceptable.

If any assumptions are incorrect, tell me which ones to update and I will revise `openspec/project.md` accordingly.

## Next steps for the repo owner
- Confirm or edit the Tech Stack section.
- Add any domain-specific notes or constraints.
- If you prefer a different testing or linting stack, update the Testing and Code Style sections.

----
Generated/updated by an AI assistant to capture initial project context; please review and approve or request changes.
