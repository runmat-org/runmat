# RunMat Contribution Guide

## Branch Workflow
- `main` is always the active branch. It contains the code that will ship next.
- The repository `README.md` reiterates that `main` is the only branch you should target unless a hotfix branch is explicitly announced.
- **Every PR must be based on and merged into `main`.** Do not open PRs against ad-hoc release branches unless directed by the release lead.
- Quick checklist:
  1. Update local `main`: `git fetch origin && git checkout main && git pull`.
  2. Create your feature branch from local `main` (`git checkout -b feature/my-task`).
  3. Keep your branch rebased on top of `main` to prevent large merge commits.
  4. Open a PR back into `main`; merging happens only via PRs.

## Pull Request Scope
- Keep PRs small and self-contained. One logical change-set per PR makes reviews faster and lowers risk.
- Each PR should be an independent change targetted towards a merge into `main`.
- Avoid bundling unrelated fixes, refactors, or features together. If the work cannot be described in a single sentence, it probably needs multiple PRs.
- When in doubt, split the work. Additional PRs are cheaper than context-switching during review or QA.

## Test Coverage
- Every PR should include tests that cover the new functionality.
- Tests should cover both positive, negative and edge paths.

## Clippy / Rustfmt
- Every PR should pass clippy and rustfmt checks.
- If clippy or rustfmt fail, CI/CD will fail.

## PR Checklist

### Changeset and Branch
- Ensure the PR is based on and being targeted for merging into the `main` branch.
- Ensure there's a single logical change-set in the PR.

### Callouts in the PR Body
- Make the PR body as clear and informative as possible, but avoid being verbose. Let the code diff show what's best shown with code, and use the PR body for high-level context, justification and to call out changes that are not obvious from the code diff.
- If any changes are made to existing test cases, ensure this is flagged in the PR body.