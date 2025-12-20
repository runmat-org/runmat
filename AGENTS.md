# AGENTS.md

This file provides instructions for AI coding agents contributing to RunMat.

## Project Overview
RunMat is a Rust-based MATLAB-like runtime and REPL.
Correctness, formatting, and CI cleanliness are critical.

## Contribution Rules (Strict)
- NEVER push directly to `main`
- All changes must be proposed via a Pull Request from a fork
- One logical change per PR

## Required Checks Before Any Commit
Before committing or proposing changes, ALWAYS run:

- cargo fmt
- cargo clippy --all-targets --all-features -D warnings
- cargo test --all

If any of these fail, do not commit.

## Formatting
- rustfmt output must be clean
- CI treats formatting and clippy warnings as errors

## Scope Discipline
- Do not refactor unrelated code
- Avoid formatting-only changes unless explicitly requested
- Keep diffs minimal and reviewable

## PR Expectations
- Clear description of what changed and why
- Mention relevant CI failures or issues fixed
- Expect maintainer review before merge
