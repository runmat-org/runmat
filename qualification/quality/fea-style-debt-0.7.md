# Reviewed FEA architecture and style debt

CI compares current FEA findings with two reviewed inventories under
[RM-1287](https://linear.app/dystr/issue/RM-1287). Compiler errors, numerical
failures and findings outside the recorded scope fail their respective gates.
Every other first-party crate retains strict all-target Clippy.

## Architecture inventory

`fea-architecture-debt-0.7.json` records 111 findings covering module
concentration, module size and existing suppression sites in the listed FEA
domains. The checker requires the exact diagnostic multiset, an ancestor review
commit, its Git input identity and the pinned compiler. The language-surface
ledger runs independently and covers 136 current members.

## Clippy inventory

`fea-lint-debt-0.7.json` records 179 diagnostic source sites, grouped into 166
identities, with 265 emissions across the library and library-test targets.
An identity includes the lint, diagnostic message and all primary source-text
anchors. CI compares both identities and emission counts; line-number shifts
alone preserve an identity. A changed finding requires review.

The consolidated Rust 1.90.0 census at
`3d1c1b3b5ec5c57ccda7ecde19747004f8b93b56` removed four previously recorded
partial-order findings in `algebra/elastic_rigidity.rs`. Every retained identity
and emission count matches the prior inventory. No new exception was added.
The retained JSONL SHA-256 is
`152dc4f633b477b298d30de7361b9961fd68fd1dc1bf3b00ff682f97019c82d3`.

CI retains the separate FEA lint log and checks it against the ledger. The
expected Clippy failure under `-D warnings` is accepted only when that exact
comparison passes. Numerical test selection and assertions remain unchanged.

## Closure

RM-1287 owns repairs to these findings and removal of the exceptions. Each
repair needs semantic review and applicable numerical evidence. Partial-order
edits must preserve NaN behavior. RM-1297 separately covers extended
transient-rod qualification.
