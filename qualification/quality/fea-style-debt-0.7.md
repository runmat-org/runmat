# FEA style debt for the 0.7.0 qualification

The approved release exception covers 183 existing Clippy findings in
`runmat-analysis-fea`, across 109 files. The all-target/all-feature census emits
273 diagnostics: 90 library findings recur in the library-test target, which
adds 93 findings. Repeated emissions count once.

| Family | Distinct findings |
| --- | ---: |
| Argument count, type complexity and enum size | 23 |
| Partial-order expression readability | 20 |
| Iterator, borrow, clone and expression simplification | 129 |
| Slice-argument convention | 2 |
| Documentation layout | 5 |
| Test-module layout | 3 |
| Floating-literal spelling | 1 |

## Source and toolchain

- Code anchor: `2ec4c54b7b3e85b8bf51ed5011bf49f4208ef00e`.
- FEA Git tree: `55c7f312626e67dd08e22ba9583531661139baa7`.
- Compiler-input Git inventory: `28bfe700e440acb45a5c7a9e0c85855e5837a98b`.
- Rust/Clippy: 1.90.0.

CI computes the inventory with
`git ls-tree -r HEAD -- Cargo.toml Cargo.lock rust-toolchain.toml .cargo crates scripts docs bindings verification | git hash-object --stdin`.
These inputs include shared dependency types and generated products.

## Measured evidence

The R1053 diagnostic census exits 101 under `-D warnings`. Its JSONL SHA-256 is
`ac2dc0c17431559fc93c0f3b4f0c14307a47190394c761dd4f7ae7cf45de3186`.
Comparison with the approved R1033 census conserves all 183 diagnostic identities.
Of 109 lint-bearing files, 105 have identical bytes. The four changed files
retain all 19 existing primary-source-line anchors. There are zero new findings.
The comparison receipt SHA-256 is
`8dd578ba21bdfcff06ea77f067e1056a4ff68738676fa76bac9570f6b8944289`.

R1054 removes a test-only clone of a Copy value in the separate Execution crate.
Its strict Execution Clippy check passes with no warnings. R1055 adopts R1054's
exact source bytes at `c816a5927360e6355120f4de1c3690a4d914fb99`. Its FEA source
and dependency inputs are unchanged from the R1053 census.

The final anchor also includes three Windows-only filesystem files and the
approved changelog and two guide corrections. The filesystem module is gated
by `cfg(windows)`; the Linux/macOS FEA lint inputs are unchanged. The Windows
private-filesystem tests pass 3/0, and the repaired Windows products retain
their separate qualification requirement. The broader CI pin includes these
source and documentation changes.

The R1057 inventory adds only the CLI remote-upload suite correction to the
previous compiler inputs. It replaces a duplicated spelling with an existing
constant; FEA source, its dependencies, features and lint-bearing files are
unchanged. The corrected CLI builds on ARM macOS, Intel macOS and Linux.
Intel and Linux each pass their31-case product cohort; Intel also passes eight
assessment commands. The real Server/registered-node witness passes successful
and failed jobs with encrypted terminal output. Server's corresponding repair
and21 passing domain tests are private-product evidence, outside this public
Clippy exception. The approved183 findings and strict policy for every other
first-party crate are unchanged; Windows retains its separate qualification.

## CI policy and closure

CI enforces strict Clippy for every other first-party crate. The FEA exception
requires the exact FEA tree, complete compiler-input inventory and compiler
version above. A changed input restores strict workspace linting. A separate
FEA lint step retains its full diagnostics and expected failure as an artifact.
Cargo checks and numerical test selection retain their existing behavior.

[RM-1287](https://linear.app/dystr/issue/RM-1287), under RM-1280, owns this debt.
Closure requires repairing the findings with semantic review, retaining
affected numerical evidence, and removing the exception when strict FEA Clippy
passes. Partial-order edits must preserve the intended NaN behavior.
RM-1297 separately covers extended transient-rod qualification.
