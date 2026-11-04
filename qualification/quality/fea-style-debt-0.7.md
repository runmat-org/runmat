# FEA style debt for the 0.7.0 qualification

The approved release exception covers the existing structural/style Clippy
findings in `runmat-analysis-fea`. Its reviewed Git tree is
`48fac1e81490d61f470796a6fc8ad507d5da4823`; the qualification toolchain is
Rust/Clippy 1.90.0.

The reviewed compiler-input Git inventory is
`fb9f1ac2a39c2c6e898c356ef1f318ea8b285433`. CI derives it by piping
`git ls-tree -r HEAD -- Cargo.toml Cargo.lock rust-toolchain.toml .cargo crates scripts docs bindings verification`
to `git hash-object --stdin`. This also pins shared dependency types and generated
inputs that could change FEA lint results while the FEA files stay unchanged.

The complete R732 diagnostic census identified 183 unique sites in 109 files.
Ninety library findings repeat when the library test target is checked; the
test target adds 93 sites. Repeated emissions count once.

| Family | Sites |
| --- | ---: |
| Argument count, type complexity and enum size | 23 |
| Partial-order expression readability | 20 |
| Iterator, borrow, clone and expression simplification | 129 |
| Slice-argument convention | 2 |
| Documentation layout | 5 |
| Test-module layout | 3 |
| Floating-literal spelling | 1 |

Source review confirmed these findings predate the release repair packet.
The partial-order comparisons retain their existing NaN behavior. This
classification establishes style debt; numerical correctness remains subject
to compilation, numerical tests and governed scientific qualification.

CI enforces strict Clippy for every other first-party crate. It permits the FEA
owner exception only while the complete FEA tree, compiler inputs and compiler
version match the values above. Changed inputs restore strict workspace linting.
The separate FEA lint step retains its complete diagnostics as an artifact;
its expected failure remains visible. Cargo checks and numerical test selection
are unchanged.

The 1.0 qualification follow-up [RM-1287](https://linear.app/dystr/issue/RM-1287)
under RM-1280 owns this debt. Closure requires repairing
these findings with semantic review, retaining relevant numerical evidence,
and removing the source-pinned exception once strict FEA Clippy passes.
RM-1297 separately covers extended transient-rod qualification.

The reviewed source is commit `4b1397ee99b9487a3cd6d42d5a6df95c0034d0c8`.
Its 27-file repair changes Runtime and native-runner code. The resolved FEA
normal/build/dev dependency closure excludes both crates, and its FEA tree is
unchanged. The R732 FEA census remains applicable; the updated complete input
pin includes the unrelated runtime repairs. Strict non-FEA qualification is
required separately.
