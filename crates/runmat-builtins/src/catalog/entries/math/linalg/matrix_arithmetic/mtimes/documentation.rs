use super::super::documentation::{builtin, example, expected_error_example, faq};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Matrix product", paragraphs: &["`mtimes(A, B)` and `A * B` multiply scalars, vectors, and matrices. For nonscalar matrices, `size(A, 2)` must equal `size(B, 1)`. Row-by-column produces a scalar dot product, while column-by-row produces an outer product.", "Scalars multiply every element without changing the other operand's shape. Logical and character operands enter floating-point arithmetic. An integer operand is supported only when the other operand is scalar; the result preserves the integer class and saturates."] },
    BuiltinDocumentationSection { heading: "Complex and empty arrays", paragraphs: &["Real and complex operands use full complex multiplication when either input is complex. Trailing singleton dimensions are accepted by the runtime's matrix view.", "An `m`-by-0 matrix multiplied by a 0-by-`n` matrix produces an `m`-by-`n` zero matrix. Incompatible inner dimensions return a matrix-dimension error."] },
    BuiltinDocumentationSection { heading: "Acceleration", paragraphs: &["A provider-resident product uses the exact owner of the resident operands. Compatible host operands may be uploaded to that owner. A typed unsupported-operation response permits host fallback; provider failures and invalid output metadata are reported.", "Supported provider results retain device residency. Exact integer scalar fallback gathers authoritative typed storage and restores the result to the resident input's provider."] },
];

const EXAMPLES: &[BuiltinExample] = &[
    example("matrices", "Multiply two matrices", "A = [1 2 3; 4 5 6];\nB = [7 8; 9 10; 11 12];\nC = A * B", "C = [58 64; 139 154]", "assert(isequal(C, [58 64; 139 154]));", BuiltinExampleHarness::Portable),
    example("dot-product", "Multiply a row by a column", "u = [1 2 3];\nv = [4; 5; 6];\nd = mtimes(u, v)", "d = 32", "assert(isequal(d, 32));", BuiltinExampleHarness::Portable),
    example("scalar", "Scale a matrix", "S = 0.5 * eye(3)", "S = diag([0.5 0.5 0.5])", "assert(isequal(S, diag([0.5 0.5 0.5])));", BuiltinExampleHarness::Portable),
    example("complex", "Multiply complex matrices", "A = [1+2i 3-4i; 5+6i 7+8i];\nB = [1-1i; 2+2i];\nC = A * B", "C = [17-1i; 9+31i]", "assert(max(abs(C - [17-1i; 9+31i])) < 1e-12);", BuiltinExampleHarness::Portable),
    example("resident", "Multiply provider-resident matrices", "G1 = gpuArray(single([1 2; 3 4]));\nG2 = gpuArray(single([5; 6]));\nG = G1 * G2;\nR = gather(G)", "R = single([17; 39])", "assert(isa(G1, 'gpuArray'), 'the first mtimes input must be a gpuArray'); assert(isa(G2, 'gpuArray'), 'the second mtimes input must be a gpuArray'); assert(isa(G, 'gpuArray'), 'mtimes must preserve explicit gpuArray residency'); assert(isequal(R, single([17; 39])), 'gathered mtimes values and class must match');", BuiltinExampleHarness::Wgpu),
    expected_error_example("dimension-error", "Reject incompatible inner dimensions", "A = rand(2, 3);\nB = rand(4, 2);\nC = A * B", "Inner matrix dimensions must agree.", "MATLAB:mtimes:InvalidInput"),
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    faq("How is mtimes different from times?", "`mtimes` performs matrix products. `times`, written `.*`, multiplies corresponding elements with implicit expansion."),
    faq("Does mtimes support scalars?", "Yes. A scalar scales the other operand. Integer inputs require this scalar form and preserve their fixed-width class."),
    faq("Are complex values supported?", "Yes. A real/complex pair is promoted to complex arithmetic."),
    faq("Will a result stay on its provider?", "A valid provider result remains resident. Only a typed unsupported operation may take the documented host fallback path."),
    faq("Is matmul a callable alias?", "No. Use `mtimes(A, B)` or the `A * B` operator."),
];

const LINKS: &[BuiltinDocumentationLink] = &[
    builtin("times"),
    builtin("eye"),
    builtin("gpuArray"),
    builtin("gather"),
    builtin("ctranspose"),
    builtin("dot"),
    builtin("mldivide"),
    builtin("mpower"),
    builtin("mrdivide"),
    builtin("trace"),
    builtin("transpose"),
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("mtimes"),
    slug: Some("mtimes"),
    summary: "Multiply scalars, vectors, or matrices with matrix-product semantics.",
    description: "`mtimes(A, B)` implements the callable form of `A * B`, including scalar, real, complex, integer-scalar, empty, and provider-resident cases.",
    keywords: &["mtimes", "matmul", "matrix multiplication", "linear algebra", "gpu"],
    related: &["times", "mldivide", "mrdivide", "mpower", "gpuArray", "gather"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Matrix-product runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/linalg/ops/matrix_arithmetic") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Host, complex, integer, provider, and WGPU matrix products", location: "builtins::math::linalg::ops::matrix_arithmetic::mtimes::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
