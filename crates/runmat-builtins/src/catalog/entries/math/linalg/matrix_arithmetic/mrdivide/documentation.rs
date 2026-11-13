use super::super::documentation::{builtin, example, faq};
use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Right division", paragraphs: &["`X = A / B` and `mrdivide(A, B)` solve `X * B = A`. Square, rectangular, and rank-deficient inputs use a minimum-norm least-squares solve.", "Both inputs must behave as matrices, with trailing singleton dimensions allowed. Nonscalar inputs must have the same number of columns."] },
    BuiltinDocumentationSection { heading: "Classes and scalar forms", paragraphs: &["A real pair returns a real floating result; a complex operand promotes the solve to complex arithmetic. Logical matrices are converted to double.", "When `B` is scalar, `A / B` scales `A` by `1/B`. Fixed-width integers are supported only in this scalar-right form, where the integer class is preserved. Integer matrix solving is rejected."] },
    BuiltinDocumentationSection { heading: "Acceleration", paragraphs: &["A resident solve is offered to the exact provider that owns its inputs. Mixed-residency operands are uploaded only when their class and precision can be represented by that owner.", "A typed unsupported-operation response permits host fallback. Provider failures and invalid output metadata are surfaced rather than silently changing execution paths."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    example("square", "Solve a square system", "A = [1 2; 3 4];\nB = [5 6; 7 8];\nX = A / B;\nR = X * B", "R agrees with A", "assert(max(abs(R(:) - A(:))) < 1e-10);", BuiltinExampleHarness::Portable),
    example("least-squares", "Compute a least-squares right division", "A = [1 2 3];\nB = [1 0 1; 0 1 1];\nX = A / B", "X = [1 2]", "assert(max(abs(X - [1 2])) < 1e-10);", BuiltinExampleHarness::Portable),
    example("scalar", "Divide by a scalar", "A = [2 4 6];\nX = A / 2", "X = [1 2 3]", "assert(isequal(X, [1 2 3]));", BuiltinExampleHarness::Portable),
    example("complex", "Solve with complex inputs", "A = [1+2i 3-4i];\nB = [2-i 1+i];\nX = A / B;\nR = X * B", "R is the least-squares reconstruction", "assert(size(X, 1) == 1); assert(size(X, 2) == 1); assert(all(isfinite([real(X), imag(X)])));", BuiltinExampleHarness::Portable),
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    faq("Why must A and B have the same number of columns?", "The equation `X * B = A` requires both matrices to describe the same output columns."),
    faq("What happens for singular or rectangular B?", "RunMat computes a minimum-norm least-squares solution with singular-value decomposition."),
    faq("Are higher-dimensional arrays supported?", "Inputs must be effective matrices. Trailing singleton dimensions are allowed; reshape other arrays before solving."),
    faq("How are logical and integer arrays treated?", "Logical matrices become double. Integer values are accepted only for scalar-right division and preserve the integer class."),
    faq("Will a resident result stay on its provider?", "Yes when the provider returns a valid result or the documented scalar fallback can restore it to the exact input owner."),
];
const LINKS: &[BuiltinDocumentationLink] = &[
    builtin("mldivide"),
    builtin("mtimes"),
    builtin("svd"),
    builtin("lu"),
    builtin("gpuArray"),
    builtin("gather"),
    builtin("mpower"),
    builtin("transpose"),
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("mrdivide"), slug: Some("mrdivide"), summary: "Solve linear systems with matrix right division.", description: "`mrdivide(A, B)` is the callable form of `A / B` and solves `X * B = A` for scalar, square, rectangular, rank-deficient, complex, and supported provider-resident inputs.", keywords: &["mrdivide", "matrix right division", "linear systems", "least squares", "gpu"], related: &["mldivide", "mtimes", "svd", "gpuArray", "gather"], sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS, media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Matrix-solve runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/linalg/ops/matrix_arithmetic/solve") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Real, complex, integer, provider, and WGPU right solves", location: "builtins::math::linalg::ops::matrix_arithmetic::mrdivide::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" }], notes: &[] }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
