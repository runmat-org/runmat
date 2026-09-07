use super::super::documentation::{builtin, example, faq};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Left division", paragraphs: &["`X = A \\ B` and `mldivide(A, B)` solve `A * X = B`. Square, rectangular, and rank-deficient inputs use a minimum-norm least-squares solve.", "Both inputs must behave as matrices, with trailing singleton dimensions allowed. Nonscalar inputs must have the same number of rows."] },
    BuiltinDocumentationSection { heading: "Classes and scalar forms", paragraphs: &["A real pair returns a real floating result; a complex operand promotes the solve to complex arithmetic. Logical matrices are converted to double.", "When `A` is scalar, `A \\ B` scales `B` by `1/A`. Fixed-width integers are supported only in this scalar-left form, where the integer class is preserved and division follows fixed-width rules. Integer matrix solving is rejected."] },
    BuiltinDocumentationSection { heading: "Acceleration", paragraphs: &["A resident solve is offered to the exact provider that owns its inputs. Mixed-residency operands are uploaded only when their class and precision can be represented by that owner.", "A typed unsupported-operation response permits host fallback. Provider failures, aliased outputs, wrong shapes, wrong types, and wrong owners are reported rather than treated as an unsupported optimization."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    example(
        "square",
        "Solve a square system",
        "A = [1 2; 3 4];\nb = [5; 6];\nx = A \\ b",
        "x = [-4; 4.5]",
        "assert(max(abs(x - [-4; 4.5])) < 1e-12);",
        BuiltinExampleHarness::Portable,
    ),
    example(
        "least-squares",
        "Solve an overdetermined system",
        "A = [1 2; 3 4; 5 6];\nb = [7; 8; 9];\nx = A \\ b",
        "x = [-6; 6.5]",
        "assert(max(abs(x - [-6; 6.5])) < 1e-10);",
        BuiltinExampleHarness::Portable,
    ),
    example(
        "scalar",
        "Scale by a scalar on the left",
        "s = 2;\nB = [2 4 6];\nX = s \\ B",
        "X = [1 2 3]",
        "assert(isequal(X, [1 2 3]));",
        BuiltinExampleHarness::Portable,
    ),
    example(
        "multiple-rhs",
        "Solve multiple right-hand sides",
        "A = [4 1; 2 3];\nB = eye(2);\nX = A \\ B",
        "X = [0.3 -0.1; -0.2 0.4]",
        "assert(max(abs(X(:) - [0.3; -0.2; -0.1; 0.4])) < 1e-12);",
        BuiltinExampleHarness::Portable,
    ),
    example(
        "complex",
        "Solve a complex system",
        "A = [2+i 1; -1 3-2i];\nB = [1; 4+i];\nX = A \\ B;\nR = A * X",
        "R agrees with B",
        "assert(max(abs(R - B)) < 1e-10);",
        BuiltinExampleHarness::Portable,
    ),
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    faq("Why must A and B have the same number of rows?", "The equation `A * X = B` requires each right-hand side to have one entry for every row of `A`."),
    faq("What happens for singular or rectangular A?", "RunMat computes a minimum-norm least-squares solution with singular-value decomposition."),
    faq("Are higher-dimensional arrays supported?", "Inputs must be effective matrices. Trailing singleton dimensions are allowed; reshape other arrays before solving."),
    faq("How are logical and integer arrays treated?", "Logical matrices become double. Integer values are accepted only for scalar-left division and preserve the integer class."),
    faq("Is backslash a callable alias?", "No. Use `mldivide(A, B)` or the `A \\ B` operator."),
];
const LINKS: &[BuiltinDocumentationLink] = &[
    builtin("mrdivide"),
    builtin("mtimes"),
    builtin("svd"),
    builtin("gpuArray"),
    builtin("gather"),
    builtin("ctranspose"),
    builtin("mpower"),
    builtin("transpose"),
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("mldivide"), slug: Some("mldivide"), summary: "Solve linear systems with matrix left division.", description: "`mldivide(A, B)` is the callable form of `A \\ B` and solves `A * X = B` for scalar, square, rectangular, rank-deficient, complex, and supported provider-resident inputs.", keywords: &["mldivide", "backslash", "matrix left division", "linear systems", "least squares", "gpu"], related: &["mrdivide", "mtimes", "svd", "gpuArray", "gather"], sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS, media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Matrix-solve runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/linalg/ops/matrix_arithmetic/solve") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Real, complex, integer, provider, and WGPU left solves", location: "builtins::math::linalg::ops::matrix_arithmetic::mldivide::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" }], notes: &[] }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
