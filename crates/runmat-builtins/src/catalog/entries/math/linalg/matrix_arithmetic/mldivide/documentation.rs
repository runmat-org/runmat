use super::super::documentation::{builtin, example, faq};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Left division", paragraphs: &["`X = A \\ B` and `mldivide(A, B)` solve `A * X = B`. Dense square, rectangular, and rank-deficient inputs use a minimum-norm least-squares solve.", "Both inputs must behave as matrices, with trailing singleton dimensions allowed. Nonscalar inputs must have the same number of rows."] },
    BuiltinDocumentationSection { heading: "Classes and scalar forms", paragraphs: &["A real pair returns a real floating result; a complex operand promotes the solve to complex arithmetic. Logical matrices are converted to double.", "When `A` is scalar, `A \\ B` scales `B` by `1/A`. Fixed-width integers are supported only in this scalar-left form, where the integer class is preserved and division follows fixed-width rules. Integer matrix solving is rejected."] },
    BuiltinDocumentationSection {
        heading: "Sparse systems",
        paragraphs: &[
            "On the host, nonsingular square CSC systems use direct triangular substitution when applicable and sparse LU otherwise. Rectangular and numerically singular CSC systems use column-pivoted Householder QR, followed by a complete orthogonal decomposition when needed, to return a minimum-norm least-squares solution. Zero-order systems preserve the solution shape. These routes borrow the coefficient matrix instead of converting it to dense storage. A dense right-hand side with the same real or complex single/double class preserves that class in the result; multiple right-hand sides share one factorization. Mixed-class sparse systems remain unsupported. Nonfinite input or a nonfinite numerical result returns an error.",
            "Sparse factor fill, solve scratch, and final results support finite managed host allocation. Fill is admitted as it grows; a sparse input can still require substantial fill and may exceed the available budget. Triangular solves write directly into admitted result storage. QR uses native-precision factors and results, with scaled double-precision norm reductions. Numerical rank uses native machine epsilon times the larger matrix dimension times the initial largest column norm; an exactly zero matrix has rank zero. A second decomposition retains that selected rank rather than applying another tolerance. This is RunMat's minimum-norm and numerical-rank policy, not a promise of identical mixed-edge or rank decisions to MATLAB. Dense rectangular and rank-deficient inputs retain the minimum-norm SVD route.",
        ],
    },
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
        "general-sparse",
        "Solve a sparse system with two right-hand sides",
        "A = sparse([0 2; 1 3]);\nB = [10 -2; 19 -1];\nX = A \\ B",
        "X = [4 2; 5 -1]",
        "assert(isequal(size(X), [2 2])); assert(max(abs(X(:) - [4; 5; 2; -1])) < 1e-12);",
        BuiltinExampleHarness::Portable,
    ),
    example(
        "tall-sparse",
        "Solve a tall sparse least-squares system",
        "A = sparse([1 0; 0 1; 1 1]);\nb = [2; 3; 5];\nx = A \\ b",
        "x = [2; 3]",
        "assert(max(abs(x - [2; 3])) < 1e-12);",
        BuiltinExampleHarness::Portable,
    ),
    example(
        "wide-sparse-minimum-norm",
        "Solve a wide sparse system with minimum norm",
        "A = sparse([1 0 1; 0 1 1]);\nb = [1; 1];\nx = A \\ b",
        "x = [1/3; 1/3; 2/3]",
        "assert(isequal(size(x), [3 1])); assert(max(abs(A*x-b)) < 1e-12); assert(abs([-1 -1 1]*x) < 1e-12); assert(max(abs(x-[1/3; 1/3; 2/3])) < 1e-12);",
        BuiltinExampleHarness::Portable,
    ),
    example(
        "singular-sparse-minimum-norm",
        "Solve a rank-deficient sparse system",
        "A = sparse([1 2; 2 4]);\nb = [5; 10];\nx = A \\ b",
        "x = [1; 2]",
        "assert(max(abs(A*x-b)) < 1e-12); assert(abs([-2 1]*x) < 1e-12); assert(max(abs(x-[1; 2])) < 1e-12);",
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
    faq("What happens for singular or rectangular A?", "Dense inputs use a minimum-norm singular-value decomposition solve. Same-class host sparse inputs use rank-revealing QR and a complete orthogonal decomposition for minimum-norm least squares. Numerical rank follows the documented native-precision tolerance. MATLAB backslash does not generally promise the same minimum-norm choice for rectangular systems."),
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
