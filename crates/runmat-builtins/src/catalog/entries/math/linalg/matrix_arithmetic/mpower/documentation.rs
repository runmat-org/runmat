use super::super::documentation::{builtin, example, expected_error_example, faq};
use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Scalar and matrix powers", paragraphs: &["`mpower(A, B)` and `A ^ B` raise a scalar or square matrix base to a scalar power. Scalar bases use ordinary numeric exponentiation. A nonscalar base must be square and currently requires an integer-valued exponent.", "A zero matrix exponent returns an identity matrix with the base's size and floating class. Positive integer matrix exponents use binary exponentiation, reducing the number of matrix products."] },
    BuiltinDocumentationSection { heading: "Classes and domains", paragraphs: &["Real and complex scalar bases follow scalar power domain rules. Complex matrices use complex matrix products. Square fixed-width integer matrices support nonnegative integer-valued scalar exponents, preserve the base class, and use saturating multiply-accumulate operations.", "Negative matrix exponents are not yet supported. Use an explicit inverse before a positive power when that behavior is appropriate for the program."] },
    BuiltinDocumentationSection { heading: "Acceleration", paragraphs: &["For a provider-resident matrix, identity construction and each product use the exact owner. Temporary handles are released after their last use, and the final handle cannot alias an input or consumed temporary.", "Only typed unsupported-operation responses may select host fallback. Operational provider errors and invalid output metadata are reported."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    example(
        "matrix",
        "Raise a matrix to an integer power",
        "A = [1 3; 2 4];\nC = A ^ 2",
        "C = [7 15; 10 22]",
        "assert(isequal(C, [7 15; 10 22]));",
        BuiltinExampleHarness::Portable,
    ),
    example(
        "zero",
        "Return the identity for exponent zero",
        "A = [5 2; 7 1];\nI = mpower(A, 0)",
        "I = eye(2)",
        "assert(isequal(I, eye(2)));",
        BuiltinExampleHarness::Portable,
    ),
    example(
        "scalar",
        "Raise a scalar to a fractional power",
        "r = mpower(4, 0.5)",
        "r = 2",
        "assert(isequal(r, 2));",
        BuiltinExampleHarness::Portable,
    ),
    expected_error_example(
        "fractional-matrix",
        "Reject a fractional matrix exponent",
        "A = [1 2; 3 4];\nC = mpower(A, 1.5)",
        "Matrix power requires integer exponent.",
        "MATLAB:mpower:InvalidArgument",
    ),
    example(
        "resident",
        "Compute a matrix power on a provider",
        "G = gpuArray(single([2 0; 0 2]));\nH = mpower(G, 3);\nR = gather(H)",
        "R = single([8 0; 0 8])",
        "assert(isa(H, 'gpuArray')); assert(isequal(R, single([8 0; 0 8])));",
        BuiltinExampleHarness::Wgpu,
    ),
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    faq("Does mpower support nonsquare matrices?", "No. Use element-wise `power`, written `.^`, when matrix-product semantics are not intended."),
    faq("Can a matrix exponent be fractional?", "Not currently. Nonscalar bases require an integer-valued exponent."),
    faq("Are negative matrix exponents supported?", "Not currently. Apply `inv` explicitly when an inverse is appropriate."),
    faq("How does mpower differ from power?", "`mpower` composes matrix products. `power` raises corresponding elements independently."),
    faq("Will results stay provider-resident?", "Yes when the exact owner supports every required identity and matrix-product operation. A typed unsupported response selects the documented fallback."),
    faq("What exponent range is accepted for matrix bases?", "The current runtime accepts integer-valued exponents representable as signed 32-bit values."),
];
const LINKS: &[BuiltinDocumentationLink] = &[
    builtin("mtimes"),
    builtin("power"),
    builtin("eye"),
    builtin("inv"),
    builtin("gpuArray"),
    builtin("gather"),
    builtin("mldivide"),
    builtin("mrdivide"),
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("mpower"), slug: Some("mpower"), summary: "Raise a scalar or square matrix to a power.", description: "`mpower(A, B)` is the callable form of `A ^ B` and implements scalar exponentiation plus repeated matrix products for supported integer matrix exponents.", keywords: &["mpower", "matrix power", "linear algebra", "gpu"], related: &["mtimes", "power", "eye", "inv", "gpuArray", "gather"], sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS, media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Matrix-power runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/linalg/ops/matrix_arithmetic/mpower") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, matrix, integer, provider, and WGPU powers", location: "builtins::math::linalg::ops::matrix_arithmetic::mpower::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" }], notes: &[] }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
