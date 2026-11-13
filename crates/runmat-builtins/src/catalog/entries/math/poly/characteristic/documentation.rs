use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("poly"),
    slug: Some("poly"),
    summary: "Form a monic polynomial from roots or a square matrix.",
    description: "`poly(r)` returns descending-power coefficients whose roots are the elements of vector `r`. `poly(A)` returns the characteristic polynomial of square matrix `A`.",
    keywords: &["poly", "polynomial", "characteristic polynomial", "roots"],
    related: &["roots", "polyval", "eig"],
    sections: &[
        BuiltinDocumentationSection { heading: "Input and result", paragraphs: &[
            "A row or column vector is interpreted as roots; a square matrix is interpreted as a characteristic-polynomial operand. A 1-by-1 input has the same result under either interpretation. An empty 0-by-0 input returns the constant polynomial 1.",
            "The result is a 1-by-(n+1) row of descending-power coefficients. RunMat preserves the input single/double class and keeps complex input in complex storage even when imaginary coefficient components cancel to zero. Nonfinite values, integer/logical arrays, nonsquare matrices, and arrays with more than two dimensions are rejected; this does not claim identical domain behavior in every MATLAB execution mode.",
        ] },
        BuiltinDocumentationSection { heading: "Execution boundary", paragraphs: &[
            "Square matrices use the ordinary Schur factorization; numerical rounding affects the coefficients. Root vectors use an exact admitted output owner under Managed execution. Automatically resident GPU input is gathered through its exact provider during ordinary Unbounded execution, and the result is host-resident. Explicit GPU input is rejected until result restoration has a certified owner; Managed readback and Managed matrix factorization likewise remain unsupported. Cancellation is checked before and within long coefficient updates, but the underlying Schur factorizer cannot be interrupted mid-factorization.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "two-roots", title: "Polynomial with roots 1 and 2", program: "p = poly([1 2]);", display_output: Some("p = [1 -3 2]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(p,[1 -3 2]));" } },
        BuiltinExample { id: "diagonal-characteristic", title: "Characteristic polynomial", program: "p = poly([1 0; 0 2]);", display_output: Some("p = [1 -3 2]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(p-[1 -3 2]))<1e-10);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does poly return roots?", answer: "No. It returns descending polynomial coefficients. Use roots to obtain zeros from coefficients." },
        BuiltinDocumentationFaq { question: "Is matrix input interpreted as root entries?", answer: "Only vectors are root lists. A nonscalar square matrix yields its characteristic polynomial." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "roots", target: BuiltinDocumentationLinkTarget::Builtin("roots") },
        BuiltinDocumentationLink { label: "polyval", target: BuiltinDocumentationLinkTarget::Builtin("polyval") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/poly/characteristic.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Host root update and Schur factorization", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/poly/characteristic.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Root and diagonal matrix coefficient witnesses", location: "builtins::math::poly::characteristic::tests" }],
        notes: &["The current host/finite-factor limitations are explicit; this entry does not claim full device or Managed matrix parity."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
