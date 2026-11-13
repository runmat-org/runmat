use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("roots"),
    slug: Some("roots"),
    summary: "Find polynomial zeros from descending-power coefficients.",
    description: "`roots(c)` finds zeros of the polynomial represented by descending-power coefficients `c` and returns them as a column vector.",
    keywords: &["roots", "polynomial", "eigenvalues", "companion"],
    related: &["polyval", "polyfit", "polyder", "polyint"],
    sections: &[
        BuiltinDocumentationSection { heading: "Coefficients and output", paragraphs: &[
            "Exactly zero leading coefficients are discarded; small nonzero coefficients retain their degree. A zero or constant polynomial returns an empty double column. Linear polynomials use `-b/a`, cubics use the current closed-form solver, and other higher degrees use companion-matrix eigenvalues. Root order is not specified.",
            "The current runtime accepts coefficient arrays with at most one nonsingleton dimension, including singleton-extended N-D vector shapes, and returns a column vector. A shape with two nonsingleton dimensions is rejected.",
            "Real coefficients can yield complex roots. The runtime retains nonzero computed real and imaginary components; it does not use a magnitude threshold to turn a small root into zero or change a complex result into a real one. The output uses double or complex-double storage even when the input is single.",
            "Single and double coefficients are accepted. RunMat mode also accepts exact-binary64 fixed-width integer coefficients; values that cannot cross that boundary exactly are rejected. The runtime additionally accepts logical coefficient arrays and converts them for the host solve.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The polynomial solve runs on the CPU. Resident coefficient values are gathered through their owning provider; `roots` returns a host-resident column. This operation ends fusion.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "quadratic", title: "Solve a quadratic", program: "p = [1 -3 2];\nr = roots(p)", display_output: Some("r contains 1 and 2"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(r),[2 1])); assert(max(abs(sort(r)-[1;2]))<1e-8);" } },
        BuiltinExample { id: "repeated", title: "Include repeated roots", program: "p = [1 -2 1 0];\nr = roots(p)", display_output: Some("r contains 0, 1, and 1"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(numel(r)==3); assert(max(abs(sort(r)-[0;1;1]))<1e-6);" } },
        BuiltinExample { id: "leading-zero", title: "Ignore leading zero coefficients", program: "p = [0 0 1 -4];\nr = roots(p)", display_output: Some("r = 4"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(numel(r)==1); assert(abs(r(1)-4)<1e-10);" } },
        BuiltinExample { id: "complex", title: "Find complex conjugate roots", program: "p = [1 0 1];\nr = roots(p)", display_output: Some("r contains i and -i"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(numel(r)==2); assert(max(abs(sort(imag(r))-[-1;1]))<1e-8); assert(max(abs(real(r)))<1e-8);" } },
        BuiltinExample { id: "resident-input", title: "Gather resident coefficients for a CPU solve", program: "p = gpuArray([1 0 -9 0]);\nr = roots(p)", display_output: Some("r contains -3, 0, and 3"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(numel(r)==3); assert(max(abs(sort(r)-[-3;0;3]))<1e-7); assert(~isa(r,'gpuArray'));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What input shape is accepted?", answer: "A row, column, scalar, empty vector, or singleton-extended N-D vector. More than one nonsingleton dimension is rejected." },
        BuiltinDocumentationFaq { question: "How are leading zeros handled?", answer: "They are removed before determining the polynomial degree; all-zero and constant polynomials return an empty column." },
        BuiltinDocumentationFaq { question: "Does roots preserve input precision?", answer: "No. This companion-matrix implementation returns double or complex-double values." },
        BuiltinDocumentationFaq { question: "Are results sorted?", answer: "No. The eigenvalue solver determines their order; sort them explicitly when order matters." },
        BuiltinDocumentationFaq { question: "Does roots solve on a GPU?", answer: "No. Resident coefficients are gathered and the CPU solves the companion-matrix problem." },
        BuiltinDocumentationFaq { question: "What happens to tiny imaginary parts?", answer: "Nonzero computed imaginary parts remain present, including small components; numerical eigensolvers can also produce small roundoff components." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "polyval", target: BuiltinDocumentationLinkTarget::Builtin("polyval") },
        BuiltinDocumentationLink { label: "polyfit", target: BuiltinDocumentationLinkTarget::Builtin("polyfit") },
        BuiltinDocumentationLink { label: "roots", target: BuiltinDocumentationLinkTarget::Builtin("roots") },
        BuiltinDocumentationLink { label: "polyder", target: BuiltinDocumentationLinkTarget::Builtin("polyder") },
        BuiltinDocumentationLink { label: "polyint", target: BuiltinDocumentationLinkTarget::Builtin("polyint") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/poly/roots.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Host companion-matrix solver", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/poly/roots.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Degree, classes, shapes, complex outputs, and stable errors", location: "builtins::math::poly::roots::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Owner gather and host output", location: "builtins::math::poly::roots::tests::roots_gpu_input_gathers_to_host" },
        ],
        notes: &["Reconciles the legacy sidecar's overly narrow N-D rejection claim with the implementation and its logical coefficient acceptance. Example acceptance and hardware qualification remain open."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
