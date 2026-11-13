use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "cubic",
        title: "Integrate a cubic polynomial",
        program: "p = [3 -2 5 7];\nq = polyint(p)",
        display_output: Some("q = [0.75 -0.6667 2.5 7 0]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(q-[0.75 -2/3 2.5 7 0]))<1e-12);",
        },
    },
    BuiltinExample {
        id: "constant",
        title: "Supply an integration constant",
        program: "p = [4 0 -8];\nq = polyint(p, 3)",
        display_output: Some("q = [1.3333 0 -8 3]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(q-[4/3 0 -8 3]))<1e-12);",
        },
    },
    BuiltinExample {
        id: "column-input",
        title: "Column coefficients return a row",
        program: "p = [2; 0; -6];\nq = polyint(p)",
        display_output: Some("q = [0.6667 0 -6 0]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(q),[1 4])); assert(max(abs(q-[2/3 0 -6 0]))<1e-12);",
        },
    },
    BuiltinExample {
        id: "empty",
        title: "Integrate the zero polynomial",
        program: "q = polyint([])",
        display_output: Some("q = 0"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(q,0));",
        },
    },
    BuiltinExample {
        id: "complex",
        title: "Integrate complex coefficients",
        program: "p = [1+2i -3 4i];\nq = polyint(p, -1i)",
        display_output: Some("q = [(1+2i)/3 -1.5 4i -1i]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(q-[(1+2i)/3 -1.5 4i -1i]))<1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu",
        title: "Integrate resident coefficients",
        program: "g = gpuArray([1 -4 6]);\nq = polyint(g);\nresult = gather(q)",
        display_output: Some("result = [0.3333 -2 6 0]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(q,'gpuArray')); assert(max(abs(result-[1/3 -2 6 0]))<1e-12);",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("polyint"),
    slug: Some("polyint"),
    summary: "Integrate a polynomial coefficient vector and append a constant term.",
    description: "`polyint(p)` integrates descending-power coefficients once. `polyint(p,k)` supplies the constant of integration; the default is zero.",
    keywords: &["polyint", "polynomial", "integral", "antiderivative"],
    related: &["polyder", "polyval", "polyfit", "roots"],
    sections: &[
        BuiltinDocumentationSection { heading: "Inputs and result", paragraphs: &[
            "Real and complex single- or double-precision scalar, row, column, and empty coefficient vectors are accepted. The integration constant must be a real or complex scalar. More than one nonsingleton coefficient dimension is rejected.",
            "The result is a row vector even when coefficients are a column vector. An empty coefficient vector produces the constant term alone. Leading zero coefficients remain in place.",
            "The current builtin rejects integer and logical coefficient and constant inputs; it does not implicitly promote them to double. Real single input retains single storage when the result remains real. Complex tensor results currently use double-complex storage.",
        ] },
        BuiltinDocumentationSection { heading: "Provider execution", paragraphs: &[
            "For resident real or complex-interleaved coefficients and a real constant, a compatible provider may integrate the vector without gathering it. A complex constant or unsupported provider hook uses host integration and restores the requested residency when the result representation can be uploaded.",
            "This is a polynomial operation rather than an elementwise fusion node. Output validation and exact-owner fallback apply before a provider result is published.",
        ] },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does polyint preserve column-vector orientation?", answer: "No. The current RunMat builtin returns a coefficient row vector for scalar, row, and column input." },
        BuiltinDocumentationFaq { question: "What happens for an empty vector?", answer: "The zero polynomial integrates to the specified constant, or zero when no constant is supplied." },
        BuiltinDocumentationFaq { question: "Can the constant be complex?", answer: "Yes. A complex scalar is appended to the integrated coefficients; a resident input may require host fallback and restoration." },
        BuiltinDocumentationFaq { question: "Are integer or logical coefficients supported?", answer: "No. Supply single- or double-precision coefficients and a matching real or complex scalar constant." },
        BuiltinDocumentationFaq { question: "Does the result remain resident?", answer: "A compatible provider result remains resident. When the operation gathers to the host, the runtime restores supported output representations when input residency requested it." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "polyder", target: BuiltinDocumentationLinkTarget::Builtin("polyder") },
        BuiltinDocumentationLink { label: "polyval", target: BuiltinDocumentationLinkTarget::Builtin("polyval") },
        BuiltinDocumentationLink { label: "polyfit", target: BuiltinDocumentationLinkTarget::Builtin("polyfit") },
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
        BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
        BuiltinDocumentationLink { label: "roots", target: BuiltinDocumentationLinkTarget::Builtin("roots") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/poly/polyint.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Polynomial integration runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/poly/polyint.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Classes, vector orientation, empty and complex values", location: "builtins::math::poly::polyint::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Resident and fallback integration", location: "builtins::math::poly::polyint::tests" },
        ],
        notes: &["Reconciles the legacy sidecar's incorrect integer/logical promotion, preserved-column orientation, and all-double arithmetic claims against current runtime tests. Standalone example acceptance and hardware qualification remain open."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
