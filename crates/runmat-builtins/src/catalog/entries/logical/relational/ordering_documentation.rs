macro_rules! define_ordering_documentation {
    (
        constant: $constant:ident,
        name: $name:literal,
        operator: $operator:literal,
        relation: $relation:literal,
        scalar_program: $scalar_program:literal,
        matrix_program: $matrix_program:literal,
        matrix_expected: $matrix_expected:literal,
        expansion_program: $expansion_program:literal,
        expansion_expected: $expansion_expected:literal,
        character_program: $character_program:literal,
        character_expected: $character_expected:literal,
        string_program: $string_program:literal,
        string_expected: $string_expected:literal,
        gpu_program: $gpu_program:literal,
        gpu_expected: $gpu_expected:literal,
        unit_tests: $unit_tests:literal,
        wgpu_test: $wgpu_test:literal
    ) => {
        use crate::{
            BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
            BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
            BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
            BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility,
            BuiltinExampleHarness, BuiltinExampleVerification,
        };

        const SECTIONS: &[BuiltinDocumentationSection] = &[
            BuiltinDocumentationSection {
                heading: "Comparison semantics",
                paragraphs: &[
                    concat!("`", $name, "(A,B)` and `A ", $operator, " B` compare corresponding elements after implicit expansion and return where the left operand is ", $relation, " the right operand. Numeric, logical, character, string, categorical, complex, and symbolic operands follow their domain rules."),
                    "Ordered comparisons project complex numeric values onto their real components. Any comparison involving NaN is false. Fixed-width integers retain exact values across signed, unsigned, and floating operands; character data compares by code point against numeric data and as text against string data.",
                ],
            },
            BuiltinDocumentationSection {
                heading: "Symbolic and accelerated execution",
                paragraphs: &[
                    "A symbolic operand produces a symbolic relation with the broadcast result shape. Nonsymbolic comparisons produce logical values.",
                    "Compatible resident operands use the exact provider's ordering operation. Complex-interleaved inputs project validated real lanes before comparison. Unsupported routes use the exact host implementation and restore explicitly requested residency. Compatible elementwise expressions may fuse.",
                ],
            },
        ];
        const EXAMPLES: &[BuiltinExample] = &[
            BuiltinExample { id: "scalar", title: "Compare scalars", program: $scalar_program, display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, true));" } },
            BuiltinExample { id: "matrix", title: "Compare a matrix with a threshold", program: $matrix_program, display_output: Some($matrix_expected), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: concat!("assert(isequal(tf, logical(", $matrix_expected, ")));"), } },
            BuiltinExample { id: "implicit-expansion", title: "Use implicit expansion", program: $expansion_program, display_output: Some($expansion_expected), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: concat!("assert(isequal(tf, logical(", $expansion_expected, ")));"), } },
            BuiltinExample { id: "characters", title: "Compare character code points", program: $character_program, display_output: Some($character_expected), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: concat!("assert(isequal(tf, logical(", $character_expected, ")));"), } },
            BuiltinExample { id: "strings", title: "Compare strings lexicographically", program: $string_program, display_output: Some($string_expected), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: concat!("assert(isequal(tf, logical(", $string_expected, ")));"), } },
            BuiltinExample { id: "complex-real-projection", title: "Order complex values by real component", program: concat!("tf = ", $name, "([1+99i 3-99i], 2)"), display_output: Some("tf matches the comparison of [1 3] with 2"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: concat!("assert(isequal(tf, ", $name, "([1 3], 2)));"), } },
            BuiltinExample { id: "gpu", title: "Keep an ordering result resident", program: $gpu_program, display_output: Some($gpu_expected), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: concat!("assert(isa(gtf, 'gpuArray'));\nassert(isequal(tf, logical(", $gpu_expected, ")));"), } },
        ];
        const FAQS: &[BuiltinDocumentationFaq] = &[
            BuiltinDocumentationFaq { question: concat!("What class does `", $name, "` return?"), answer: "Nonsymbolic comparisons return logical values. Symbolic operands construct symbolic relations." },
            BuiltinDocumentationFaq { question: "How are NaN values treated?", answer: "Every ordered comparison involving NaN is false." },
            BuiltinDocumentationFaq { question: "How are complex values ordered?", answer: "Only the real components participate in ordering." },
            BuiltinDocumentationFaq { question: "How are strings compared?", answer: "String values compare lexicographically, with implicit expansion for compatible shapes." },
            BuiltinDocumentationFaq { question: "How are character arrays compared?", answer: "Against numeric data they compare as code points; against text they compare as text." },
            BuiltinDocumentationFaq { question: "Are wide integers rounded?", answer: "No. Mixed integer comparisons retain exact fixed-width values." },
            BuiltinDocumentationFaq { question: "Does a GPU comparison gather?", answer: "A supported same-owner operation stays resident. Exact fallback restores explicitly requested residency." },
            BuiltinDocumentationFaq { question: "Can the comparison fuse?", answer: "Compatible elementwise regions may fuse on the selected provider." },
        ];
        const LINKS: &[BuiltinDocumentationLink] = &[
            BuiltinDocumentationLink { label: "eq", target: BuiltinDocumentationLinkTarget::Builtin("eq") },
            BuiltinDocumentationLink { label: "ne", target: BuiltinDocumentationLinkTarget::Builtin("ne") },
            BuiltinDocumentationLink { label: "lt", target: BuiltinDocumentationLinkTarget::Builtin("lt") },
            BuiltinDocumentationLink { label: "le", target: BuiltinDocumentationLinkTarget::Builtin("le") },
            BuiltinDocumentationLink { label: "gt", target: BuiltinDocumentationLinkTarget::Builtin("gt") },
            BuiltinDocumentationLink { label: "ge", target: BuiltinDocumentationLinkTarget::Builtin("ge") },
            BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
            BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
            BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source(concat!("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/rel/", $name, "/mod.rs")) },
        ];
        const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink { label: "Relational runtime", target: BuiltinDocumentationLinkTarget::Source(concat!("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/rel/", $name, "/mod.rs")) }];
        const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
            implementation: IMPLEMENTATION,
            verification: &[
                BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Numeric, complex, text, broadcast, symbolic, and error behavior", location: $unit_tests },
                BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU comparison and residency", location: $wgpu_test },
            ],
            notes: &[],
        };
        pub(super) const $constant: BuiltinDocumentation = BuiltinDocumentation {
            authority: BuiltinDocumentationAuthority::Catalog,
            title: Some($name),
            slug: Some($name),
            summary: concat!("Compare values with the element-wise `", $operator, "` relation."),
            description: concat!("`", $name, "(A,B)` and `A ", $operator, " B` compare compatible operands element by element or construct symbolic relations."),
            keywords: &[$name, $operator, $relation, "logical", "symbolic", "integer", "complex", "gpuArray"],
            related: &["eq", "ne", "lt", "le", "gt", "ge", "gpuArray", "gather"],
            sections: SECTIONS,
            examples: EXAMPLES,
            example_exemption: None,
            faqs: FAQS,
            links: LINKS,
            media: &[],
            evidence: EVIDENCE,
            introduced: Some("Before R2006a"),
            status: Some(BuiltinDocumentationStatus::Stable),
        };
    };
}

pub(super) use define_ordering_documentation;
