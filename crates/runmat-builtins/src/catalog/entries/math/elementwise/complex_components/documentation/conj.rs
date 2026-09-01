use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Runtime implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/conj.rs"),
    }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "CPU and representation tests", location: "builtins::math::elementwise::conj::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Provider round-trip test", location: "builtins::math::elementwise::conj::tests::conj_gpu_provider_roundtrip" },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`Y = conj(X)` negates each imaginary component while preserving shape and numeric class. Complex values remain complex even when their imaginary component is zero; real numeric and logical values are unchanged.",
            "Typed complex integers use saturating imaginary-component negation. Signed-minimum and unsigned-imaginary endpoint behavior remains explicitly evidence-open in the typed integer capability contract.",
            "Character input is a RunMat extension that returns double code points. String arrays are unsupported.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "Real resident values can use an exact identity path. Supported floating and complex providers execute through the owning provider, while typed unsupported operations gather exactly through that owner and restore a class-preserving result.",
            "The fusion planner may treat real floating-point conjugation as identity. Complex and typed-integer values retain their dedicated runtime semantics.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "complex-scalar",
        title: "Conjugate a complex scalar",
        program: "z = 3 + 4i;\nresult = conj(z)",
        display_output: Some("result = 3 - 4i"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(result - (3 - 4i)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "complex-matrix",
        title: "Conjugate every element of a complex matrix",
        program: "Z = [1+2i 4-3i; -5+0i 7+8i];\nC = conj(Z)",
        display_output: Some("C = [1-2i 4+3i; -5+0i 7-8i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C, [1-2i 4+3i; -5+0i 7-8i]));",
        },
    },
    BuiltinExample {
        id: "real-identity",
        title: "Leave real input unchanged",
        program: "data = [-2.5 0 9.75];\nunchanged = conj(data)",
        display_output: Some("unchanged = [-2.5 0 9.75]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(unchanged, data));",
        },
    },
    BuiltinExample {
        id: "logical-identity",
        title: "Preserve a logical mask",
        program: "mask = logical([0 1 0; 1 1 0]);\nnumeric = conj(mask)",
        display_output: Some("numeric retains logical class and values"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(numeric));\nassert(isequal(numeric, mask));",
        },
    },
    BuiltinExample {
        id: "character-codes",
        title: "Conjugate character code points in RunMat mode",
        program: "chars = 'RunMat';\ncodes = conj(chars)",
        display_output: Some("codes = [82 117 110 77 97 116]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source:
                "assert(isa(codes, \"double\"));\nassert(isequal(codes, [82 117 110 77 97 116]));",
        },
    },
    BuiltinExample {
        id: "gpu-array",
        title: "Preserve provider-resident real data",
        program: "G = gpuArray([1 -2; 3 -4]);\nH = conj(G);\nhost = gather(H)",
        display_output: Some("host = [1 -2; 3 -4]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(host, [1 -2; 3 -4]));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `conj` change real input?", answer: "No. Real numeric and logical values retain their class and value. Character input is a RunMat extension that returns double code points." },
    BuiltinDocumentationFaq { question: "How does `conj` handle a complex zero?", answer: "The value remains complex and its imaginary zero is negated." },
    BuiltinDocumentationFaq { question: "Can I call `conj` on string arrays?", answer: "No. Numeric and logical input is supported; RunMat mode additionally accepts character arrays as an explicit extension." },
    BuiltinDocumentationFaq { question: "Does `conj` always allocate?", answer: "No. Real host values and resident integer or logical handles can use an identity path. Other providers may allocate, and fusion can eliminate intermediates." },
    BuiltinDocumentationFaq { question: "What happens when a provider lacks conjugation support?", answer: "RunMat gathers through the exact owner, applies host semantics, and restores the result through that provider when representable." },
    BuiltinDocumentationFaq { question: "Does GPU execution match CPU behavior?", answer: "Yes. Real values are exact identities, and supported complex values apply the same component negation in their native precision." },
    BuiltinDocumentationFaq { question: "Can `conj` participate in fusion?", answer: "Yes for supported real floating-point expressions, where conjugation is identity. Complex and typed-integer cases use their dedicated paths." },
];

pub(in super::super) const CONJ_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("conj"), slug: Some("conj"),
    summary: "Compute complex conjugates elementwise.",
    description: "`conj(X)` negates each imaginary component while preserving the input shape and numeric class; real values remain unchanged.",
    keywords: &["conj", "complex conjugate", "complex", "elementwise", "gpu"],
    related: &["abs", "angle", "complex", "double", "exp", "expm1", "gather", "gpuArray", "imag", "log", "log1p", "log2", "log10", "real", "sign", "single", "sqrt"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS,
    links: &[], media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
