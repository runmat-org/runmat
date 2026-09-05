use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/magnitude_phase_sign/abs/mod.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "CPU and representation tests", location: "builtins::math::elementwise::magnitude_phase_sign::abs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Provider round-trip test", location: "builtins::math::elementwise::magnitude_phase_sign::abs::tests::provider::abs_gpu_provider_roundtrip" },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`Y = abs(X)` computes elementwise absolute values for real input and magnitudes for floating complex input while preserving shape. Complex single and double input returns the corresponding real floating class; typed complex integers are rejected.",
        "All eight real integer classes preserve class and shape. Unsigned values are unchanged; signed values are negated; the signed minimum saturates to the corresponding maximum.",
        "Sparse single and double input retains CSC storage and class. Exact sparse integer storage, logical-to-double input, and character-code input are RunMat compatibility extensions. Strings are unsupported, and duration/table/timetable overloads are not yet implemented.",
        "NaN propagates unchanged under IEEE rules.",
    ] },
    BuiltinDocumentationSection { heading: "GPU execution", paragraphs: &[
        "Supported providers compute real absolute values or complex magnitudes on the owning device. Complex magnitude returns a real resident tensor with the original shape.",
        "Typed unsupported operations gather through authoritative native storage and restore the result through the exact owner. Integer fallback retains the complete fixed-width class, including 64-bit values. Fusion can combine supported elementwise expressions without changing the value contract.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Take the absolute value of a scalar",
        program: "y = abs(-42)",
        display_output: Some("y = 42"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(y == 42);",
        },
    },
    BuiltinExample {
        id: "vector",
        title: "Take absolute values across a vector",
        program: "v = [-2 -1 0 1 2];\nresult = abs(v)",
        display_output: Some("result = [2 1 0 1 2]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(result, [2 1 0 1 2]));",
        },
    },
    BuiltinExample {
        id: "complex-magnitude",
        title: "Measure complex magnitudes",
        program: "z = [3+4i 1-1i];\nmagnitudes = abs(z)",
        display_output: Some("magnitudes = [5 1.4142]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(magnitudes - [5 sqrt(2)])) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu-array",
        title: "Take absolute values of provider-resident data",
        program: "G = gpuArray([-3 4; -5 6]);\npositive = abs(G);\nhost = gather(positive)",
        display_output: Some("host = [3 4; 5 6]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(host, [3 4; 5 6]));",
        },
    },
    BuiltinExample {
        id: "logical-extension",
        title: "Take magnitudes of logical values in RunMat mode",
        program: "mask = logical([0 1 0; 1 0 1]);\nnumeric = abs(mask)",
        display_output: Some("numeric = [0 1 0; 1 0 1]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(numeric, \"double\"));\nassert(isequal(numeric, [0 1 0; 1 0 1]));",
        },
    },
    BuiltinExample {
        id: "character-extension",
        title: "Take magnitudes of character code points in RunMat mode",
        program: "c = 'ABC';\ncodes = abs(c)",
        display_output: Some("codes = [65 66 67]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(codes, \"double\"));\nassert(isequal(codes, [65 66 67]));",
        },
    },
    BuiltinExample {
        id: "fused-expression",
        title: "Use `abs` in an elementwise expression",
        program: "x = linspace(-2, 2, 5);\ny = abs(x) + x.^2",
        display_output: Some("y = [6 2 0 2 6]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(y - [6 2 0 2 6])) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `abs` change NaN?", answer: "No. `abs(NaN)` returns NaN under IEEE arithmetic." },
    BuiltinDocumentationFaq { question: "What happens to complex values?", answer: "Floating complex input returns `sqrt(real(X).^2 + imag(X).^2)` in the corresponding real floating class." },
    BuiltinDocumentationFaq { question: "Can I call `abs` on strings?", answer: "No. Numeric input is supported; RunMat mode additionally enables logical and character extensions." },
    BuiltinDocumentationFaq { question: "Does `abs` preserve sparse arrays?", answer: "Real sparse single and double input retains CSC storage. RunMat mode additionally preserves exact typed sparse integers." },
    BuiltinDocumentationFaq { question: "Is GPU execution exact?", answer: "Execution follows IEEE behavior in the provider precision. Single-precision providers can differ slightly from CPU double results." },
    BuiltinDocumentationFaq { question: "How do I keep the result on the GPU?", answer: "Do not call `gather` unless host data is needed. The planner preserves residency when the provider supports the operation." },
    BuiltinDocumentationFaq { question: "Does `abs` allocate?", answer: "It returns a new value; fusion can avoid materializing an intermediate when safe." },
    BuiltinDocumentationFaq { question: "Can `abs` accept logical masks?", answer: "Only in RunMat compatibility mode, where logical values become double zero or one." },
];

pub(in super::super) const ABS_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("abs"), slug: Some("abs"),
    summary: "Compute absolute values and complex magnitudes elementwise.",
    description: "`abs(X)` returns absolute values for real input and magnitudes for floating complex input, with class-preserving saturating semantics for fixed-width integers.",
    keywords: &["abs", "absolute value", "magnitude", "complex", "integer", "sparse", "gpu"],
    related: &["angle", "conj", "double", "gather", "gpuArray", "hypot", "imag", "real", "sign", "single", "sqrt", "sum"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS,
    links: &[], media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
