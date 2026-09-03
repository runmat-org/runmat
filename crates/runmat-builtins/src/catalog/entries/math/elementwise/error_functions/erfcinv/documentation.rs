use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Inverse complementary error-function values",
        paragraphs: &[
            "`erfcinv(X)` returns the real value `Y` for which the complementary error function satisfies `erfc(Y) = X`. It evaluates dense real single or double input element by element and preserves class and shape.",
            "Inputs from 0 through 2 produce real results. The endpoints map to positive and negative infinity, 1 maps to zero, values outside the interval return NaN, and NaN propagates. Logical, fixed-width integer, character, string, complex, object, and sparse input is rejected.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Accelerated and distributed execution",
        paragraphs: &[
            "A real floating provider input can use its owner's `unary_erfcinv` operation. RunMat validates shape, storage, precision, owner, device, and non-aliasing, then restores the input's residency intent on the result. Only a typed unsupported result enters host fallback.",
            "Fallback gathers through the exact owner, evaluates the same inverse, and restores the result to that owner with the original precision and residency intent. Distributed arrays use partition-local unary mapping.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "values", title: "Invert complementary error-function values", program: "X = [0.25 0.5 1 1.5 1.75];\nY = erfcinv(X)", display_output: Some("erfc(Y) equals X"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(1 - erf(Y) - X)) < 1e-11);" } },
    BuiltinExample { id: "endpoints", title: "Inspect endpoint behavior", program: "Y = erfcinv([0 1 2])", display_output: Some("Y = [Inf 0 -Inf]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isinf(Y(1)) && Y(1) > 0);\nassert(Y(2) == 0);\nassert(isinf(Y(3)) && Y(3) < 0);" } },
    BuiltinExample { id: "shape", title: "Preserve array shape", program: "X = reshape([0.5 1 1.5 2], [2 2]);\nY = erfcinv(X)", display_output: Some("Y is a 2-by-2 double matrix"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(Y), [2 2]));\nassert(max(abs(1 - erf(Y(:)) - X(:))) < 1e-11);" } },
    BuiltinExample { id: "gpu-residency", title: "Keep inverse values resident", program: "G = gpuArray([0.5 1.5]);\nGy = erfcinv(G);\nY = gather(Gy)", display_output: Some("Y contains opposite-signed values of equal magnitude"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Gy, 'gpuArray'));\nassert(abs(Y(1) + Y(2)) < 1e-5);\nassert(abs(1 - erf(Y(1)) - 0.5) < 1e-5);" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How is erfcinv related to erf?", answer: "It solves `1 - erf(Y) = X` for real Y. Within the supported interval, applying that relation to the result reconstructs X up to numerical precision." },
    BuiltinDocumentationFaq { question: "What happens outside the real domain?", answer: "Real inputs less than 0 or greater than 2 return NaN; the input remains a valid real floating value even though no finite real inverse exists." },
    BuiltinDocumentationFaq { question: "Can erfcinv keep a provider input resident?", answer: "Yes. A valid direct result remains resident, and an unsupported provider operation uses exact-owner gather and restoration." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "erf", target: BuiltinDocumentationLinkTarget::Builtin("erf") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/error_functions/erfcinv") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Inverse complementary error-function runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/error_functions/erfcinv") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Reference values, tails, endpoints, shapes, classes, and rejected representations", location: "crates/runmat-runtime/src/builtins/math/elementwise/error_functions/erfcinv.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Direct execution, fallback, ownership, and output validation", location: "crates/runmat-runtime/src/builtins/math/elementwise/error_functions/erfcinv.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU inverse parity and residency", location: "crates/runmat-runtime/src/builtins/math/elementwise/error_functions/erfcinv.rs" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("erfcinv"),
    slug: Some("erfcinv"),
    summary: "Evaluate the inverse complementary error function element by element.",
    description: "`erfcinv` evaluates dense real single or double values, preserves their class and shape, and supports validated provider-resident execution.",
    keywords: &["erfcinv", "inverse complementary error function", "special function", "elementwise", "single", "double", "gpu"],
    related: &["erf", "gpuArray", "gather"],
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
