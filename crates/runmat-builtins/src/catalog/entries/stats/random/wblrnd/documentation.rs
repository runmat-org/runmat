use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Parameters and output shape",
        paragraphs: &[
            "`wblrnd(a,b)` draws independent Weibull samples with positive scale parameter `a` and positive shape parameter `b`. A scalar parameter expands to the shape of the other parameter; two nonscalar parameters must have matching shapes.",
            "Without size controls, the output follows the scalar-expanded parameter shape. A single scalar size creates a square matrix. A row-vector size or separate scalar dimensions selects an explicit shape, which must match each nonscalar parameter. Nonpositive dimensions produce an empty array, and trailing singleton dimensions beyond the second are ignored.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes and RunMat extensions",
        paragraphs: &[
            "Documented parameters and size controls are dense real single or double values. If either distribution parameter is single, the result is single; otherwise it is double. Size controls select shape only.",
            "RunMat compatibility mode also accepts all eight fixed-width integer classes for either parameter and for size controls, plus logical parameters and sizes. Integer parameters must be exactly representable at the binary64 sampling boundary. Integer size values are decoded directly from authoritative storage.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Sampling, random state, and resident arrays",
        paragraphs: &[
            "RunMat applies inverse-transform sampling using `a * (-log(U))^(1/b)` and its shared deterministic random state. Calling `rng(\"default\")` resets the sequence used by later calls.",
            "Floating provider parameters preserve output class and residency when their exact owner can represent the result. Sampling currently uses the host RNG and restores the result through that owner. Explicit provider residency must be preserved or the call fails. Distributed parameter arrays are materialized before sampling.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "matrix", title: "Generate a matrix of Weibull samples", program: "rng(\"default\");\nr = wblrnd(4, 3, 2, 3)", display_output: Some("r is a positive 2-by-3 double matrix"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(r), [2 3]));\nassert(isa(r, 'double'));\nassert(all(r(:) > 0));" } },
    BuiltinExample { id: "parameter-arrays", title: "Use several Weibull distributions", program: "rng(\"default\");\na = [1 2 3 4];\nb = [0.5 1 2 4];\nr = wblrnd(a, b)", display_output: Some("r follows the 1-by-4 parameter shape"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(r), size(a)));\nassert(all(r(:) > 0));" } },
    BuiltinExample { id: "single", title: "Generate native single-precision samples", program: "rng(\"default\");\nr = wblrnd(4, single([1 2 3]))", display_output: Some("r is a 1-by-3 single array"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(r, 'single'));\nassert(isequal(size(r), [1 3]));\nassert(all(r(:) > 0));" } },
    BuiltinExample { id: "reproducible", title: "Reset the shared random sequence", program: "rng(\"default\");\nfirst = wblrnd(2, 3, 1, 4);\nrng(\"default\");\nsecond = wblrnd(2, 3, 1, 4)", display_output: Some("first and second contain the same samples"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(first, second));" } },
    BuiltinExample { id: "integer-extension", title: "Use fixed-width parameters and sizes", program: "rng(\"default\");\nr = wblrnd(uint16(4), uint8(3), uint16([2 2]))", display_output: Some("r is a positive 2-by-2 double matrix"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(r, 'double'));\nassert(isequal(size(r), [2 2]));\nassert(all(r(:) > 0));" } },
    BuiltinExample { id: "logical-extension", title: "Use logical parameters and a logical size", program: "rng(\"default\");\nr = wblrnd(true, true, true)", display_output: Some("r is a positive double scalar"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(r, 'double'));\nassert(isequal(size(r), [1 1]));\nassert(r > 0);" } },
    BuiltinExample { id: "gpu-residency", title: "Preserve explicit provider residency", program: "rng(\"default\");\na = gpuArray(single([2 3 4]));\ngr = wblrnd(a, 2);\nr = gather(gr)", display_output: Some("gr remains a single-precision gpuArray"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(gr, 'gpuArray'));\nassert(isa(r, 'single'));\nassert(isequal(size(r), [1 3]));\nassert(all(r(:) > 0));" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How do a and b determine the distribution?", answer: "`a` sets the scale and `b` sets the shape. Both must be positive, and each output uses the corresponding scalar-expanded pair." },
    BuiltinDocumentationFaq { question: "How is output size selected?", answer: "Without explicit dimensions, the output follows the parameter shape. One scalar size creates a square matrix; a row vector or separate scalars select an explicit shape." },
    BuiltinDocumentationFaq { question: "Which class does wblrnd return?", answer: "A single distribution parameter selects single output. Otherwise the result is double. Size controls never select the class." },
    BuiltinDocumentationFaq { question: "Can a call be reproduced?", answer: "Yes. `rng(\"default\")` resets RunMat's shared deterministic random state before the call." },
    BuiltinDocumentationFaq { question: "Does wblrnd accept fixed-width integers or logical values?", answer: "RunMat mode accepts exact fixed-width parameters, structural integer sizes, and logical parameters or sizes. MATLAB compatibility mode restricts the call to documented single and double values." },
    BuiltinDocumentationFaq { question: "What happens to a gpuArray input?", answer: "RunMat samples through its shared host RNG and restores class and residency through the exact input owner. Explicit residency cannot be silently discarded." },
];

const RELATED: &[&str] = &["rng", "wblinv", "random", "gpuArray", "gather"];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "rng", target: BuiltinDocumentationLinkTarget::Builtin("rng") },
    BuiltinDocumentationLink { label: "wblinv", target: BuiltinDocumentationLinkTarget::Builtin("wblinv") },
    BuiltinDocumentationLink { label: "random", target: BuiltinDocumentationLinkTarget::Builtin("random") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/distribution_random/wblrnd.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Weibull random runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/distribution_random/wblrnd.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Parameters, shapes, classes, domains, extensions, RNG state, and errors", location: "crates/runmat-runtime/src/builtins/stats/random/distribution_random/wblrnd/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Exact-owner output restoration", location: "crates/runmat-runtime/src/builtins/stats/random/distribution_random/wblrnd/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU residency restoration", location: "crates/runmat-runtime/src/builtins/stats/random/distribution_random/wblrnd/tests.rs::wgpu_fallback_preserves_explicit_residency_and_precision" },
    ],
    notes: &[],
};

pub(super) const WBLRND_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("wblrnd"),
    slug: Some("wblrnd"),
    summary: "Generate Weibull-distributed random samples.",
    description: "`wblrnd` draws from Weibull distributions with scalar parameter expansion, explicit shape forms, deterministic RunMat random-state integration, and validated provider residency.",
    keywords: &["wblrnd", "weibull", "random", "distribution", "statistics", "scale", "shape", "gpu"],
    related: RELATED,
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
