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
            "`gamrnd(a,b)` draws independent gamma-distributed samples with nonnegative shape parameter `a` and positive scale parameter `b`. Scalar parameters expand to the shape of an array parameter; two array parameters must have matching shapes.",
            "Without size arguments, the output uses the scalar-expanded parameter shape. A single scalar size produces a square matrix. A row-vector size or separate scalar dimensions selects an explicit shape, which must match nonscalar parameters. Nonpositive dimensions produce an empty array, and trailing singleton dimensions beyond the second are ignored.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes and RunMat extensions",
        paragraphs: &[
            "Documented parameters and sizes are dense real single or double values. If either distribution parameter is single, the result is single; otherwise it is double. Size controls select shape only.",
            "RunMat compatibility mode also accepts all eight fixed-width integer classes for either parameter or the size controls. Parameter values must cross the binary64 sampling boundary exactly. Integer size values are decoded directly from authoritative storage and do not select the result class.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Random state and resident arrays",
        paragraphs: &[
            "`gamrnd` uses RunMat's shared deterministic random state. Calling `rng(\"default\")` resets the sequence used by later calls, which makes tests and repeated experiments reproducible within the same runtime implementation.",
            "Floating provider parameters preserve output class and residency when their exact owner can represent the result. The current implementation samples through the host RNG and restores the result through that owner. Explicit resident intent must be preserved or the call fails; automatic placement may remain on the host when restoration is unavailable.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "matrix", title: "Generate a matrix of gamma samples", program: "rng(\"default\");\nr = gamrnd(3, 7, 2, 3)", display_output: Some("r is a nonnegative 2-by-3 double matrix"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(r), [2 3]));\nassert(isa(r, 'double'));\nassert(all(r(:) >= 0));" } },
    BuiltinExample { id: "parameter-arrays", title: "Draw from several gamma distributions", program: "rng(\"default\");\na = [1 2 3 4];\nr = gamrnd(a, 2)", display_output: Some("r is a nonnegative 1-by-4 array"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(r), size(a)));\nassert(all(r(:) >= 0));" } },
    BuiltinExample { id: "single", title: "Generate native single-precision samples", program: "rng(\"default\");\nr = gamrnd(single([1 2 3]), 2)", display_output: Some("r is a 1-by-3 single array"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(r, 'single'));\nassert(isequal(size(r), [1 3]));\nassert(all(r(:) >= 0));" } },
    BuiltinExample { id: "zero-shape-parameter", title: "Use a zero shape parameter", program: "rng(\"default\");\nr = gamrnd([0 2], 3)", display_output: Some("The first sample is zero"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(r(1) == 0);\nassert(r(2) >= 0);" } },
    BuiltinExample { id: "integer-extension", title: "Use exact fixed-width parameters and sizes", program: "rng(\"default\");\nr = gamrnd(uint16(2), uint8(3), uint16([2 2]))", display_output: Some("r is a nonnegative 2-by-2 double matrix"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(r, 'double'));\nassert(isequal(size(r), [2 2]));\nassert(all(r(:) >= 0));" } },
    BuiltinExample { id: "gpu-residency", title: "Preserve explicit provider residency", program: "rng(\"default\");\na = gpuArray([2 3 4]);\ngr = gamrnd(a, 1);\nr = gather(gr)", display_output: Some("gr remains a gpuArray"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(gr, 'gpuArray'));\nassert(isequal(size(r), [1 3]));\nassert(all(r(:) >= 0));" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How do a and b determine the distribution?", answer: "`a` is the nonnegative shape parameter and `b` is the positive scale parameter. Each output element uses the corresponding scalar-expanded pair." },
    BuiltinDocumentationFaq { question: "How is output size selected?", answer: "Without explicit dimensions, the output follows the parameter shape. One scalar size creates a square matrix; a row vector or separate scalar dimensions selects that shape." },
    BuiltinDocumentationFaq { question: "Which class does gamrnd return?", answer: "A single distribution parameter selects single output. Otherwise the result is double. Size values never select the class." },
    BuiltinDocumentationFaq { question: "Can a call be reproduced?", answer: "Yes. `rng(\"default\")` resets RunMat's shared deterministic random state before the call." },
    BuiltinDocumentationFaq { question: "Does gamrnd accept fixed-width integers?", answer: "RunMat mode accepts exact fixed-width parameters and structural sizes. MATLAB compatibility mode restricts the call to documented single/double inputs." },
    BuiltinDocumentationFaq { question: "What happens to gpuArray input?", answer: "RunMat samples through its shared host RNG and restores class and residency through the exact input owner. Explicit residency cannot be silently discarded." },
];

const RELATED: &[&str] = &["rng", "normrnd", "random", "gpuArray", "gather"];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "rng", target: BuiltinDocumentationLinkTarget::Builtin("rng") },
    BuiltinDocumentationLink { label: "normrnd", target: BuiltinDocumentationLinkTarget::Builtin("normrnd") },
    BuiltinDocumentationLink { label: "random", target: BuiltinDocumentationLinkTarget::Builtin("random") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/distribution_random/gamrnd.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Gamma random runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/distribution_random/gamrnd.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Parameters, sizes, classes, domains, extensions, RNG state, and errors", location: "crates/runmat-runtime/src/builtins/stats/random/distribution_random/gamrnd/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Exact-owner output restoration", location: "crates/runmat-runtime/src/builtins/stats/random/distribution_random/gamrnd/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU residency restoration", location: "crates/runmat-runtime/src/builtins/stats/random/distribution_random/gamrnd/tests.rs::wgpu_fallback_preserves_explicit_residency_and_precision" },
    ],
    notes: &[],
};

pub(super) const GAMRND_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("gamrnd"),
    slug: Some("gamrnd"),
    summary: "Generate gamma-distributed random samples.",
    description: "`gamrnd` draws from gamma distributions with scalar expansion, explicit shape forms, deterministic RunMat random-state integration, and validated provider residency.",
    keywords: &["gamrnd", "gamma", "random", "distribution", "shape", "scale", "gpu"],
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
