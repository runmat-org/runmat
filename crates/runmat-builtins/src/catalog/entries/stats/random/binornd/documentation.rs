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
            "`binornd(n,p)` draws independent binomial samples. Each value of `n` must be a positive integer trial count, and each probability in `p` must be between zero and one, inclusive. A scalar parameter expands to the shape of the other parameter; two nonscalar parameters must have matching shapes.",
            "Without size controls, the output follows the scalar-expanded parameter shape. A single scalar size creates a square matrix. A row-vector size or separate scalar dimensions selects an explicit shape, which must match every nonscalar parameter. Nonpositive dimensions produce an empty array, and trailing singleton dimensions beyond the second are ignored.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes and RunMat extensions",
        paragraphs: &[
            "The documented parameter and size classes are dense real single and double. If either distribution parameter is single, the result is single; otherwise it is double. Size controls select shape only.",
            "RunMat compatibility mode also accepts all eight fixed-width integer classes for either parameter and for size controls, plus logical values for `n` or `p`. Integer parameters must be exactly representable at the binary64 sampling boundary. Integer size values are decoded directly from authoritative storage.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Random state, provider arrays, and distributed arrays",
        paragraphs: &[
            "`binornd` uses RunMat's shared deterministic random state. Calling `rng(\"default\")` resets the sequence used by later calls.",
            "Floating provider parameters preserve output class and residency when their exact owner can represent the result. Sampling currently uses the host RNG and restores the result through that owner. Explicit provider residency must be preserved or the call fails.",
            "Distributed parameter arrays are materialized before sampling, so the returned array is host-resident. Values and documented scalar-expansion behavior are preserved across that boundary.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "matrix", title: "Generate a matrix of binomial samples", program: "rng(\"default\");\nr = binornd(10, 0.5, 2, 3)", display_output: Some("r is a 2-by-3 double matrix with values from 0 through 10"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(r), [2 3]));\nassert(isa(r, 'double'));\nassert(all(r(:) >= 0));\nassert(all(r(:) <= 10));\nassert(all(r(:) == fix(r(:))));" } },
    BuiltinExample { id: "parameter-arrays", title: "Use several trial counts and probabilities", program: "rng(\"default\");\nn = [2 4 6 8];\np = [0.2 0.4 0.6 0.8];\nr = binornd(n, p)", display_output: Some("r follows the 1-by-4 parameter shape"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(r), size(n)));\nassert(all(r(:) >= 0));\nassert(all(r(:) <= n(:)));\nassert(all(r(:) == fix(r(:))));" } },
    BuiltinExample { id: "single", title: "Generate native single-precision samples", program: "rng(\"default\");\nr = binornd(10, single([0.2 0.5 0.8]))", display_output: Some("r is a 1-by-3 single array"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(r, 'single'));\nassert(isequal(size(r), [1 3]));\nassert(all(r(:) >= 0));\nassert(all(r(:) <= 10));" } },
    BuiltinExample { id: "endpoints", title: "Use deterministic probability endpoints", program: "rng(\"default\");\nr = binornd([3 5], [0 1])", display_output: Some("r is [0 5]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(r, [0 5]));" } },
    BuiltinExample { id: "integer-extension", title: "Use fixed-width parameters and sizes", program: "rng(\"default\");\nr = binornd(uint16(8), uint8(1), uint16([2 2]))", display_output: Some("r is a 2-by-2 double matrix containing 8"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(r, 'double'));\nassert(isequal(size(r), [2 2]));\nassert(all(r(:) == 8));" } },
    BuiltinExample { id: "gpu-residency", title: "Preserve explicit provider residency", program: "rng(\"default\");\np = gpuArray(single([0.2 0.5 0.8]));\ngr = binornd(10, p);\nr = gather(gr)", display_output: Some("gr remains a single-precision gpuArray"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(gr, 'gpuArray'));\nassert(isa(r, 'single'));\nassert(isequal(size(r), [1 3]));\nassert(all(r(:) >= 0));\nassert(all(r(:) <= 10));" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Can n be fractional?", answer: "No. Every trial count must be a positive integer." },
    BuiltinDocumentationFaq { question: "How are array parameters combined?", answer: "A scalar expands to the other parameter's shape. Two nonscalar parameters must have the same shape; `binornd` does not apply general singleton-dimension expansion." },
    BuiltinDocumentationFaq { question: "Which class does binornd return?", answer: "A single `n` or `p` selects single output. Otherwise the result is double. Size controls never select the class." },
    BuiltinDocumentationFaq { question: "Can a call be reproduced?", answer: "Yes. Call `rng(\"default\")` before the operation to reset RunMat's shared deterministic random state." },
    BuiltinDocumentationFaq { question: "Does binornd accept fixed-width integers?", answer: "RunMat mode accepts exact fixed-width parameters and structural sizes. MATLAB compatibility mode restricts the call to documented single and double inputs." },
    BuiltinDocumentationFaq { question: "What happens to a gpuArray input?", answer: "RunMat samples through its shared host RNG and restores class and residency through the exact input owner. Explicit residency cannot be silently discarded." },
];

const RELATED: &[&str] = &["rng", "random", "binocdf", "gpuArray", "gather"];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "rng", target: BuiltinDocumentationLinkTarget::Builtin("rng") },
    BuiltinDocumentationLink { label: "random", target: BuiltinDocumentationLinkTarget::Builtin("random") },
    BuiltinDocumentationLink { label: "binocdf", target: BuiltinDocumentationLinkTarget::Builtin("binocdf") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/distribution_random/binornd.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Binomial random runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/distribution_random/binornd.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Parameters, shapes, classes, domains, extensions, RNG state, and errors", location: "crates/runmat-runtime/src/builtins/stats/random/distribution_random/binornd/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Exact-owner output restoration", location: "crates/runmat-runtime/src/builtins/stats/random/distribution_random/binornd/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU residency restoration", location: "crates/runmat-runtime/src/builtins/stats/random/distribution_random/binornd/tests.rs::wgpu_fallback_preserves_explicit_residency_and_precision" },
    ],
    notes: &[],
};

pub(super) const BINORND_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("binornd"),
    slug: Some("binornd"),
    summary: "Generate binomially distributed random samples.",
    description: "`binornd` draws from binomial distributions with scalar parameter expansion, explicit shape forms, deterministic RunMat random-state integration, and validated provider residency.",
    keywords: &["binornd", "binomial", "random", "distribution", "trials", "probability", "gpu"],
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
