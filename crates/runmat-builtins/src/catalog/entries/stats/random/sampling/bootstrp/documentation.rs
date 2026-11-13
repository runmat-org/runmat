//! Conserved reference prose and independently asserted examples.
use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Each row of `bootstat` contains one bootstrap replicate. Nonscalar `bootfun` outputs are flattened into one row per replicate.",
            "A single vector input is sampled along its first nonsingleton dimension; multiple nonscalar inputs are sampled by rows and must have matching row counts.",
            "`bootfun` can be a function handle or an empty array. Empty `bootfun` skips callback evaluation and returns an empty double `bootstat` with `bootsam` available as the second output; a text function name is a gated RunMat extension.",
            "`\"Weights\",w` samples rows proportionally to nonnegative weights with at least one positive value.",
            "`\"Options\",opts` accepts a statset-style options struct for compatibility. Parallel execution options are accepted but currently evaluated serially.",
            "Public numeric controls, data, and weights are single or double; logical, string, char, and cell sample data is also supported. Integer `nboot`, integer sample data, integer weights, logical `nboot`, and logical weights are independently gated RunMat extensions.",
            "Integer sample data is indexed without floating conversion. Homogeneous numeric or logical callback output determines `bootstat` class, including native single and every integer class; `bootsam` is always double.",
            "Interactive gpuArray input is independently gated and gathered only after compatibility admission.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution and residency",
        paragraphs: &["Interactive gpuArray input is a RunMat-only extension. After compatibility admission, RunMat gathers exact host storage before resampling and callback evaluation; outputs remain host-resident."],
    },
    BuiltinDocumentationSection {
        heading: "Random state and fusion",
        paragraphs: &["Random generation and RNG state control are stateful boundaries and are not fused into elementwise kernels."],
    },
    BuiltinDocumentationSection {
        heading: "Storage precision",
        paragraphs: &["Supported sample/storage class declaration: integer. See the distribution, integer-extension, and residency rules above."],
    },
    BuiltinDocumentationSection {
        heading: "Storage precision",
        paragraphs: &["Supported sample/storage class declaration: f32. See the distribution, integer-extension, and residency rules above."],
    },
    BuiltinDocumentationSection {
        heading: "Storage precision",
        paragraphs: &["Supported sample/storage class declaration: f64. See the distribution, integer-extension, and residency rules above."],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation limits",
        paragraphs: &["Options may accept parallel controls, but bootstrap callback execution remains serial."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Bootstrap a mean",
        program: "rng(0);\nx = (1:20)';\nb = bootstrp(1000, @mean, x)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(b),[1000 1])); assert(all(b>=min(x) & b<=max(x)));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Return sample indices",
        program: "rng(0);\nx = (1:20)';\n[b,bootsam] = bootstrp(200, @median, x)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(b),[200 1])); assert(isequal(size(bootsam),[20 200])); assert(all(bootsam(:)>=1 & bootsam(:)<=20)); assert(isequal(b(1),median(x(bootsam(:,1)))));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Bootstrap a two-input statistic",
        program: "rng(0);\nx = (1:20)'; y = 2*x + 3;\nb = bootstrp(500, @corr, x, y)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(b),[500 1])); assert(all(isnan(b) | abs(b-1)<1e-10));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Retain the callback's native integer class",
        program: "rng(0); base = bitshift(uint64(1),53); X = base + uint64([1;3;5]); [B,I] = bootstrp(uint8(4),@max,X);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(B,'uint64')); assert(isa(I,'double')); assert(isequal(size(B),[4 1])); assert(isequal(size(I),[3 4])); for j=1:4; assert(B(j)==max(X(I(:,j)))); end;",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("bootstrp"),
    slug: Some("bootstrp"),
    summary: "Bootstrap samples and evaluate a statistic.",
    description: "`bootstrp(nboot,bootfun,d)` draws `nboot` bootstrap samples with replacement from `d` and applies `bootfun` to each sample. `bootstrp(nboot,bootfun,d1,...,dN)` resamples rows jointly across nonscalar data arguments while scalar arguments are passed through unchanged. `[bootstat,bootsam] = bootstrp(...)` also returns one-based bootstrap sample indices.",
    keywords: &["bootstrp", "bootstrap", "resampling", "statistics", "weights"],
    related: &[],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does bootstrp use the global random stream?",
            answer: "Yes. It uses RunMat's global random stream, like datasample and randsample.",
        },
        BuiltinDocumentationFaq {
            question: "Does bootstrp run callbacks in parallel?",
            answer: "No. RunMat accepts the Options name-value for MATLAB compatibility, but callback evaluation is currently serial.",
        },
        BuiltinDocumentationFaq {
            question: "Can bootstrp sample integer arrays?",
            answer: "In RunMat mode, yes: all eight integer classes are sampled exactly and the callback receives the same class. MATLAB-compatible mode rejects this extension because the public nongrouping numeric data domain is single or double.",
        },
        BuiltinDocumentationFaq {
            question: "What class does bootstat use?",
            answer: "The callback's homogeneous numeric or logical output class determines bootstat, while bootsam is double. Inconsistent callback output sizes or classes raise an error.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "datasample",
            target: BuiltinDocumentationLinkTarget::Builtin("datasample"),
        },
        BuiltinDocumentationLink {
            label: "randsample",
            target: BuiltinDocumentationLinkTarget::Builtin("randsample"),
        },
        BuiltinDocumentationLink {
            label: "mean",
            target: BuiltinDocumentationLinkTarget::Builtin("mean"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/stats/random/sampling.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/sampling.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::stats::random::sampling::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "integration",
                location: "crates/runmat-vm/tests/stats_ml.rs",
            },
        ],
        notes: &["Integer compatibility and placement are declared in the canonical contract. Examples assert shape, class, bounds, or state replay rather than fixed random transcripts."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
