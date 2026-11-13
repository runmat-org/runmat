//! Conserved reference prose and independently asserted examples.
use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Samples with replacement by default.",
            "`\"Replace\",false` samples without replacement and requires `k` to be no larger than the sampled dimension.",
            "`\"Weights\",w` samples proportionally to nonnegative weights with at least one positive value.",
            "Documented ordinary populations include single, double, logical, character, string, and table data. RunMat currently supports numeric, logical, character, string, and cell populations; table and tall-array paths remain unimplemented.",
            "Typed-integer population data is a RunMat extension sampled directly from authoritative integer storage, preserving every selected value and its integer class.",
            "Typed-integer `k`, typed-integer `dim`, typed-integer weights, logical weights, numeric `Replace`, and resident inputs are independently gated RunMat extensions. Compatibility-mode `k`, `dim`, and weights use documented single or double forms, and `Replace` uses logical.",
            "The default sampled dimension is the first nonsingleton dimension.",
            "The documented `RandStream` calling form, tables, and tall-array execution are not yet implemented.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution and residency",
        paragraphs: &["Resident inputs are a gated RunMat extension and are gathered before sampling until provider sampling hooks exist; sampled output is currently host-resident."],
    },
    BuiltinDocumentationSection {
        heading: "Random state and fusion",
        paragraphs: &["Random generation and RNG state control are stateful boundaries and are not fused into elementwise kernels."],
    },
    BuiltinDocumentationSection {
        heading: "Storage precision",
        paragraphs: &["Supported sample/storage class declaration: logical. See the distribution, integer-extension, and residency rules above."],
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
        paragraphs: &["The leading RandStream form, table populations, and tall-array execution are not implemented."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Sample rows",
        program: "rng(0);\nX = reshape(1:20, 5, 4);\nY = datasample(X, 5)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(Y),[5 4])); assert(all(ismember(Y(:,1),X(:,1))));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Sample columns without replacement",
        program: "rng(0);\nX = reshape(1:20, 5, 4);\nY = datasample(X, 3, 2, \"Replace\", false)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(Y),[5 3])); assert(numel(unique(Y(1,:)))==3); assert(all(ismember(Y(1,:),X(1,:))));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Return sampled indices",
        program: "rng(0);\nX = reshape(1:20, 5, 4);\n[Y,idx] = datasample(X, 10)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(idx),[10 1])); assert(all(idx>=1 & idx<=5)); assert(isequal(Y,X(idx,:)));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Preserve integer values beyond binary64 precision",
        program: "rng(0); base = bitshift(uint64(1),53); X = base + uint64([1 3 5]); [Y,idx] = datasample(X,uint8(3));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(Y,'uint64')); assert(isequal(size(Y),[1 3])); assert(isa(idx,'double')); assert(isequal(Y,X(idx)));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("datasample"),
    slug: Some("datasample"),
    summary: "Randomly sample from data with or without replacement.",
    description: "`datasample(data,k)` samples `k` observations from `data`. `datasample(data,k,dim)` samples along a selected dimension. `[y,idx] = datasample(...)` also returns one-based sampled indices.",
    keywords: &["datasample", "random", "sample", "replacement", "weights", "statistics"],
    related: &[],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does datasample change RNG state?",
        answer: "Yes. It uses RunMat's global random stream, like other random builtins.",
    }],
    links: &[
        BuiltinDocumentationLink {
            label: "randsample",
            target: BuiltinDocumentationLinkTarget::Builtin("randsample"),
        },
        BuiltinDocumentationLink {
            label: "randperm",
            target: BuiltinDocumentationLinkTarget::Builtin("randperm"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/stats/random/sampling.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/sampling.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "unit",
            location: "builtins::stats::random::sampling::tests",
        }],
        notes: &["Integer compatibility and placement are declared in the canonical contract. Examples assert shape, class, bounds, or state replay rather than fixed random transcripts."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
