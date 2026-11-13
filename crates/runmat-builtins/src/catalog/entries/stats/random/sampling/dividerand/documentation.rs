//! Conserved reference prose and independently asserted examples.
use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &["`Q` must be a nonnegative integer scalar. All eight typed integer scalar classes are decoded exactly before allocation.", "`Q` is capped at 10,000,000 so index partitions fail cleanly before excessive host allocation.", "Ratios must be finite nonnegative scalar values, and at least one ratio must be positive.", "RunMat normalizes ratios and uses largest remainders for allocation. The public documentation does not specify this tie-breaking algorithm.", "Returned indices are one-based row vectors and together form a random permutation of `1:Q`.", "Empty partitions are returned as `1-by-0` double tensors.", "MathWorks marks `dividerand` for removal in the compatibility target; RunMat retains it for backward compatibility."],
    },
    BuiltinDocumentationSection {
        heading: "Execution and residency",
        paragraphs: &["No GPU capability is documented. Host output is standard; resident scalar arguments are accepted only as a named RunMat extension."],
    },
    BuiltinDocumentationSection {
        heading: "Random state and fusion",
        paragraphs: &["Random generation and RNG state control are stateful boundaries and are not fused into elementwise kernels."],
    },
    BuiltinDocumentationSection {
        heading: "Storage precision",
        paragraphs: &["Supported sample/storage class declaration: f64. See the distribution, integer-extension, and residency rules above."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Default split",
        program: "rng(0);\n[trainInd,valInd,testInd] = dividerand(100)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(sort([trainInd valInd testInd]),1:100)); assert(numel(trainInd)==70); assert(numel(valInd)==15); assert(numel(testInd)==15);",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Explicit split ratios",
        program: "rng(0);\n[trainInd,valInd,testInd] = dividerand(3000, 0.6, 0.2, 0.2)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(sort([trainInd valInd testInd]),1:3000)); assert(numel(trainInd)==1800); assert(numel(valInd)==600); assert(numel(testInd)==600);",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Partition an exact integer count",
        program: "rng(0); [a,b,c] = dividerand(uint16(10),0.6,0.2,0.2);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(a,'double')); assert(isequal(size(a),[1 6])); assert(isequal(size(b),[1 2])); assert(isequal(size(c),[1 2])); assert(isequal(sort([a b c]),1:10));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dividerand"),
    slug: Some("dividerand"),
    summary: "Divide target indices randomly into training, validation, and test sets.",
    description: "`dividerand(Q,trainRatio,valRatio,testRatio)` randomly permutes the one-based target indices `1:Q` and divides them into training, validation, and test row vectors. `dividerand(Q)` uses the documented default ratios `0.7`, `0.15`, and `0.15`.",
    keywords: &["dividerand", "random", "partition", "train", "validation", "test", "statistics", "machine-learning"],
    related: &[],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does dividerand respect rng?",
        answer: "Yes. The random permutation is drawn from RunMat's global random generator, so `rng('default')` and numeric seeds make results reproducible.",
    }],
    links: &[
        BuiltinDocumentationLink {
            label: "rng",
            target: BuiltinDocumentationLinkTarget::Builtin("rng"),
        },
        BuiltinDocumentationLink {
            label: "randperm",
            target: BuiltinDocumentationLinkTarget::Builtin("randperm"),
        },
        BuiltinDocumentationLink {
            label: "MathWorks dividerand documentation",
            target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/deeplearning/ref/dividerand.html"),
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
    status: Some(BuiltinDocumentationStatus::Stable),
};
