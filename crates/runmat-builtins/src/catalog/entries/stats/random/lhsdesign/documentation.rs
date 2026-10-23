//! Conserved reference prose and independently asserted examples.
use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &["`n` and `p` must be positive integer-valued single or double scalars in MATLAB-compatible mode; native typed-integer dimensions are a RunMat extension and decode exactly.", "`\"Smooth\",\"on\"` jitters points uniformly within each stratum. `\"Smooth\",\"off\"` places points at stratum midpoints.", "`\"Criterion\"` supports `\"maximin\"`, `\"correlation\"`, and `\"none\"`; the correlation criterion uses midpoint strata, matching MATLAB's discrete correlation-optimized design behavior.", "`\"Iterations\",k` generates multiple candidate designs for optimized criteria and keeps the best scored candidate.", "The result uses column-major `n`-by-`p` shape and values in the open interval `(0,1)`.", "Native typed-integer Iterations and Smooth controls are separately gated RunMat extensions.", "Interactive gpuArray input is a RunMat-only host-fallback extension. Automatic internal residency may gather transparently without changing compatibility admission."],
    },
    BuiltinDocumentationSection {
        heading: "Execution and residency",
        paragraphs: &["Latin hypercube designs are generated on the host from RunMat's global random stream."],
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
        title: "Default maximin design",
        program: "rng(0);\nX = lhsdesign(20, 4)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(X),[20 4])); assert(all(X(:)>0 & X(:)<1)); for j=1:4; assert(isequal(sort(floor(20*X(:,j))), (0:19)')); end;",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Midpoint design with correlation criterion",
        program: "rng(0);\nX = lhsdesign(12, 3, \"Smooth\", \"off\", \"Criterion\", \"correlation\", \"Iterations\", 20)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(X),[12 3])); for j=1:3; assert(max(abs(sort(X(:,j)) - ((1:12)'-0.5)/12)) < 1e-12); end;",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Use exact integer controls with midpoint strata",
        program: "rng(0); X = lhsdesign(uint8(4),uint8(2),'Smooth','off','Criterion','none');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(X,'double')); assert(isequal(size(X),[4 2])); for j=1:2; assert(isequal(sort(X(:,j)),[0.125;0.375;0.625;0.875])); end;",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("lhsdesign"),
    slug: Some("lhsdesign"),
    summary: "Generate Latin hypercube sample designs.",
    description: "`lhsdesign(n,p)` returns an `n`-by-`p` Latin hypercube design with one sample in each stratum of every column.",
    keywords: &["lhsdesign", "latin hypercube", "design of experiments", "random", "statistics"],
    related: &[],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does lhsdesign use RunMat's rng state?",
        answer: "Yes. It consumes values from the same global random stream as rand, datasample, and randsample.",
    }],
    links: &[
        BuiltinDocumentationLink {
            label: "datasample",
            target: BuiltinDocumentationLinkTarget::Builtin("datasample"),
        },
        BuiltinDocumentationLink {
            label: "randsample",
            target: BuiltinDocumentationLinkTarget::Builtin("randsample"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/stats/random/lhsdesign.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/lhsdesign.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::stats::random::lhsdesign::tests",
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
