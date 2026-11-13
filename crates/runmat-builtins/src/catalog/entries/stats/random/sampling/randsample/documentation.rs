//! Conserved reference prose and independently asserted examples.
use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &["Samples without replacement by default.", "When the first argument is a numeric scalar, it is treated as the upper limit `n`.", "Population vectors preserve their row or column orientation.", "Weights are supported for sampling with replacement, matching MATLAB's documented restriction.", "The documented numeric range, population, count, and weight classes are single and double, while replacement is logical. Typed-integer range limits, populations, counts, replacement flags, and weights are five independently gated RunMat-mode extensions.", "Typed-integer populations preserve exact native class, orientation, and supported owner residency; weights cross a checked binary64 probability boundary, while range limits, counts, and replacement flags remain structural.", "Numeric, logical, string, and char population vectors are supported."],
    },
    BuiltinDocumentationSection {
        heading: "Execution and residency",
        paragraphs: &["GPU inputs are gathered to host for compatibility until provider sampling hooks exist."],
    },
    BuiltinDocumentationSection {
        heading: "Random state and fusion",
        paragraphs: &["Random generation and RNG state control are stateful boundaries and are not fused into elementwise kernels."],
    },
    BuiltinDocumentationSection {
        heading: "Storage precision",
        paragraphs: &["Supported sample/storage class declaration: f64. See the distribution, integer-extension, and residency rules above."],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation limits",
        paragraphs: &["Only the range and population calling forms described above are implemented. A leading RandStream argument is not implemented."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Sample unique integers",
        program: "rng(0);\ny = randsample(20, 10)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(y),[10 1])); assert(numel(unique(y))==10); assert(all(y>=1 & y<=20));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Sample with replacement and weights",
        program: "rng(0);\npopulation = [10 20 30]; weights = [1 2 3];\ny = randsample(population, 5, true, weights)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(y),[1 5])); assert(all(ismember(y,population)));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Preserve a native integer population without replacement",
        program: "rng(0); base = bitshift(uint64(1),53); population = base + uint64([1 3 5]); Y = randsample(population,uint8(3));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(Y,'uint64')); assert(isequal(size(Y),[1 3])); assert(isequal(sort(Y),population));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("randsample"),
    slug: Some("randsample"),
    summary: "Randomly sample from an integer range or population vector.",
    description: "`randsample(n,k)` samples from integers `1:n`. `randsample(population,k)` samples from a population vector. Replacement and weights are supported with `randsample(...,replacement,w)`.",
    keywords: &["randsample", "random", "sample", "replacement", "weights", "statistics"],
    related: &[],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "How is randsample different from datasample?",
        answer: "`randsample(n,k)` is specialized for integer ranges and population vectors. Use `datasample` to sample along dimensions of arrays.",
    }],
    links: &[
        BuiltinDocumentationLink {
            label: "datasample",
            target: BuiltinDocumentationLinkTarget::Builtin("datasample"),
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
