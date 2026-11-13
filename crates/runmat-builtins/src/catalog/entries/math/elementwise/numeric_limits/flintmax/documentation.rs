use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationFaq,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinExample,
    BuiltinExampleCompatibility, BuiltinExampleHarness, BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`flintmax()` returns 2^53 in `double`, and `flintmax(\"single\")` returns 2^24 in native `single`. Every integer from zero through that bound has an exact representation in the selected class. Above the bound, adjacent integers can round to the same floating-point value.",
            "`flintmax(\"like\", prototype)` returns a 1-by-1 value with the prototype's floating class, complexity, sparsity, and applicable placement. A complex result has a zero imaginary component. A distributed prototype retains its distribution scheme without materializing the source value.",
        ],
    },
    BuiltinDocumentationSection { heading: "GPU execution", paragraphs: super::super::evidence::FLOATING_GPU_PARAGRAPHS },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "double-consecutive-integer-limit",
        title: "Query the `double` consecutive-integer limit",
        program: "limit = flintmax",
        display_output: Some("limit = 9007199254740992"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(limit, \"double\"));\nassert(limit == 2^53);\nassert(limit + 1 == limit);\nassert(limit - 1 ~= limit);" },
    },
    BuiltinExample {
        id: "single-consecutive-integer-limit",
        title: "Query the `single` consecutive-integer limit",
        program: "limit = flintmax(\"single\")",
        display_output: Some("limit = single(16777216)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(limit, \"single\"));\nassert(limit == single(2^24));\nassert(limit + single(1) == limit);" },
    },
    BuiltinExample {
        id: "like-single-prototype",
        title: "Match a `single` prototype",
        program: "prototype = single([1 2 3]);\nlimit = flintmax(\"like\", prototype)",
        display_output: Some("limit = single(16777216)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(limit, \"single\"));\nassert(limit == single(16777216));\nassert(isequal(size(limit), [1 1]));" },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What does consecutive mean?", answer: "Every integer from zero through this value has an exact representation in the selected floating-point class. Above it, adjacent integers can map to the same floating-point value." },
    super::super::evidence::LIKE_FAQ,
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("flintmax"),
    slug: Some("flintmax"),
    summary: "Return the largest consecutive integer in a floating-point class.",
    description: "`flintmax` returns the largest consecutive integer representable in `double` or `single`: 2^53 for `double` and 2^24 for `single`. The `like` form derives its representation and applicable placement from a floating-point prototype.",
    keywords: &["flintmax", "floating point", "integer precision", "single", "double", "like", "gpu", "distributed"],
    related: &["realmax", "realmin", "intmax"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: &[],
    media: &[],
    evidence: super::super::evidence::FLOATING,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
