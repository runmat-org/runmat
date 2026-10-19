use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationFaq,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinExample,
    BuiltinExampleCompatibility, BuiltinExampleHarness, BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`realmin()` returns the smallest positive normalized `double` value, while `realmin(\"single\")` returns the corresponding native `single` value. Subnormal values are smaller, so `realmin` is not the smallest positive nonzero value in either format.",
            "`realmin(\"like\", prototype)` returns a 1-by-1 value with the prototype's floating class, complexity, sparsity, and applicable placement. A complex result has a zero imaginary component. A distributed prototype retains its distribution scheme without materializing the source value.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: super::super::evidence::FLOATING_GPU_PARAGRAPHS,
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "double-minimum",
        title: "Query the normalized `double` minimum",
        program: "limit = realmin",
        display_output: Some("limit = 2.2251e-308"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(limit, \"double\"));\nassert(isfinite(limit));\nassert(limit > 0);" },
    },
    BuiltinExample {
        id: "single-minimum",
        title: "Query the normalized `single` minimum",
        program: "limit = realmin(\"single\")",
        display_output: Some("limit = single(1.1755e-38)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(limit, \"single\"));\nassert(isfinite(limit));\nassert(limit > single(0));" },
    },
    BuiltinExample {
        id: "like-sparse-single",
        title: "Match a sparse `single` prototype",
        program: "prototype = sparse(single(1));\nlimit = realmin(\"like\", prototype)",
        display_output: Some("limit is a 1-by-1 sparse single value"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(issparse(limit));\nassert(isa(limit, \"single\"));\nassert(isequal(size(limit), [1 1]));\nassert(full(limit) > single(0));" },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `realmin` include subnormal values?", answer: "No. It is the smallest positive normalized value for the selected floating-point class." },
    super::super::evidence::LIKE_FAQ,
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("realmin"),
    slug: Some("realmin"),
    summary: "Return the smallest positive normalized floating-point value.",
    description: "`realmin` returns the smallest positive normalized value in `double` or `single`. The default is `double`; the `like` form derives its representation and applicable placement from a floating-point prototype.",
    keywords: &["realmin", "floating point", "limits", "single", "double", "like", "gpu", "distributed"],
    related: &["realmax", "flintmax"],
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
