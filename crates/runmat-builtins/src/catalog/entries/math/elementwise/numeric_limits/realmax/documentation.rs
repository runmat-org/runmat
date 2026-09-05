use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationFaq,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinExample,
    BuiltinExampleCompatibility, BuiltinExampleHarness, BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`realmax()` returns the largest finite `double` value, while `realmax(\"single\")` returns the largest finite native `single` value. The result is finite; the next sufficiently large operation may overflow to infinity.",
            "`realmax(\"like\", prototype)` returns a 1-by-1 value with the prototype's floating class, complexity, sparsity, and applicable placement. A complex result has a zero imaginary component. A distributed prototype retains its distribution scheme without materializing the source value.",
        ],
    },
    BuiltinDocumentationSection { heading: "GPU execution", paragraphs: super::super::evidence::FLOATING_GPU_PARAGRAPHS },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "double-maximum",
        title: "Query the largest finite `double` value",
        program: "limit = realmax",
        display_output: Some("limit = 1.7977e+308"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(limit, \"double\"));\nassert(isfinite(limit));\nassert(isinf(limit * 2));" },
    },
    BuiltinExample {
        id: "single-maximum",
        title: "Query the largest finite `single` value",
        program: "limit = realmax(\"single\")",
        display_output: Some("limit = single(3.4028e+38)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(limit, \"single\"));\nassert(isfinite(limit));\nassert(isinf(limit * single(2)));" },
    },
    BuiltinExample {
        id: "like-complex-single",
        title: "Match a complex `single` prototype",
        program: "prototype = complex(single(1), single(2));\nlimit = realmax(\"like\", prototype)",
        display_output: Some("limit is a 1-by-1 complex single value"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(limit, \"single\"));\nassert(~isreal(limit));\nassert(real(limit) == realmax(\"single\"));\nassert(imag(limit) == single(0));" },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Is the result infinite?",
        answer: "No. It is the largest finite value in the selected floating-point class.",
    },
    super::super::evidence::LIKE_FAQ,
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("realmax"),
    slug: Some("realmax"),
    summary: "Return the largest finite floating-point value.",
    description: "`realmax` returns the largest finite value in `double` or `single`. The default is `double`; the `like` form derives its representation and applicable placement from a floating-point prototype.",
    keywords: &["realmax", "floating point", "limits", "single", "double", "like", "gpu", "distributed"],
    related: &["realmin", "flintmax"],
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
