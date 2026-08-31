use super::FLOATING_EVIDENCE;
use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationFaq,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinExample,
    BuiltinExampleCompatibility, BuiltinExampleHarness, BuiltinExampleVerification,
};

const FLOATING_GPU_PARAGRAPHS: &[&str] = &[
    "Class-name forms return host scalars. With a floating `gpuArray` prototype, the `like` form creates only the scalar result on the prototype's registered provider and device. It preserves precision, complexity, storage kind, and explicit-placement provenance without downloading the prototype.",
    "Numeric-limit queries are scalar construction boundaries rather than elementwise or reduction kernels, so they are not fused with neighboring operations.",
];

const REALMIN_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`realmin()` returns the smallest positive normalized `double` value, while `realmin(\"single\")` returns the corresponding native `single` value. Subnormal values are smaller, so `realmin` is not the smallest positive nonzero value in either format.",
            "`realmin(\"like\", prototype)` returns a 1-by-1 value with the prototype's floating class, complexity, sparsity, and applicable placement. A complex result has a zero imaginary component. A distributed prototype retains its distribution scheme without materializing the source value.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: FLOATING_GPU_PARAGRAPHS,
    },
];

const REALMAX_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`realmax()` returns the largest finite `double` value, while `realmax(\"single\")` returns the largest finite native `single` value. The result is finite; the next sufficiently large operation may overflow to infinity.",
            "`realmax(\"like\", prototype)` returns a 1-by-1 value with the prototype's floating class, complexity, sparsity, and applicable placement. A complex result has a zero imaginary component. A distributed prototype retains its distribution scheme without materializing the source value.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: FLOATING_GPU_PARAGRAPHS,
    },
];

const FLINTMAX_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`flintmax()` returns 2^53 in `double`, and `flintmax(\"single\")` returns 2^24 in native `single`. Every integer from zero through that bound has an exact representation in the selected class. Above the bound, adjacent integers can round to the same floating-point value.",
            "`flintmax(\"like\", prototype)` returns a 1-by-1 value with the prototype's floating class, complexity, sparsity, and applicable placement. A complex result has a zero imaginary component. A distributed prototype retains its distribution scheme without materializing the source value.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: FLOATING_GPU_PARAGRAPHS,
    },
];

const REALMIN_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "double-minimum",
        title: "Query the normalized `double` minimum",
        program: "limit = realmin",
        display_output: Some("limit = 2.2251e-308"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(limit, \"double\"));\nassert(isfinite(limit));\nassert(limit > 0);",
        },
    },
    BuiltinExample {
        id: "single-minimum",
        title: "Query the normalized `single` minimum",
        program: "limit = realmin(\"single\")",
        display_output: Some("limit = single(1.1755e-38)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(limit, \"single\"));\nassert(isfinite(limit));\nassert(limit > single(0));",
        },
    },
    BuiltinExample {
        id: "like-sparse-single",
        title: "Match a sparse `single` prototype",
        program: "prototype = sparse(single(1));\nlimit = realmin(\"like\", prototype)",
        display_output: Some("limit is a 1-by-1 sparse single value"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(issparse(limit));\nassert(isa(limit, \"single\"));\nassert(isequal(size(limit), [1 1]));\nassert(full(limit) > single(0));",
        },
    },
];

const REALMAX_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "double-maximum",
        title: "Query the largest finite `double` value",
        program: "limit = realmax",
        display_output: Some("limit = 1.7977e+308"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(limit, \"double\"));\nassert(isfinite(limit));\nassert(isinf(limit * 2));",
        },
    },
    BuiltinExample {
        id: "single-maximum",
        title: "Query the largest finite `single` value",
        program: "limit = realmax(\"single\")",
        display_output: Some("limit = single(3.4028e+38)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(limit, \"single\"));\nassert(isfinite(limit));\nassert(isinf(limit * single(2)));",
        },
    },
    BuiltinExample {
        id: "like-complex-single",
        title: "Match a complex `single` prototype",
        program: "prototype = complex(single(1), single(2));\nlimit = realmax(\"like\", prototype)",
        display_output: Some("limit is a 1-by-1 complex single value"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(limit, \"single\"));\nassert(~isreal(limit));\nassert(real(limit) == realmax(\"single\"));\nassert(imag(limit) == single(0));",
        },
    },
];

const FLINTMAX_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "double-consecutive-integer-limit",
        title: "Query the `double` consecutive-integer limit",
        program: "limit = flintmax",
        display_output: Some("limit = 9007199254740992"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(limit, \"double\"));\nassert(limit == 2^53);\nassert(limit + 1 == limit);\nassert(limit - 1 ~= limit);",
        },
    },
    BuiltinExample {
        id: "single-consecutive-integer-limit",
        title: "Query the `single` consecutive-integer limit",
        program: "limit = flintmax(\"single\")",
        display_output: Some("limit = single(16777216)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(limit, \"single\"));\nassert(limit == single(2^24));\nassert(limit + single(1) == limit);",
        },
    },
    BuiltinExample {
        id: "like-single-prototype",
        title: "Match a `single` prototype",
        program: "prototype = single([1 2 3]);\nlimit = flintmax(\"like\", prototype)",
        display_output: Some("limit = single(16777216)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(limit, \"single\"));\nassert(limit == single(16777216));\nassert(isequal(size(limit), [1 1]));",
        },
    },
];

const LIKE_FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Does the `like` form copy the prototype shape?",
    answer: "No. It returns a 1-by-1 value while preserving class, complexity, sparsity, and applicable placement.",
};

const REALMIN_FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does `realmin` include subnormal values?",
        answer: "No. It is the smallest positive normalized value for the selected floating-point class.",
    },
    LIKE_FAQ,
];

const REALMAX_FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Is the result infinite?",
        answer: "No. It is the largest finite value in the selected floating-point class.",
    },
    LIKE_FAQ,
];

const FLINTMAX_FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "What does consecutive mean?",
        answer: "Every integer from zero through this value has an exact representation in the selected floating-point class. Above it, adjacent integers can map to the same floating-point value.",
    },
    LIKE_FAQ,
];

pub(crate) const REALMIN_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("realmin"),
    slug: Some("realmin"),
    summary: "Return the smallest positive normalized floating-point value.",
    description: "`realmin` returns the smallest positive normalized value in `double` or `single`. The default is `double`; the `like` form derives its representation and applicable placement from a floating-point prototype.",
    keywords: &["realmin", "floating point", "limits", "single", "double", "like", "gpu", "distributed"],
    related: &["realmax", "flintmax"],
    sections: REALMIN_SECTIONS,
    examples: REALMIN_EXAMPLES,
    example_exemption: None,
    faqs: REALMIN_FAQS,
    links: &[],
    media: &[],
    evidence: FLOATING_EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

pub(crate) const REALMAX_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("realmax"),
    slug: Some("realmax"),
    summary: "Return the largest finite floating-point value.",
    description: "`realmax` returns the largest finite value in `double` or `single`. The default is `double`; the `like` form derives its representation and applicable placement from a floating-point prototype.",
    keywords: &["realmax", "floating point", "limits", "single", "double", "like", "gpu", "distributed"],
    related: &["realmin", "flintmax"],
    sections: REALMAX_SECTIONS,
    examples: REALMAX_EXAMPLES,
    example_exemption: None,
    faqs: REALMAX_FAQS,
    links: &[],
    media: &[],
    evidence: FLOATING_EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

pub(crate) const FLINTMAX_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("flintmax"),
    slug: Some("flintmax"),
    summary: "Return the largest consecutive integer in a floating-point class.",
    description: "`flintmax` returns the largest consecutive integer representable in `double` or `single`: 2^53 for `double` and 2^24 for `single`. The `like` form derives its representation and applicable placement from a floating-point prototype.",
    keywords: &["flintmax", "floating point", "integer precision", "single", "double", "like", "gpu", "distributed"],
    related: &["realmax", "realmin", "intmax"],
    sections: FLINTMAX_SECTIONS,
    examples: FLINTMAX_EXAMPLES,
    example_exemption: None,
    faqs: FLINTMAX_FAQS,
    links: &[],
    media: &[],
    evidence: FLOATING_EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
