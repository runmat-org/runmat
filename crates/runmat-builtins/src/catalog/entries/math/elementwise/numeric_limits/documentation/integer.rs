use super::INTEGER_EVIDENCE;
use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationFaq,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinExample,
    BuiltinExampleCompatibility, BuiltinExampleHarness, BuiltinExampleVerification,
};

const INTMIN_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`intmin()` returns the smallest `int32` value. A class-name argument selects `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, or `uint64`. The result is always 1-by-1 in the selected integer class; signed minima remain exact and every unsigned minimum is zero.",
            "`intmin(\"like\", prototype)` selects the integer class, complexity, and applicable placement from a real or complex integer prototype. It does not copy the prototype shape. A complex result has a zero imaginary component, and a distributed prototype produces a distributed scalar under the same distribution scheme without materializing the source value.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "Class-name forms return host scalars. With an integer `gpuArray` prototype, the `like` form creates only the scalar result on the prototype's registered provider and device. It preserves integer class, complexity, storage kind, and explicit-placement provenance without downloading the prototype.",
            "`intmin` is a scalar construction boundary rather than an elementwise or reduction kernel, so it is not fused with neighboring operations.",
        ],
    },
];

const INTMAX_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`intmax()` returns the largest `int32` value. A class-name argument selects `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, or `uint64`. The result is always 1-by-1 in the selected integer class, and the `int64` and `uint64` bounds remain exact instead of passing through floating point.",
            "`intmax(\"like\", prototype)` selects the integer class, complexity, and applicable placement from a real or complex integer prototype. It does not copy the prototype shape. A complex result has a zero imaginary component, and a distributed prototype produces a distributed scalar under the same distribution scheme without materializing the source value.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "Class-name forms return host scalars. With an integer `gpuArray` prototype, the `like` form creates only the scalar result on the prototype's registered provider and device. It preserves integer class, complexity, storage kind, and explicit-placement provenance without downloading the prototype.",
            "`intmax` is a scalar construction boundary rather than an elementwise or reduction kernel, so it is not fused with neighboring operations.",
        ],
    },
];

const INTMIN_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "int16-minimum",
        title: "Query the `int16` minimum",
        program: "limit = intmin(\"int16\")",
        display_output: Some("limit = int16(-32768)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(limit, \"int16\"));\nassert(limit == int16(-32768));\nassert(isequal(size(limit), [1 1]));",
        },
    },
    BuiltinExample {
        id: "unsigned-minimum",
        title: "Query an unsigned minimum",
        program: "limit = intmin(\"uint16\")",
        display_output: Some("limit = uint16(0)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(limit, \"uint16\"));\nassert(limit == uint16(0));",
        },
    },
    BuiltinExample {
        id: "like-integer-prototype",
        title: "Match an integer prototype",
        program: "prototype = int64([1 2 3]);\nlimit = intmin(\"like\", prototype)",
        display_output: Some("limit = int64(-9223372036854775808)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(limit, \"int64\"));\nassert(isequal(limit, intmin(\"int64\")));\nassert(isequal(size(limit), [1 1]));",
        },
    },
];

const INTMAX_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "default-maximum",
        title: "Query the default integer maximum",
        program: "limit = intmax",
        display_output: Some("limit = int32(2147483647)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(limit, \"int32\"));\nassert(limit == int32(2147483647));\nassert(isequal(size(limit), [1 1]));",
        },
    },
    BuiltinExample {
        id: "uint64-maximum",
        title: "Query the exact `uint64` maximum",
        program: "limit = intmax(\"uint64\")",
        display_output: Some("limit = uint64(18446744073709551615)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(limit, \"uint64\"));\nassert(limit > uint64(9007199254740992));\nassert(limit + uint64(1) == limit);",
        },
    },
    BuiltinExample {
        id: "like-integer-prototype",
        title: "Match an integer prototype",
        program: "prototype = uint16([1 2 3]);\nlimit = intmax(\"like\", prototype)",
        display_output: Some("limit = uint16(65535)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(limit, \"uint16\"));\nassert(limit == uint16(65535));\nassert(isequal(size(limit), [1 1]));",
        },
    },
];

const COMMON_FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "What is the default class?",
        answer: "`int32`.",
    },
    BuiltinDocumentationFaq {
        question: "Does the `like` form copy the prototype shape?",
        answer: "No. It always returns a 1-by-1 scalar while preserving the integer class, complexity, and applicable placement.",
    },
];

pub(crate) const INTMIN_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("intmin"),
    slug: Some("intmin"),
    summary: "Return the smallest value of an integer class.",
    description: "`intmin` returns the smallest value representable by an integer class. The default class is `int32`; the `like` form derives its representation and applicable placement from an integer prototype.",
    keywords: &["intmin", "integer", "limits", "like", "gpu", "distributed"],
    related: &["intmax", "flintmax"],
    sections: INTMIN_SECTIONS,
    examples: INTMIN_EXAMPLES,
    example_exemption: None,
    faqs: COMMON_FAQS,
    links: &[],
    media: &[],
    evidence: INTEGER_EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

pub(crate) const INTMAX_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("intmax"),
    slug: Some("intmax"),
    summary: "Return the largest value of an integer class.",
    description: "`intmax` returns the largest value representable by an integer class. The default class is `int32`; the `like` form derives its representation and applicable placement from an integer prototype.",
    keywords: &["intmax", "integer", "limits", "like", "gpu", "distributed"],
    related: &["intmin", "flintmax"],
    sections: INTMAX_SECTIONS,
    examples: INTMAX_EXAMPLES,
    example_exemption: None,
    faqs: COMMON_FAQS,
    links: &[],
    media: &[],
    evidence: INTEGER_EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
