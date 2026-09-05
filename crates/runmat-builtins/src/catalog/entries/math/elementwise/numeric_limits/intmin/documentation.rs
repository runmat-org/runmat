use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationFaq,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinExample,
    BuiltinExampleCompatibility, BuiltinExampleHarness, BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
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

const EXAMPLES: &[BuiltinExample] = &[
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

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What is the default class?", answer: "`int32`." },
    BuiltinDocumentationFaq { question: "Does the `like` form copy the prototype shape?", answer: "No. It always returns a 1-by-1 scalar while preserving the integer class, complexity, and applicable placement." },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("intmin"),
    slug: Some("intmin"),
    summary: "Return the smallest value of an integer class.",
    description: "`intmin` returns the smallest value representable by an integer class. The default class is `int32`; the `like` form derives its representation and applicable placement from an integer prototype.",
    keywords: &["intmin", "integer", "limits", "like", "gpu", "distributed"],
    related: &["intmax", "flintmax"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: &[],
    media: &[],
    evidence: super::super::evidence::INTEGER,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
