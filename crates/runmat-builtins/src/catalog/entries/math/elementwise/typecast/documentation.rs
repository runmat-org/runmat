use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Byte reinterpretation",
        paragraphs: &[
            "`Y = typecast(X, newtype)` reads the native byte sequence of a full numeric or logical scalar or vector as elements of `newtype`. It does not numerically convert the values.",
            "The source byte count must be divisible by the output element width. A row remains a row and a column remains a column; changing the element width changes only the number of elements.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes and native byte order",
        paragraphs: &[
            "Double, single, logical, and all eight fixed-width integer classes use their authoritative storage. Multi-byte values follow the platform's native byte order, so byte-vector presentation can differ between little-endian and big-endian systems while a same-platform round trip remains exact.",
            "Character reinterpretation is not available because RunMat's character representation cannot preserve every raw UTF-16 code unit. Sparse arrays and nonscalar matrices are rejected.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Prototype selection",
        paragraphs: &[
            "`Y = typecast(X, \"like\", prototype)` selects the output class and real or complex storage from a host prototype. A complex prototype pairs adjacent decoded components as real and imaginary values.",
            "The prototype controls representation only; its value and shape do not contribute to the output.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Accelerated execution",
        paragraphs: &[
            "Real numeric gpuArray input is gathered exactly through its owning provider, reinterpreted on the host, and restored to the same owner. RunMat preserves fixed-width integer bytes without a floating-point intermediary.",
            "Complex and logical gpuArray input and the `like` form with resident values are rejected before transfer. `typecast` is a fusion boundary.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "inspect-uint32-bytes",
        title: "Inspect the native bytes of unsigned integers",
        program: "values = uint32([1 256]);\nbytes = typecast(values, \"uint8\")",
        display_output: Some("bytes is a 1-by-8 uint8 vector in native byte order"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(bytes, \"uint8\"));\nassert(isequal(size(bytes), [1 8]));\nassert(isequal(typecast(bytes, \"uint32\"), values));" },
    },
    BuiltinExample {
        id: "wide-integer-round-trip",
        title: "Round-trip wide integer bit patterns exactly",
        program: "values = uint64([9007199254740993 intmax(\"uint64\")]);\ncopy = typecast(typecast(values, \"uint8\"), \"uint64\")",
        display_output: Some("copy retains both uint64 values exactly"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(copy, \"uint64\"));\nassert(isequal(copy, values));" },
    },
    BuiltinExample {
        id: "complex-prototype",
        title: "Select paired complex integer output with a prototype",
        program: "z = typecast(int16([-1 2 -3 4]), \"like\", complex(int16(0)))",
        display_output: Some("z = [-1+2i -3+4i] with complex int16 storage"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(z, \"int16\"));\nassert(isequal(z, complex(int16([-1 -3]), int16([2 4]))));" },
    },
    BuiltinExample {
        id: "column-orientation",
        title: "Preserve column orientation while changing width",
        program: "words = uint32([1; 2; 3]);\nbytes = typecast(words, \"uint8\")",
        display_output: Some("bytes is a 12-by-1 uint8 column"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(bytes), [12 1]));\nassert(isequal(typecast(bytes, \"uint32\"), words));" },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does typecast convert numeric values?", answer: "No. It preserves the source bytes and changes how fixed-width elements are interpreted." },
    BuiltinDocumentationFaq { question: "Why can the number of elements change?", answer: "The byte count stays fixed. Reading the same bytes with a narrower class produces more elements, and a wider class produces fewer." },
    BuiltinDocumentationFaq { question: "Which byte order does typecast use?", answer: "It uses the host platform's native byte order, matching the native in-memory representation." },
    BuiltinDocumentationFaq { question: "What does the like form copy?", answer: "It copies the prototype's class and real-or-complex representation, not its data or shape." },
    BuiltinDocumentationFaq { question: "Can typecast accept matrices?", answer: "Only scalar and vector input is supported. A nonscalar matrix is rejected." },
    BuiltinDocumentationFaq { question: "Can typecast remain resident on a GPU provider?", answer: "Supported real numeric input returns to its original owner after exact host reinterpretation; unsupported resident forms fail explicitly." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "swapbytes",
        target: BuiltinDocumentationLinkTarget::Builtin("swapbytes"),
    },
    BuiltinDocumentationLink {
        label: "uint8",
        target: BuiltinDocumentationLinkTarget::Builtin("uint8"),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/typecast") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Class, byte, shape, prototype, and diagnostic behavior", location: "builtins::math::elementwise::typecast::tests::host" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Exact provider transfer and ownership", location: "builtins::math::elementwise::typecast::tests::provider" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("typecast"),
    slug: Some("typecast"),
    summary: "Reinterpret numeric bytes as another class without converting values.",
    description: "`typecast` preserves the native byte sequence of a full numeric or logical scalar or vector and decodes that sequence with a selected output representation.",
    keywords: &["typecast", "reinterpret", "bytes", "integer", "single", "double", "gpuArray"],
    related: &["swapbytes", "uint8"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
