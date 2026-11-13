use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Accepts two or more inputs and compares every input against the first.",
            "Treats `NaN` values as equal, including inside numeric arrays, complex values, cells, and structs.",
            "Requires arrays to have identical shape.",
            "Compares cell arrays and structs recursively.",
            "Returns a scalar logical value.",
            "Sparse values compare to full values, and real values compare to complex values when the imaginary component is zero.",
            "Structs and struct arrays compare matching field names and values regardless of field insertion order.",
            "Complex scalars and one-element complex arrays compare component values exactly across storage classes.",
            "Nested cells, structs and struct arrays are compared recursively; unsupported object, callable, symbolic, foreign and other value kinds return false without invoking user code.",
            "Zero requested outputs evaluates and discards the result; more than one output fails before any provider access. At least two inputs are required."
        ]
    },
    BuiltinDocumentationSection {
        heading: "Placement and execution",
        paragraphs: &[
            "All resident numeric leaves, including leaves in cells and structs, download through their exact owning provider with class, shape and payload validation. Inputs retain residency; the result is a host logical scalar. There is no device comparison or fusion kernel.",
            "Distributed arguments require materialization before recursive comparison. The portable async runtime binding is shared by interpreter and compiled calls; downloads may suspend or fail. No callbacks, workspace, filesystem, randomness or time services are used. Transfer costs and scheduling belong to shared materialization services."
        ]
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    FIELD_ORDER_EXAMPLE,
    BuiltinExample {
        id: "example-1",
        title: "Comparing NaN values",
        program: "tf = isequaln(NaN, NaN)",
        display_output: Some("ans = logical\n     1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(tf)); assert(isscalar(tf)); assert(tf == true);",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Comparing arrays recursively",
        program: "tf = isequaln({[1 NaN]}, {[1 NaN]})",
        display_output: Some("ans = logical\n     1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(tf)); assert(isscalar(tf)); assert(tf == true);",
        },
    },
    BuiltinExample {
        id: "cross-class",
        title: "Compare exact numeric values across classes",
        program: "tf = isequaln(uint64(255), int16(255), 255, single(255))",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf); assert(islogical(tf));",
        },
    },
    BuiltinExample {
        id: "nested-resident",
        title: "Compare nested resident values without changing their residency",
        program: "G = gpuArray(uint16([1 2]));\ntf = isequaln({G}, {uint16([1 2])})",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf); assert(isgpuarray(G)); assert(isa(gather(G), 'uint16'));",
        },
    },
];
const FIELD_ORDER_EXAMPLE: BuiltinExample = BuiltinExample {
    id: "field-order",
    title: "Compare structures independently of field order",
    program: "left = struct('a', 1, 'b', 2);\nright = struct('b', 2, 'a', 1);\ntf = isequaln(left, right)",
    display_output: Some("tf = true"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(tf); assert(islogical(tf));"
    },
};
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isequaln"),
    slug: Some("isequaln"),
    summary: "Test array equality while treating NaN values as equal.",
    description: "`isequaln(A, B, ...)` returns one host logical scalar when all inputs match in shape and values. Numeric, logical and character values compare across classes with exact integer comparisons; strings compare as text. Corresponding NaNs compare equal, including recursively.",
    keywords: &[
        "isequaln",
        "equality",
        "comparison",
        "nan",
        "logical"
    ],
    related: &[
        "isequal"
    ],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "How is `isequaln` different from `isequal`?",
            answer: "`isequaln` treats `NaN` values as equal. `isequal` returns false when corresponding values are both `NaN`."
        }
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "isequal",
            target: BuiltinDocumentationLinkTarget::Builtin("isequal")
        },
        BuiltinDocumentationLink {
            label: "Implementation",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/rel/isequal.rs")
        }
    ],
    media: &[
    ],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Implementation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/rel/isequal.rs")
            }
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Deep equality semantic tests",
                location: "crates/runmat-runtime/src/builtins/logical/rel/isequal/tests.rs"
            }
        ],
        notes: &[
            "Legacy sidecar prose, examples, FAQs, links and source reconciled here; class-matching claims corrected to exact cross-class runtime semantics. GPU and fusion metadata replaced by typed placement. Executable examples use logical value oracles. Required integrated native, browser and real-provider evidence is recorded separately."
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
