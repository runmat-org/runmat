use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`isequal(A, B)` compares two values for equality.",
            "`isequal(A, B, C, ...)` compares all inputs against each other; returns `true` only if all are equal.",
            "Arrays must have identical shape (size) to be equal.",
            "Numeric values compare across classes without lossy promotion of wide integers.",
            "All corresponding elements must be identical for arrays to be equal.",
            "NaN values are NOT equal to each other (NaN ~= NaN).",
            "Empty arrays `[]` are equal to other empty arrays with the same shape.",
            "Cell arrays are compared element-by-element recursively.",
            "Structs are compared field-by-field.",
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
        title: "Comparing two identical scalars",
        program: "tf = isequal(5, 5)",
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
        title: "Comparing two different scalars",
        program: "tf = isequal(5, 6)",
        display_output: Some("ans = logical\n     0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(tf)); assert(isscalar(tf)); assert(tf == false);",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Comparing identical matrices",
        program: "A = [1 2; 3 4];\nB = [1 2; 3 4];\ntf = isequal(A, B)",
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
        id: "example-4",
        title: "Comparing matrices with different shapes",
        program: "A = [1 2 3];\nB = [1; 2; 3];\ntf = isequal(A, B)",
        display_output: Some("ans = logical\n     0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(tf)); assert(isscalar(tf)); assert(tf == false);",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Comparing multiple arrays at once",
        program: "tf = isequal([1 2], [1 2], [1 2])",
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
        id: "example-6",
        title: "NaN values are not equal",
        program: "tf = isequal(NaN, NaN)",
        display_output: Some("ans = logical\n     0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(tf)); assert(isscalar(tf)); assert(tf == false);",
        },
    },
    BuiltinExample {
        id: "example-7",
        title: "Verifying that every cell starts with `[]`",
        program: "C = cell(2, 2);\ntf = isequal(C{1,1}, [], C{2,2}, [])",
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
        program: "tf = isequal(uint64(255), int16(255), 255, single(255))",
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
        program: "G = gpuArray(uint16([1 2]));\ntf = isequal({G}, {uint16([1 2])})",
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
    program:
        "left = struct('a', 1, 'b', 2);\nright = struct('b', 2, 'a', 1);\ntf = isequal(left, right)",
    display_output: Some("tf = true"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(tf); assert(islogical(tf));",
    },
};
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isequal"),
    slug: Some("isequal"),
    summary: "Test array equality in MATLAB and RunMat.",
    description: "`isequal(A, B, ...)` returns one host logical scalar when all inputs match in shape and values. Numeric, logical and character values compare across classes with exact integer comparisons; strings compare as text. NaNs compare unequal, including recursively.",
    keywords: &[
        "isequal",
        "equality",
        "comparison",
        "logical",
        "array comparison"
    ],
    related: &[
        "eq",
        "strcmp",
        "ge",
        "gt",
        "le",
        "lt",
        "ne"
    ],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "How is `isequal` different from `==`?",
            answer: "`isequal` returns a single logical value indicating whether all inputs are completely identical. The `==` operator performs element-wise comparison and returns an array of the same size as the inputs."
        },
        BuiltinDocumentationFaq {
            question: "How does `isequal` handle NaN values?",
            answer: "NaN values are NOT considered equal. `isequal(NaN, NaN)` returns `false`. Use `isequaln` if you want NaN values to be treated as equal."
        },
        BuiltinDocumentationFaq {
            question: "Can I compare more than two arrays?",
            answer: "Yes. `isequal(A, B, C, ...)` accepts any number of inputs and returns `true` only if all inputs are equal to each other."
        },
        BuiltinDocumentationFaq {
            question: "Are empty arrays equal?",
            answer: "Empty arrays with the same shape are equal. For example, `isequal([], [])` returns `true`, and `isequal(zeros(0,3), zeros(0,3))` returns `true`."
        },
        BuiltinDocumentationFaq {
            question: "How are cell arrays compared?",
            answer: "Cell arrays are compared element-by-element. Each corresponding pair of cells must contain equal values for the cell arrays to be equal."
        },
        BuiltinDocumentationFaq {
            question: "What happens with different data types?",
            answer: "Numeric classes compare by exact value, including signed and unsigned integers, floating-point values, logical values and character codes. Values are not preconverted through double; other unlike value kinds return false."
        }
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "eq",
            target: BuiltinDocumentationLinkTarget::Builtin("eq")
        },
        BuiltinDocumentationLink {
            label: "strcmp",
            target: BuiltinDocumentationLinkTarget::Builtin("strcmp")
        },
        BuiltinDocumentationLink {
            label: "ge",
            target: BuiltinDocumentationLinkTarget::Builtin("ge")
        },
        BuiltinDocumentationLink {
            label: "gt",
            target: BuiltinDocumentationLinkTarget::Builtin("gt")
        },
        BuiltinDocumentationLink {
            label: "le",
            target: BuiltinDocumentationLinkTarget::Builtin("le")
        },
        BuiltinDocumentationLink {
            label: "lt",
            target: BuiltinDocumentationLinkTarget::Builtin("lt")
        },
        BuiltinDocumentationLink {
            label: "ne",
            target: BuiltinDocumentationLinkTarget::Builtin("ne")
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
