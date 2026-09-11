use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Runtime implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/structs/core/struct.rs"),
    }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, array, field-name, copy, and error tests", location: "builtins::structs::core::r#struct::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Resident field preservation test", location: "builtins::structs::core::r#struct::tests::struct_preserves_gpu_handles_with_registered_provider" },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`struct()` creates a scalar struct with no fields. Name/value pairs create fields in insertion order; a repeated name replaces its earlier value. Field names must be nonempty MATLAB identifiers within `namelengthmax` and cannot be language keywords.",
        "When any value is a cell array, every cell-valued field must have the same shape. Each cell contributes to the corresponding struct-array element, while non-cell values are replicated across the array. `struct([])` creates a 0-by-0 struct array.",
        "`struct(S)` copies an existing scalar struct or struct array. Numeric fields retain exact class, shape, and storage. Construction does not gather resident values or otherwise transform field payloads.",
    ] },
    BuiltinDocumentationSection { heading: "GPU execution", paragraphs: &[
        "Struct assembly is host bookkeeping, not a provider operation. A gpuArray field remains an owned resident handle inside the result; no kernel or implicit download occurs.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "named-fields", title: "Create a scalar struct with named fields", program: "s = struct(\"name\", \"entry\", \"score\", 42)", display_output: Some("s has fields name and score"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(s.name == \"entry\");\nassert(s.score == 42);\nassert(isfield(s, \"name\"));" } },
    BuiltinExample { id: "struct-array", title: "Build a struct array from cell-valued fields", program: "names = {\"first\", \"second\"};\nages = {36, 45};\nrecords = struct(\"name\", names, \"age\", ages)", display_output: Some("records is a 1-by-2 struct array"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(records), [1 2]));\nassert(records(1).name == \"first\");\nassert(records(2).age == 45);" } },
    BuiltinExample { id: "replicated-field", title: "Replicate a scalar field across a struct array", program: "ids = struct(\"id\", {101, 102, 103}, \"department\", \"Research\")", display_output: Some("ids is a 1-by-3 struct array"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(ids), [1 3]));\nassert(ids(1).department == \"Research\");\nassert(ids(3).department == \"Research\");" } },
    BuiltinExample { id: "copy", title: "Copy an existing struct", program: "a = struct(\"id\", 7, \"label\", \"demo\");\nb = struct(a);\nb.id = 8", display_output: Some("a.id is 7 and b.id is 8"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(a.id == 7);\nassert(b.id == 8);\nassert(b.label == \"demo\");" } },
    BuiltinExample { id: "empty-array", title: "Create an empty struct array", program: "s = struct([]);\nshape = size(s)", display_output: Some("shape = [0 0]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(shape, [0 0]));" } },
    BuiltinExample { id: "resident-field", title: "Keep a gpuArray field resident", program: "G = gpuArray(uint16([1 2; 3 4]));\ns = struct(\"data\", G, \"label\", \"resident\");\nhost = gather(s.data)", display_output: Some("host remains a uint16 matrix"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(s.label == \"resident\");\nassert(isa(host, \"uint16\"));\nassert(isequal(host, uint16([1 2; 3 4])));" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Which field names are valid?", answer: "Names must satisfy MATLAB identifier rules: start with a letter, contain only letters, digits, or underscores, fit within `namelengthmax`, and not be a keyword." },
    BuiltinDocumentationFaq { question: "How do I create a struct array?", answer: "Use equally shaped cell arrays as field values. Non-cell field values are replicated across every result element." },
    BuiltinDocumentationFaq { question: "What happens when a field name repeats?", answer: "The last value replaces the earlier value for that field." },
    BuiltinDocumentationFaq { question: "What does `struct([])` return?", answer: "It returns a 0-by-0 struct array." },
    BuiltinDocumentationFaq { question: "Does `struct` copy an existing struct?", answer: "Yes. `struct(S)` returns an independent copy of a supported struct or struct array." },
    BuiltinDocumentationFaq { question: "Does construction gather GPU fields?", answer: "No. Resident handles remain resident field values until another operation explicitly uses or gathers them." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "fieldnames",
        target: BuiltinDocumentationLinkTarget::Builtin("fieldnames"),
    },
    BuiltinDocumentationLink {
        label: "getfield",
        target: BuiltinDocumentationLinkTarget::Builtin("getfield"),
    },
    BuiltinDocumentationLink {
        label: "isfield",
        target: BuiltinDocumentationLinkTarget::Builtin("isfield"),
    },
    BuiltinDocumentationLink {
        label: "orderfields",
        target: BuiltinDocumentationLinkTarget::Builtin("orderfields"),
    },
    BuiltinDocumentationLink {
        label: "rmfield",
        target: BuiltinDocumentationLinkTarget::Builtin("rmfield"),
    },
    BuiltinDocumentationLink {
        label: "setfield",
        target: BuiltinDocumentationLinkTarget::Builtin("setfield"),
    },
    BuiltinDocumentationLink {
        label: "gpuArray",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuArray"),
    },
    BuiltinDocumentationLink {
        label: "gather",
        target: BuiltinDocumentationLinkTarget::Builtin("gather"),
    },
];

pub(super) const STRUCT_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("struct"), slug: Some("struct"), summary: "Create scalar structs or struct arrays from field/value inputs.",
    description: "`struct` creates scalar structs and struct arrays from name/value pairs, copies existing structs, and expands equally shaped cell-valued fields.",
    keywords: &["struct", "structure", "name-value", "record", "struct array", "fields"],
    related: &["fieldnames", "gather", "getfield", "gpuArray", "isfield", "orderfields", "rmfield", "setfield"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS,
    links: LINKS, media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
