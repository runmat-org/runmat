use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("jsondecode"), slug: Some("jsondecode"),
    summary: "Parse UTF-8 JSON text into MATLAB-compatible values in RunMat.",
    description: "`jsondecode(text)` converts JSON text to numeric, logical, character, string-array, struct, or cell values. Rectangular arrays use column-major RunMat storage; irregular arrays retain source order in cells.",
    keywords: &["jsondecode", "json", "parse json", "struct from json", "gpu gather"],
    related: &["jsonencode", "struct", "cell", "fileread"],
    sections: &[
        BuiltinDocumentationSection { heading: "Input and scalar mapping", paragraphs: &[
            "Input must be a host character row, string scalar, or one-element string array. Leading and trailing whitespace is ignored. Comments and trailing commas are not JSON. Resident numeric handles reject before provider access; this operation is not a GPU parser.",
            "JSON numbers become double values; booleans become logical values; strings become character rows; null becomes empty double []. Numeric literals outside the finite double range report a parser error with its source detail.",
        ] },
        BuiltinDocumentationSection { heading: "Arrays and objects", paragraphs: &[
            "Perfectly rectangular homogeneous arrays become dense numeric, logical, or string arrays with source dimensions and RunMat column-major storage. Rectangular heterogeneous arrays become cells with the corresponding layout; irregular arrays become 1-by-N cells.",
            "Objects become scalar structs with lexically ordered keys. Rectangular object arrays sharing one field set become typed struct arrays; differing field sets remain cells of scalar structs. Unicode and escaped surrogate pairs are decoded by the JSON parser.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Parsing and result construction run on the host in native and browser runtimes. The result and intermediate JSON tree are materialized in memory, so large documents can require substantial host memory.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "scalar-number", title: "Parse a scalar number", program: "value = jsondecode(\"42\");\nvalue", display_output: Some("value = 42"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(value == 42); assert(isa(value, 'double'));" } },
        BuiltinExample { id: "matrix", title: "Decode a matrix of doubles", program: "text = \"[[1,2,3],[4,5,6]]\";\nmatrix = jsondecode(text)", display_output: Some("matrix =\n     1     2     3\n     4     5     6"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(matrix, [1 2 3; 4 5 6]));" } },
        BuiltinExample { id: "scalar-object", title: "Convert an object into a struct", program: "record = jsondecode('{\"name\":\"entry\",\"age\":37}');\ndisp(record.name);\ndisp(record.age)", display_output: Some("entry\n    37"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(record.name, 'entry')); assert(record.age == 37);" } },
        BuiltinExample { id: "object-array", title: "Decode a structure array", program: "text = '[{\"name\":\"first\",\"role\":\"review\"},{\"name\":\"second\",\"role\":\"build\"}]';\nrecords = jsondecode(text);\nrecords(1).name\nrecords(2).role", display_output: Some("ans = 'first'\nans = 'build'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(records(1).name, 'first')); assert(strcmp(records(2).role, 'build'));" } },
        BuiltinExample { id: "heterogeneous", title: "Preserve mixed values in a cell array", program: "data = jsondecode('[\"RunMat\",42,true]')", display_output: Some("data =\n  1x3 cell array\n    {\"RunMat\"}    {[42]}    {[1]}"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(iscell(data)); assert(strcmp(data{1}, 'RunMat')); assert(data{2} == 42); assert(data{3});" } },
        BuiltinExample { id: "null", title: "Decode null as empty double", program: "value = jsondecode(\"null\");\ndisp(value);\nisempty(value)", display_output: Some("[]\nans = logical 1"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isempty(value)); assert(isa(value, 'double'));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What encodings does `jsondecode` support?", answer: "JSON text uses Unicode; escaped surrogate pairs are decoded by the parser. The input is already RunMat character or string text, not raw file bytes." },
        BuiltinDocumentationFaq { question: "How are JSON objects represented?", answer: "Objects become scalar structs. Rectangular arrays with common fields become struct arrays; differing fields remain cells. Keys have deterministic lexical order." },
        BuiltinDocumentationFaq { question: "What happens with JSON numbers that overflow double?", answer: "The parser rejects numbers outside the finite double range and retains its error detail." },
        BuiltinDocumentationFaq { question: "Does `jsondecode` support comments or trailing commas?", answer: "No. Input must be standard JSON." },
        BuiltinDocumentationFaq { question: "How are arrays of strings represented?", answer: "Rectangular homogeneous arrays become string arrays; mixed arrays become cells." },
        BuiltinDocumentationFaq { question: "How is `null` represented?", answer: "As empty double []. Within a mixed array it occupies its own cell." },
        BuiltinDocumentationFaq { question: "Can I decode deeply nested arrays?", answer: "Yes, subject to parser and host-memory limits. Rectangular numeric, logical, and string arrays retain dimensions; irregular arrays use cells." },
        BuiltinDocumentationFaq { question: "What if input is invalid JSON?", answer: "The call raises an invalid-JSON error including the parser cause." },
        BuiltinDocumentationFaq { question: "Does whitespace matter?", answer: "Leading and trailing whitespace is ignored; embedded newlines are valid where JSON grammar permits them." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "jsonencode", target: BuiltinDocumentationLinkTarget::Builtin("jsonencode") },
        BuiltinDocumentationLink { label: "struct", target: BuiltinDocumentationLinkTarget::Builtin("struct") },
        BuiltinDocumentationLink { label: "cell", target: BuiltinDocumentationLinkTarget::Builtin("cell") },
        BuiltinDocumentationLink { label: "fileread", target: BuiltinDocumentationLinkTarget::Builtin("fileread") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "JSON text admission and runtime binding", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/json/jsondecode.rs") },
            BuiltinDocumentationLink { label: "JSON value, array, and object conversion", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/json/jsondecode/conversion.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Decoder unit suite", location: "builtins::io::json::jsondecode::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Matrix layout", location: "builtins::io::json::jsondecode::tests::jsondecode_matrix_to_tensor" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar struct", location: "builtins::io::json::jsondecode::tests::jsondecode_object_to_struct" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "String array", location: "builtins::io::json::jsondecode::tests::jsondecode_string_array" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Mixed cell", location: "builtins::io::json::jsondecode::tests::jsondecode_mixed_array_returns_cell" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Rectangular cell layout", location: "builtins::io::json::jsondecode::tests::jsondecode_rectangular_cell_array_preserves_layout" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Object array", location: "builtins::io::json::jsondecode::tests::jsondecode_array_of_objects_returns_typed_column_vector" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Null class", location: "builtins::io::json::jsondecode::tests::jsondecode_null_returns_empty_double" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Encoder round trip", location: "builtins::io::json::jsondecode::tests::jsondecode_round_trip_with_jsonencode" },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
