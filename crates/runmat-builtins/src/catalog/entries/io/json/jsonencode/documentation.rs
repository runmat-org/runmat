use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("jsonencode"), slug: Some("jsonencode"),
    summary: "Serialize MATLAB values to JSON text in RunMat.",
    description: "`jsonencode(value)` serializes supported numeric, logical, character, string, struct, cell, and object values to a character row containing JSON text. Options control indentation and non-finite numbers.",
    keywords: &["jsonencode", "json", "serialization", "struct to json", "pretty print json", "gpu gather"],
    related: &["jsondecode", "gpuArray", "gather"],
    sections: &[
        BuiltinDocumentationSection { heading: "Values and layout", paragraphs: &[
            "The result is a 1-by-N character row, not a string scalar. Numeric and logical arrays become JSON arrays in logical row/column order from column-major storage; scalars become bare literals. Character and string values become strings or arrays of strings according to their shape.",
            "Struct scalars become objects and struct arrays become arrays of objects. Object keys are emitted in canonical lexical order. Empty struct arrays become [] and cannot retain their field schema in JSON. Cell arrays preserve nested rows and columns as JSON arrays.",
            "All eight real integer classes serialize exact signed or unsigned decimal literals directly from their integer storage, including uint64 values above flintmax. Complex object encoding and sparse densification are separate RunMat-only extensions.",
        ] },
        BuiltinDocumentationSection { heading: "Options and non-finite values", paragraphs: &[
            "By default PrettyPrint is false and ConvertInfAndNaN is true, so NaN and positive or negative infinity become null. With ConvertInfAndNaN false, encountering a non-finite value raises an error. PrettyPrint true uses four-space indentation and LF newlines.",
            "Pass a scalar options struct or complete name-value pairs. Supported names are PrettyPrint and ConvertInfAndNaN, case-insensitively. Logical values are the compatible option form. Floating numeric, typed-integer, and boolean-like text coercions are separately gated RunMat extensions; a documented 0 or 1 numeric ConvertInfAndNaN value remains accepted. Invalid names, pair counts, and known extension gates are checked before resident data gather.",
        ] },
        BuiltinDocumentationSection { heading: "Residency and portability", paragraphs: &[
            "No GPU JSON kernel runs. Automatic internal residency gathers through the owning provider; explicit gpuArray values require the RunMat resident-input extension before gather. Missing owners return the normal gather failure. Native and browser runtimes share the host serializer.",
            "Printable Unicode is emitted directly; quotes and control characters use standard JSON escapes. Serialization builds an intermediate JSON tree and result character row, so large nested values can require substantial host memory.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "struct", title: "Convert a struct to JSON", program: "record = struct('name', 'entry', 'age', 37);\nencoded = jsonencode(record)", display_output: Some("encoded = '{\"age\":37,\"name\":\"entry\"}'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(encoded, '{\"age\":37,\"name\":\"entry\"}'));" } },
        BuiltinExample { id: "pretty", title: "Pretty-print a matrix", program: "A = [8 1 6; 3 5 7; 4 9 2];\nencoded = jsonencode(A, 'PrettyPrint', true)", display_output: Some("encoded =\n'[\n    [8,1,6],\n    [3,5,7],\n    [4,9,2]\n]'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(jsondecode(encoded), A)); assert(contains(encoded, sprintf('\\n')));" } },
        BuiltinExample { id: "nested-cell", title: "Encode nested cells", program: "C = {struct('task','encode','ok',true), {'nested', 42}};\nencoded = jsonencode(C)", display_output: Some("encoded = '[{\"ok\":true,\"task\":\"encode\"},[\"nested\",42]]'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(encoded, '[{\"ok\":true,\"task\":\"encode\"},[\"nested\",42]]'));" } },
        BuiltinExample { id: "nonfinite", title: "Convert NaN and Inf to null", program: "data = [1 NaN Inf];\nencoded = jsonencode(data)", display_output: Some("encoded = '[1,null,null]'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(encoded, '[1,null,null]'));" } },
        BuiltinExample { id: "nonfinite-error", title: "Reject non-finite values when conversion is disabled", program: "data = [1 NaN Inf];\ntry\n  jsonencode(data, 'ConvertInfAndNaN', false);\ncatch err\n  message = err.message;\n  disp(message);\nend", display_output: Some("jsonencode: ConvertInfAndNaN must be true to encode NaN or Inf values"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(message, 'jsonencode: ConvertInfAndNaN must be true to encode NaN or Inf values'));" } },
        BuiltinExample { id: "resident", title: "Serialize GPU-resident data", program: "G = gpuArray(eye(2));\nencoded = jsonencode(G)", display_output: Some("encoded = '[[1,0],[0,1]]'"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(encoded, '[[1,0],[0,1]]'));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What values are supported?", answer: "Numeric, logical, character, string, struct, cell, and supported object values. Opaque foreign values and function handles reject." },
        BuiltinDocumentationFaq { question: "Why are fields alphabetic?", answer: "JSON object keys use deterministic lexical order rather than runtime insertion order." },
        BuiltinDocumentationFaq { question: "How are complex numbers encoded?", answer: "The compatible surface rejects them; RunMat mode has a separately gated real/imag object extension." },
        BuiltinDocumentationFaq { question: "Does the result use char or string?", answer: "A row character array. Use string(jsonencode(x)) if a string scalar is wanted." },
        BuiltinDocumentationFaq { question: "Can nested values be pretty-printed?", answer: "Yes, with PrettyPrint true; indentation uses four spaces per level." },
        BuiltinDocumentationFaq { question: "How do empty arrays encode?", answer: "Empty numeric, logical, and string arrays become []; a zero-row character array also becomes [], while a 1-by-0 character row becomes an empty JSON string. Scalar empty structs become {}, while empty struct arrays become []." },
        BuiltinDocumentationFaq { question: "Is column-major layout preserved?", answer: "The JSON uses logical row/column order; decoding a rectangular result reconstructs the corresponding RunMat shape." },
        BuiltinDocumentationFaq { question: "What if ConvertInfAndNaN is false?", answer: "Any NaN, Inf, or -Inf raises an error instead of emitting null." },
        BuiltinDocumentationFaq { question: "Which newline style is used?", answer: "Pretty-printed output uses LF on every platform." },
        BuiltinDocumentationFaq { question: "Are Unicode characters escaped?", answer: "Printable Unicode is emitted directly. Quotes and control characters use JSON escapes." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "jsondecode", target: BuiltinDocumentationLinkTarget::Builtin("jsondecode") },
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
        BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "JSON runtime binding and gather", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/json/jsonencode.rs") },
            BuiltinDocumentationLink { label: "Option admission and compatibility gates", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/json/jsonencode/options.rs") },
            BuiltinDocumentationLink { label: "Value and container conversion", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/json/jsonencode/convert.rs") },
            BuiltinDocumentationLink { label: "JSON rendering and escaping", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/json/jsonencode/render.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Encoder unit suite", location: "builtins::io::json::jsonencode::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Struct round trip", location: "builtins::io::json::jsonencode::tests::basic::jsonencode_struct_round_trip" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Resident gather", location: "builtins::io::json::jsonencode::tests::extended::jsonencode_gpu_tensor_gathers_host_data" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Pretty struct options", location: "builtins::io::json::jsonencode::tests::basic::jsonencode_struct_options_enable_pretty_print" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Non-finite control", location: "builtins::io::json::jsonencode::tests::extended::jsonencode_convert_inf_and_nan_controls_null_output" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Empty character rows", location: "builtins::io::json::jsonencode::tests::extended::jsonencode_char_array_zero_rows_is_empty_array" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "WGPU readback", location: "builtins::io::json::jsonencode::tests::extended::jsonencode_gpu_tensor_wgpu_gathers_host_data" },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
