use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`formatSpec` must be a string or character format for substitution. A cell array of character vectors is supported only by unary `compose(formatSpec)`, which converts it to cellstr output.",
        "If `formatSpec` is scalar and any argument array has more than one element, RunMat broadcasts the scalar specification over the array dimensions.",
        "When `formatSpec` is a string or character array with multiple elements, the output has the same shape as the specification. Each element uses the corresponding row or cell during formatting.",
        "Arguments can be numeric, logical, string, or character data. Cell-array data arguments are rejected because they are outside the documented domain. All eight integer classes are formatted directly from native values, so `%d` and `%i` preserve exact decimal text above `flintmax`.",
        "Resident arguments are a RunMat-only extension because MATLAB does not document GPU-array support for compose. RunMat mode gathers them through the owning provider before formatting and returns host text in the class selected by formatSpec.",
        "When you omit additional arguments, string formatSpec returns a string array, while character and cellstr formatSpec return a cell array of character vectors.",
        "General nonscalar row-wise formatting geometry remains a known gap: scalar expansion and equal-element-count forms are covered, but every higher-dimensional MATLAB format cycling case has not yet been modeled.",
        "Errors are raised if argument shapes are incompatible with the specification or if format specifiers are incomplete.",
    ] },
    BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
        "Formatting runs on the CPU. Resident inputs are a mode-gated RunMat extension and gather through the owning provider before substitution. A string format specification returns a host string array, while a character format specification returns a host cell array of character vectors.",
        "`compose` is a residency sink. In RunMat mode, resident tensors gather through their owning provider before formatting. All formatted text lives in host memory, so providers do not need compose-specific kernels.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Formatting A Scalar Value Into A Sentence",
        program: "msg = compose(\"The answer is %d.\", 42)",
        display_output: Some("msg = \"The answer is 42.\""),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(msg, \"The answer is 42.\"));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Broadcasting A Scalar Format Spec Over A Vector",
        program: "result = compose(\"Trial %d\", 1:4)",
        display_output: Some(
            "result = 1×4 string\n    \"Trial 1\"    \"Trial 2\"    \"Trial 3\"    \"Trial 4\"",
        ),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(result, [\"Trial 1\" \"Trial 2\" \"Trial 3\" \"Trial 4\"]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Using A String Array Of Formats",
        program:
            "spec = [\"max: %0.2f\", \"min: %0.2f\"];\nvalues = compose(spec, [3.14159, 0.125])",
        display_output: Some("values = 1×2 string\n    \"max: 3.14\"    \"min: 0.12\""),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(values, [\"max: 3.14\" \"min: 0.12\"]));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Using A Character Format Vector",
        program: "idx = compose('Row %02d', (1:3).')",
        display_output: Some("idx = 3×1 cell\n    {'Row 01'}\n    {'Row 02'}\n    {'Row 03'}"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(iscell(idx)); assert(isequal(idx, {'Row 01'; 'Row 02'; 'Row 03'}));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Combining Real And Imaginary Parts",
        program: "Z = [1+2i, 3-4i];\ntxt = compose(\"z = %s\", Z)",
        display_output: Some("txt = 1×2 string\n    \"z = 1+2i\"    \"z = 3-4i\""),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(txt, [\"z = 1+2i\" \"z = 3-4i\"]));",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Formatting GPU-Resident Data",
        program: "G = gpuArray([10 20 30]);\nlabels = compose(\"Value %d\", G)",
        display_output: Some(
            "labels = 1×3 string\n    \"Value 10\"    \"Value 20\"    \"Value 30\"",
        ),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(labels, [\"Value 10\" \"Value 20\" \"Value 30\"]));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What happens if the number of format arguments does not match the placeholders?", answer: "RunMat raises `compose: format data arguments must be scalars or match formatSpec size`. Ensure that each placeholder has a corresponding value or broadcast the specification appropriately." },
    BuiltinDocumentationFaq { question: "Can `compose` handle complex numbers?", answer: "Yes. Complex numbers use MATLAB's canonical `a + bi` formatting, so `%s` specifiers receive the string form of the complex scalar." },
    BuiltinDocumentationFaq { question: "How does `compose` treat logical inputs?", answer: "Logical values are converted to numeric `1` or `0` before formatting so they work with `%d`, `%i`, or `%f` placeholders." },
    BuiltinDocumentationFaq { question: "Does `compose` modify the shape of the output?", answer: "No. The output matches the broadcasted size between `formatSpec` and the input arguments. Scalar specifications broadcast across non-scalar arguments." },
    BuiltinDocumentationFaq { question: "What if I pass GPU arrays?", answer: "In RunMat mode, resident inputs are gathered before formatting and the text result lives on the host. Compatibility mode rejects this explicitly classified extension." },
    BuiltinDocumentationFaq { question: "How do I emit literal percent signs?", answer: "Use `%%` inside `formatSpec` just like `sprintf`. The formatter converts `%%` into a single `%`." },
    BuiltinDocumentationFaq { question: "Can I mix scalars and arrays in the arguments list?", answer: "Yes, as long as non-scalar arguments all share the same number of elements or match the size of `formatSpec`. Scalars broadcast across the target shape." },
    BuiltinDocumentationFaq { question: "What happens when `formatSpec` is empty?", answer: "An empty string formatSpec returns an empty string array. Empty character or cellstr formatSpec returns empty cellstr-compatible output with the corresponding shape." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
    BuiltinDocumentationLink { label: "sprintf", target: BuiltinDocumentationLinkTarget::Builtin("sprintf") },
    BuiltinDocumentationLink { label: "strcat", target: BuiltinDocumentationLinkTarget::Builtin("strcat") },
    BuiltinDocumentationLink { label: "join", target: BuiltinDocumentationLinkTarget::Builtin("join") },
    BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
    BuiltinDocumentationLink { label: "num2str", target: BuiltinDocumentationLinkTarget::Builtin("num2str") },
    BuiltinDocumentationLink { label: "str2double", target: BuiltinDocumentationLinkTarget::Builtin("str2double") },
    BuiltinDocumentationLink { label: "strcmp", target: BuiltinDocumentationLinkTarget::Builtin("strcmp") },
    BuiltinDocumentationLink { label: "strcmpi", target: BuiltinDocumentationLinkTarget::Builtin("strcmpi") },
    BuiltinDocumentationLink { label: "string.empty", target: BuiltinDocumentationLinkTarget::Builtin("string.empty") },
    BuiltinDocumentationLink { label: "strings", target: BuiltinDocumentationLinkTarget::Builtin("strings") },
    BuiltinDocumentationLink { label: "strlength", target: BuiltinDocumentationLinkTarget::Builtin("strlength") },
    BuiltinDocumentationLink { label: "strncmp", target: BuiltinDocumentationLinkTarget::Builtin("strncmp") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/compose.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("compose"), slug: Some("compose"),
    summary: "Format values into string arrays using printf-style placeholders in MATLAB and RunMat.",
    description: "`compose(formatSpec, A1, ..., An)` substitutes data into `%` placeholders. A string format specification returns a string array and a character format specification returns a cell array of character vectors. Integer arguments are read from exact native storage, including int64 and uint64 values above `flintmax`.",
    keywords: &["compose", "format", "string array", "sprintf", "gpu"],
    related: &["string", "sprintf", "strcat", "join", "char", "num2str", "str2double", "strcmp", "strcmpi", "string.empty", "strings", "strlength", "strncmp"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: FAQS, links: LINKS, media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/compose.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Existing identity family tests", location: "builtins::strings::core::compose::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Existing identity integration tests", location: "builtins::strings::core::compose::tests::compose_gpu_argument" },
        ],
        notes: &["Source-only canonical cutover; original JSON retained until compiled field and executable-example conservation."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
