use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Formatting", paragraphs: &[
        "`int2str(N)` returns a character array. Floating-point values are rounded to whole-number text; integer values use their native payload, so wide integers are not rounded through double. `NaN` and infinities retain their conventional text.",
        "A scalar or vector produces one character row. A matrix produces one text row for each numeric row, with padded columns. An empty numeric array produces a `0x0` character array. Complex, nonnumeric, and N-D inputs are rejected.",
        "Logical input is accepted only in RunMat compatibility mode and formats as 0 or 1. There are no precision or format-string options; use `num2str` for decimal formatting or a selected format.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and compatibility", paragraphs: &[
        "Formatting executes on the host in native and browser runtimes. Resident numeric input is downloaded through its provider owner before formatting; there is no GPU formatting kernel or fusion path. A download may suspend, and the result is a host character array.",
        "The builtin can throw for unsupported input, dimensions, options, and output count. It has no callback, workspace, filesystem, network, random, or time effect. Direct distributed-array formatting is not advertised by this binding.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "integer", title: "Convert an integer to character text",
        program: "chr = int2str(256)", display_output: Some("'256'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(chr, '256')); assert(ischar(chr));" },
    },
    BuiltinExample {
        id: "rounded", title: "Round a floating-point value",
        program: "chr = int2str(3.14159)", display_output: Some("'3'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(chr, '3'));" },
    },
    BuiltinExample {
        id: "matrix", title: "Align numeric matrix columns",
        program: "chr = int2str([5 10 20; 100 200 400])",
        display_output: Some("'  5   10   20'; '100  200  400'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(chr, 1), 2)); assert(strcmp(strtrim(chr(1,:)), '5   10   20')); assert(strcmp(chr(2,:), '100  200  400'));" },
    },
    BuiltinExample {
        id: "logical-extension", title: "Format a logical value in RunMat mode",
        program: "chr = int2str(true)", display_output: Some("'1'"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(chr, '1'));" },
    },
    BuiltinExample {
        id: "gpu-gather", title: "Format resident numeric input on the host",
        program: "G = gpuArray([10.2 20.8]); chr = int2str(G)",
        display_output: Some("'10  21'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(chr)); assert(strcmp(chr, '10  21'));" },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("int2str"), slug: Some("int2str"),
    summary: "Convert rounded integer values to a character array.",
    description: "`int2str` rounds real floating-point values or formats integer values exactly and returns aligned character text.",
    keywords: &["int2str", "integer to string", "number to string", "round", "gpu"],
    related: &["num2str", "sprintf", "string", "char"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How does int2str differ from num2str?", answer: "int2str rounds to integer text and accepts no format or precision option. num2str supports decimal and explicit formatting." },
        BuiltinDocumentationFaq { question: "Does int2str return a string scalar?", answer: "No. It returns a character array. Use string(...) for a string scalar." },
        BuiltinDocumentationFaq { question: "Does int2str execute on a GPU?", answer: "No. Supported resident numeric input is gathered to the host through its provider before formatting." },
        BuiltinDocumentationFaq { question: "Does it accept complex input?", answer: "No. Complex input raises an error." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "num2str", target: BuiltinDocumentationLinkTarget::Builtin("num2str") },
        BuiltinDocumentationLink { label: "sprintf", target: BuiltinDocumentationLinkTarget::Builtin("sprintf") },
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/int2str.rs") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/int2str.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "rounding, exact integer, shape, logical mode and GPU gather", location: "builtins::strings::core::int2str::tests" }],
        notes: &["Legacy sidecar summary, matrix alignment, empty behavior, GPU placement, FAQ and related links retained; display-only examples now have executable assertions. Logical mode is explicitly gated."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
