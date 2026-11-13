use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Variable-aware rendering", paragraphs: &[
        "`display(X)` prints a MATLAB-style `X =` header for a simple variable argument; an explicit expression prints only its value. Single-line values stay beside the header and multi-line matrices, structs, tables, datetime/duration arrays, and N-D pages print below it.",
        "Objects and handle objects first dispatch a registered or inherited `display` method, then an overloaded `disp` method. The generic fallback uses the same host formatter as `disp`.",
        "No output argument is exposed. A requested output is rejected before method dispatch or provider gathering.",
    ] },
    BuiltinDocumentationSection { heading: "Customization and residency", paragraphs: &[
        "The full `matlab.mixin.CustomDisplay` override protocol is not yet modeled; object-array customization is bounded by current classdef infrastructure. Unsupported object families use their runtime summary string.",
        "Provider-resident values are gathered to host memory before generic formatting. `display` runs no provider kernel and is a fusion boundary. An overloaded object method can instead handle its own value according to that method's contract.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "named",
        title: "Display a named variable",
        program: "A = [1 2; 3 4];\ndisplay(A)",
        display_output: Some("A =\n     1       2\n     3       4"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Succeeds,
    },
    BuiltinExample {
        id: "expression",
        title: "Display an expression",
        program: "display(1 + 2)",
        display_output: Some("3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Succeeds,
    },
    BuiltinExample {
        id: "string",
        title: "Display a string scalar",
        program: "msg = \"ready\";\ndisplay(msg)",
        display_output: Some("msg = ready"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Succeeds,
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How is `display` different from `disp`?", answer: "`disp(X)` prints the value alone. `display(X)` includes a variable-name header for a simple variable argument." },
    BuiltinDocumentationFaq { question: "Does `display` support class-specific output?", answer: "Yes. Registered `display` methods take priority over overloaded `disp` methods; the full `matlab.mixin.CustomDisplay` protocol is not modeled." },
    BuiltinDocumentationFaq { question: "Does `display` gather GPU arrays?", answer: "Yes. Generic formatting is host-side and gathers provider-resident data through the active provider." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "disp",
        target: BuiltinDocumentationLinkTarget::Builtin("disp"),
    },
    BuiltinDocumentationLink {
        label: "inputname",
        target: BuiltinDocumentationLinkTarget::Builtin("inputname"),
    },
    BuiltinDocumentationLink {
        label: "class",
        target: BuiltinDocumentationLinkTarget::Builtin("class"),
    },
    BuiltinDocumentationLink {
        label: "ismethod",
        target: BuiltinDocumentationLinkTarget::Builtin("ismethod"),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("display"), slug: Some("display"),
    summary: "Display a value with a MATLAB-style variable-name header.",
    description: "`display(X)` prints a variable-aware display block. For classdef objects it tries registered `display` and then `disp` methods before generic host formatting.",
    keywords: &["display", "disp", "object display", "command window", "print"],
    related: &["disp", "inputname", "class", "ismethod"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: FAQS, links: LINKS, media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Display runtime and method dispatch", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/display.rs") },
            BuiltinDocumentationLink { label: "Shared host formatter", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/disp.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Variable labels and class dispatch", location: "builtins::io::display::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Display in VM function execution", location: "crates/runmat-vm/tests/functions.rs" },
        ], notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
