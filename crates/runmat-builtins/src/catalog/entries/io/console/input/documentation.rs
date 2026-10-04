use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Interactive call forms", paragraphs: &[
        "`input(prompt)` displays a character-row or string-scalar prompt, then waits for one line from the active host. `input(prompt,'s')` returns the exact typed text, without a trailing newline, as a character row.",
        "The no-prompt `input()` form uses `Input: ` and is a RunMat extension. `input('s',prompt)` is an additional RunMat-only swapped-argument form. Strict compatibility mode rejects either extension before requesting a line.",
        "Native REPL/CLI hosts and browser hosts with an installed async input handler can provide the line. If no interaction service is available, input raises `RunMat:input:InteractionFailed`; it does not fabricate a value.",
    ] },
    BuiltinDocumentationSection { heading: "Expression and numeric results", paragraphs: &[
        "Without `'s'`, empty text returns an empty double matrix. Scalar numbers and constants (`pi`, `Inf`, `NaN`), logical values, and rectangular vector/matrix literals are parsed directly; logical matrices retain logical class and column-major layout.",
        "An installed expression-evaluation hook handles expressions beyond the direct literal parser. When no hook exists, the current fallback calls `str2double`; unsupported text is rejected and the interactive prompt continues until a valid response is supplied. Thus arbitrary expression support depends on the active host hook, rather than being guaranteed by the builtin alone.",
        "Prompt and flag controls must be host text scalars. Provider-resident controls are rejected before provider access. An evaluated expression result may itself be resident according to the expression's own execution route; input does not force it to the host.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Read a scalar from the interactive host",
        program: "value = input(\"Enter a scalar: \" );\nassert(value == 41);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::InteractiveHost,
        fixture: BuiltinExampleFixture::CliInteraction(BuiltinCliInteractionFixture {
            id: BuiltinExampleFixtureId {
                local_name: "scalar",
            },
            transcript: &[
                BuiltinCliTranscriptStep::ExpectOutput("Enter a scalar: "),
                BuiltinCliTranscriptStep::SendLine("41"),
                BuiltinCliTranscriptStep::SendEndOfInput,
            ],
        }),
        requirements: BuiltinExampleRequirements {
            host: BuiltinExampleHostRequirement::NativeOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(value == 41);",
        },
    },
    BuiltinExample {
        id: "raw-text",
        title: "Read unevaluated text",
        program: "name = input('Name: ', 's');\nassert(strcmp(name, 'RunMat'));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::InteractiveHost,
        fixture: BuiltinExampleFixture::CliInteraction(BuiltinCliInteractionFixture {
            id: BuiltinExampleFixtureId {
                local_name: "raw-text",
            },
            transcript: &[
                BuiltinCliTranscriptStep::ExpectOutput("Name: "),
                BuiltinCliTranscriptStep::SendLine("RunMat"),
                BuiltinCliTranscriptStep::SendEndOfInput,
            ],
        }),
        requirements: BuiltinExampleRequirements {
            host: BuiltinExampleHostRequirement::NativeOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(name, 'RunMat'));",
        },
    },
    BuiltinExample {
        id: "invalid-flag",
        title: "Reject an invalid string flag before interaction",
        program: "input('Name: ', 'raw')",
        display_output: Some("input: invalid string flag"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::ExpectedError {
            identifier: "RunMat:input:InvalidStringFlag",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does RunMat evaluate arbitrary MATLAB expressions?", answer: "It can when the active host installs an expression-evaluation hook. Direct scalar and matrix literals work without one; otherwise the no-hook fallback uses `str2double` and may reprompt on unsupported text." },
    BuiltinDocumentationFaq { question: "What happens if the user just presses Enter?", answer: "The numeric branch returns an empty double `[]`; string mode returns an empty character row." },
    BuiltinDocumentationFaq { question: "Can I call `input` inside GPU workflows?", answer: "Yes, but prompt and flag controls must be host text. Resident controls are rejected before provider access; the expression result retains its own route." },
    BuiltinDocumentationFaq { question: "What does input do in MATLAB?", answer: "`input(prompt)` waits for a response and evaluates it as an expression; `input(prompt,'s')` returns raw character text." },
    BuiltinDocumentationFaq { question: "How do I get string input from the user in MATLAB?", answer: "Use `str = input('Enter text: ', 's')`. RunMat returns a character row for this form." },
    BuiltinDocumentationFaq { question: "What is the difference between input and keyboard in MATLAB?", answer: "`input` requests one value; `keyboard` enters an interactive debugging session. RunMat's input supports scripted interaction." },
    BuiltinDocumentationFaq { question: "Can I validate user input in MATLAB?", answer: "Call `input` in a loop until the value passes a condition, for example while a requested number is negative." },
    BuiltinDocumentationFaq { question: "Does input work in RunMat's browser sandbox?", answer: "Yes when the embedding host installs an async input handler. Otherwise the operation reports `RunMat:input:InteractionFailed`." },
];
const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "disp",
    target: BuiltinDocumentationLinkTarget::Builtin("disp"),
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("input"), slug: Some("input"),
    summary: "Prompt users for interactive input in MATLAB and RunMat.",
    description: "`input` requests a line from the active host and returns evaluated input or raw character text when called with `'s'`. Its host interaction can suspend; host control and expression result have separate residency.",
    keywords: &["input", "prompt", "stdin", "interactive", "pause", "gather"],
    related: &["disp"], sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: FAQS, links: LINKS, media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Input binding and parsers", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/input.rs") },
            BuiltinDocumentationLink { label: "Host interaction service", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/interaction.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Input modes, matrix layout, extension gating and resident rejection", location: "builtins::io::input::tests" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
