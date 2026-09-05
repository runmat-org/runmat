use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Console control",
        paragraphs: &[
            "`clc` requests that the active host clear its visible command output. It emits a control event instead of writing terminal escape bytes, so terminal, desktop, and browser hosts can apply the request through their own display layer.",
            "The operation does not alter workspace variables, figures, the working directory, or execution state. It accepts no input arguments and suppresses automatic display of its internal empty return value.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Host behavior",
        paragraphs: &[
            "Terminal hosts translate the control event into their clear-screen behavior. WebAssembly hosts expose a `clear` stream entry to the embedding UI. A host may retain output in an external log even after removing it from the visible command window.",
            "`clc` does not inspect array data and never invokes an accelerator provider. Its contract is portable across native and browser runtimes, but its visible effect is implemented by the host.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "clear-between-messages",
        title: "Clear between command-window messages",
        program: "disp('Step 1 complete');\nclc;\ndisp('Ready for the next command')",
        display_output: Some("Ready for the next command"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Succeeds,
    },
    BuiltinExample {
        id: "preserve-workspace",
        title: "Leave workspace values unchanged",
        program: "sample_count = 42;\nclc;\nassert(sample_count == 42);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(sample_count == 42);",
        },
    },
    BuiltinExample {
        id: "reject-input",
        title: "Reject input arguments",
        program: "clc(1)",
        display_output: Some("clc: expected no input arguments"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Native,
        verification: BuiltinExampleVerification::ExpectedError {
            identifier: "RunMat:clc:ArgumentCount",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does clc delete previous output permanently?", answer: "It asks the active host to clear visible output. A host may retain the same entries in logs or execution history." },
    BuiltinDocumentationFaq { question: "What does a browser host receive?", answer: "The WebAssembly execution stream contains an entry whose stream is `clear`; the embedding UI decides how to update its console." },
    BuiltinDocumentationFaq { question: "Does clc change variables or figures?", answer: "No. Use `clear` for workspace bindings and `close` for figures." },
    BuiltinDocumentationFaq { question: "Can clc accept arguments?", answer: "No. Passing an argument returns `clc: expected no input arguments`." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "disp", target: BuiltinDocumentationLinkTarget::Builtin("disp") },
    BuiltinDocumentationLink { label: "clear", target: BuiltinDocumentationLinkTarget::Builtin("clear") },
    BuiltinDocumentationLink { label: "close", target: BuiltinDocumentationLinkTarget::Builtin("close") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/clc.rs") },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Console control runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/clc.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Argument rejection and empty sink result", location: "builtins::io::clc::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Clear-screen execution stream", location: "command_controls::clc_emits_clear_screen_control_stream" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("clc"),
    slug: Some("clc"),
    summary: "Clear the visible command window or console display.",
    description: "`clc` emits a portable clear-screen control event for the active host without changing workspace or figure state.",
    keywords: &["clc", "clear console", "command window", "console", "screen"],
    related: &["disp", "clear", "close"],
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
