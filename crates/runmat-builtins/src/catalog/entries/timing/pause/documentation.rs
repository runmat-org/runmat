use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "pause-1",
        title: "Pausing for a fixed duration",
        program: "tic;\npause(0.05);\nelapsed = toc",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(elapsed >= 0);",
        },
    },
    BuiltinExample {
        id: "pause-2",
        title: "Waiting for user input mid-script",
        program: "disp('Press any key to continue the demo...');\npause;",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::InteractiveHost,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(pause('query'), 'on'));",
        },
    },
    BuiltinExample {
        id: "pause-3",
        title: "Temporarily disabling pauses in automated runs",
        program: "previous = pause('off');\npause(1.0);\npause(previous);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(pause('query'), previous));",
        },
    },
    BuiltinExample {
        id: "pause-4",
        title: "Querying the current pause mode",
        program: "current = pause('query');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(current, 'on') || strcmp(current, 'off'));",
        },
    },
    BuiltinExample {
        id: "pause-5",
        title: "Using empty input to rely on the default behaviour",
        program: "pause([]);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::InteractiveHost,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(pause('query'), 'on'));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("pause"), slug: Some("pause"),
    summary: "Suspend execution until user interaction or for a duration.",
    description: "`pause` supports keypress wait, nonnegative scalar duration, and session-owned on/off/query state control.",
    keywords: &["pause", "sleep", "wait", "delay"], related: &["tic", "toc", "timeit"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior and placement", paragraphs: &[
        "`pause` and `pause([])` wait for a keypress when enabled; noninteractive standard input may complete immediately. Positive-infinite numeric input follows the default keypress form.",
        "`pause(n)` waits for a nonnegative real scalar duration in seconds. Zero provides an asynchronous yield point. Waiting is cooperative and cancellation stops the pending timer.",
        "`pause('off')` and `pause('on')` change the current session's policy and return the previous state as a character row; `pause('query')` reads the current state without changing it. Each new runtime context begins enabled.",
        "Every built-in integer scalar class is accepted from authoritative storage before converting at the host timer boundary; logical scalars are also accepted.",
        "Explicit gpuArray input is a RunMat-mode extension gated before provider access. Automatically resident input is gathered through its owner; no pause operation runs on the GPU or joins fusion.",
    ] }], examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does `pause` block forever when standard input is not interactive?", answer: "No. A noninteractive host may complete a keypress request immediately; interactive hosts wait for a key." },
        BuiltinDocumentationFaq { question: "What happens if I call `pause` with a negative duration?", answer: "RunMat raises `RunMat:pause:InvalidInputArgument`." },
        BuiltinDocumentationFaq { question: "Does `pause` accept logical or integer values?", answer: "Yes. Logical scalars and all eight integer classes are accepted; integer storage is read exactly before the host timer conversion." },
        BuiltinDocumentationFaq { question: "Can I force pausing off globally?", answer: "`pause('off')` disables waits in the current session. Save the returned state and pass it to `pause` to restore it." },
        BuiltinDocumentationFaq { question: "Does `pause('query')` change the pause state?", answer: "No. It returns the current mode, 'on' or 'off'." },
        BuiltinDocumentationFaq { question: "Is `pause` affected by GPU fusion or auto-offload?", answer: "It does not enter a GPU timing kernel or fusion; automatically resident input may require owner-aware gather." },
        BuiltinDocumentationFaq { question: "What is the default pause state?", answer: "'on' for every new RunMat session." },
        BuiltinDocumentationFaq { question: "Can I pass a gpuArray as the duration?", answer: "RunMat mode accepts explicit gpuArray duration as a gated extension and gathers through the owning provider." },
    ],
    links: &[BuiltinDocumentationLink { label: "tic", target: BuiltinDocumentationLinkTarget::Builtin("tic") }, BuiltinDocumentationLink { label: "toc", target: BuiltinDocumentationLinkTarget::Builtin("toc") }],
    media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Pause runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/timing/pause.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "pause runtime and session policy", location: "crates/runmat-runtime/src/builtins/timing/pause.rs tests" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
