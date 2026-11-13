use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "State and file selection", paragraphs: &[
        "`diary` with no input toggles logging using the current diary file, named `diary` by default. `diary on` enables that file; `diary off` disables it.",
        "`diary filename` and `diary('filename')` select a file and enable logging. Existing files are appended to. The input must be a nonempty string scalar, singleton string array, or character row.",
        "This command-window sink has no output arguments. Asking for an output raises `RunMat:diary:TooManyOutputs` before opening a file.",
    ] },
    BuiltinDocumentationSection { heading: "Host effects", paragraphs: &[
        "When enabled, diary captures the current RunMat filesystem provider and working directory and keeps one append handle open until disabled or changed. Later provider or directory changes do not redirect that active log. Sandboxed and virtual providers are supported. Text captured by `evalc` is excluded while that capture scope is active.",
        "Diary does not log workspace values or figures by itself. It does not execute a GPU kernel or participate in fusion; the host console text stream is the source of the file contents.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "log-command-output",
        title: "Log command-window output",
        program: "diary runmat_session.log\nA = magic(3);\ndisp(A)\ndiary off\nlogged = fileread('runmat_session.log');\nassert(~isempty(logged));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(~isempty(logged));" },
    },
    BuiltinExample {
        id: "function-syntax",
        title: "Select a diary file with function syntax",
        program: "diary('experiment.log')\ndisp('started')\ndiary off\nlogged = fileread('experiment.log');\nassert(~isempty(strfind(logged, 'started')));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(~isempty(strfind(logged, 'started')));" },
    },
    BuiltinExample {
        id: "toggle",
        title: "Toggle current diary state",
        program: "diary\ndisp('this line is logged')\ndiary\nlogged = fileread('diary');\nassert(~isempty(strfind(logged, 'this line is logged')));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(~isempty(strfind(logged, 'this line is logged')));" },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does `diary` overwrite an existing file?",
        answer: "No. It opens the selected file in append mode.",
    },
    BuiltinDocumentationFaq {
        question: "Does `diary` log text captured by `evalc`?",
        answer: "No. Captured text is diverted into the returned string while capture is active.",
    },
    BuiltinDocumentationFaq {
        question: "Does `diary` log workspace variables or figure state?",
        answer: "No. It logs command-window text, not variables or figures.",
    },
    BuiltinDocumentationFaq {
        question: "Can `diary` return an output value?",
        answer: "No. Requesting an output raises `RunMat:diary:TooManyOutputs`.",
    },
    BuiltinDocumentationFaq {
        question: "Can `diary` write to host-specific filesystems?",
        answer: "Yes. The provider and directory active when the diary is enabled determine the destination for that logging period.",
    },
    BuiltinDocumentationFaq {
        question: "Does GPU residency matter for `diary`?",
        answer: "No. Diary is a host-side text sink, not a GPU kernel or fusion operation.",
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "disp",
        target: BuiltinDocumentationLinkTarget::Builtin("disp"),
    },
    BuiltinDocumentationLink {
        label: "evalc",
        target: BuiltinDocumentationLinkTarget::Builtin("evalc"),
    },
    BuiltinDocumentationLink {
        label: "fprintf",
        target: BuiltinDocumentationLinkTarget::Builtin("fprintf"),
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("diary"), slug: Some("diary"),
    summary: "Log command-window text to a file.",
    description: "`diary` records command-window stdout and stderr text to an append-only log through the filesystem provider captured when enabled. It accepts command and function syntax for enabling, disabling, toggling, and selecting the file.",
    keywords: &["diary", "log", "command window", "console", "file"],
    related: &["disp", "evalc", "fprintf"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: FAQS, links: LINKS, media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Diary runtime binding", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/diary.rs") },
            BuiltinDocumentationLink { label: "Console diary owner", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/console.rs") },
            BuiltinDocumentationLink { label: "Captured diary provider and append handle", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/console/diary.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Provider append, capture exclusion, output rejection", location: "builtins::io::diary::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Diary console file provider", location: "diary_logs_command_window_output_to_file_provider" },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
