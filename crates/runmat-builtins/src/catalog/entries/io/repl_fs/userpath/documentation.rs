use crate::*;

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "query",
    title: "Inspect the application user path",
    program: "p = userpath();\nassert(ischar(p));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(ischar(p));",
    },
}];

pub(in super::super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("userpath"),
    slug: Some("userpath"),
    summary: "Query or change the application user-path setting.",
    description: "`userpath()` reads the application user path. `userpath('reset')` restores its default, `userpath('clear')` removes it, and `userpath(newpath)` stores one exact path spelling. A successful change updates the active session search path; a later session reads the same application setting before source resolution.",
    keywords: &["userpath", "search path", "application settings"],
    related: &["path", "addpath", "rmpath", "restoredefaultpath"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Setting and resolution",
            paragraphs: &[
                "The application setting is distinct from project files and from preference groups. A custom path retains its original spelling; a relative path is resolved against the captured working directory for the active session. `reset` and `clear` are separate operations: reset selects the home-directory default, while clear removes the user entry.",
                "Relative and other partial custom paths are available in the R2026b compatibility release. With the current R2026a pin, they require RunMat extension mode; absolute custom paths, reset, and clear do not.",
                "The default is `Documents/MATLAB` beneath the available home directory. On Linux, an absent `Documents` folder yields an empty default and no folder is created. Ordinary startup hydration and queries do not create folders. An explicit reset creates a missing default folder through the selected provider on macOS, Windows, and browser hosts; an unsupported provider reports its typed error. The active search path orders the user entry before configured environment paths and the RunMat installation path, while the current folder remains an implicit lookup location.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Availability and failures",
            paragraphs: &[
                "Application settings storage must be installed by the host. An unavailable or failed store is reported before source resolution; it is not silently replaced by session memory. Numeric and provider-resident values are not path text and reject before storage access.",
                "If a setting commits but the captured local filesystem authority is revoked before the session path can be updated, RunMat reports that the durable commit succeeded but local publication did not. The session re-reads the setting with a fresh path authority before its next source lookup.",
            ],
        },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does clear erase the application default?", answer: "No. Clear stores an explicit no-user-entry setting; reset selects the default again." },
        BuiltinDocumentationFaq { question: "Is a relative custom path rewritten in settings?", answer: "No. The exact text is stored, while each session resolves it against its captured filesystem and working directory." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "MATLAB userpath reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/userpath.html") },
        BuiltinDocumentationLink { label: "path", target: BuiltinDocumentationLinkTarget::Builtin("path") },
        BuiltinDocumentationLink { label: "restoredefaultpath", target: BuiltinDocumentationLinkTarget::Builtin("restoredefaultpath") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "User-path runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/compat/user_path.rs") },
            BuiltinDocumentationLink { label: "Session hydration", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-core/src/session/user_path.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Path origin, revision, and captured namespace", location: "builtins::common::path_state::tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
