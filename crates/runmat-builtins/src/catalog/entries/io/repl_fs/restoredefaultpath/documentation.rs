use crate::*;

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "restore",
    title: "Restore the captured startup path",
    program: "restored = restoredefaultpath();\nassert(ischar(restored));\nassert(strcmp(path(), restored));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(restored)); assert(strcmp(path(), restored));" },
}];

pub(in super::super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("restoredefaultpath"),
    slug: Some("restoredefaultpath"),
    summary: "Restore the session's captured startup search path.",
    description: "`restoredefaultpath()` resets the active ordered search path to its captured startup layers: the application user-path setting, configured environment paths, and the known RunMat installation path. The current folder remains an implicit lookup location. A requested path output is a RunMat extension; the MATLAB-compatible form is statement-only.",
    keywords: &["restoredefaultpath", "search path", "startup"],
    related: &["path", "userpath", "addpath", "rmpath"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Session baseline",
            paragraphs: &[
                "Restoration uses the session's captured startup baseline rather than rereading the process working directory or current environment. A project or manifest source inventory has separate ownership and is not cleared by this search-path operation.",
                "Changes made by `path`, `addpath`, and `rmpath` to the current search path are removed. `restoredefaultpath` does not change the durable application user-path setting; use `userpath('reset')` to reset that setting.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Host requirements",
            paragraphs: &[
                "The application user-path setting must be hydrated before source resolution. If a prior application-store outcome left the local user path stale, the host must rehydrate it before restoration rather than restoring an obsolete entry.",
            ],
        },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does this clear the current folder?", answer: "No. The current folder is an implicit source lookup location, separate from the ordered search path." },
        BuiltinDocumentationFaq { question: "Does this reset the stored user path?", answer: "No. It restores the session path using the current stored setting; `userpath('reset')` changes the application setting." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "MATLAB restoredefaultpath reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/restoredefaultpath.html") },
        BuiltinDocumentationLink { label: "userpath", target: BuiltinDocumentationLinkTarget::Builtin("userpath") },
        BuiltinDocumentationLink { label: "path", target: BuiltinDocumentationLinkTarget::Builtin("path") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Restoration runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/compat/user_path.rs") },
            BuiltinDocumentationLink { label: "Origin-preserving search path", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/common/path_state/layers.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Factory restoration and origin preservation", location: "builtins::common::path_state::tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
