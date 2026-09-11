use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Create directories",
        paragraphs: &[
            "`mkdir(folderName)` creates the named directory and any missing intermediate directories. `mkdir(parentFolder, folderName)` creates a relative child beneath the parent and creates the parent when necessary.",
            "With outputs, `status` is a logical scalar. A newly created directory returns true with empty character rows for `msg` and `msgID`. An existing directory also returns true and supplies an informational message and identifier.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors and execution",
        paragraphs: &[
            "When outputs are requested, an operational filesystem failure returns false with diagnostic text. Without outputs, the same failure raises an error; an existing directory emits a warning governed by the session's warning settings.",
            "Folder names accept character rows and scalar strings. Relative paths use the current folder, a leading `~` expands through the active environment, and native Windows paths may use drive-letter or UNC syntax. The filesystem service receives the path without rewriting platform-specific components.",
            "Directory creation runs through the host or browser filesystem service rather than an acceleration provider. Provider-resident arguments are gathered before validation; directory creation itself never executes on a GPU.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "single",
        title: "Create a directory",
        program: "folder = tempname();\nstatus = mkdir(folder);\nexistsAfter = exist(folder, 'dir');\nrmdir(folder);",
        display_output: Some("status is logical true and existsAfter is nonzero"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(status) && status); assert(existsAfter ~= 0);",
        },
    },
    BuiltinExample {
        id: "missing-parent",
        title: "Create a missing parent and nested child",
        program: "parent = tempname();\nstatus = mkdir(parent, fullfile('archive', '2026'));\ncreated = exist(fullfile(parent, 'archive', '2026'), 'dir');\nrmdir(parent, 's');",
        display_output: Some("status is logical true and the nested directory exists"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(status) && status); assert(created ~= 0);",
        },
    },
    BuiltinExample {
        id: "existing",
        title: "Inspect an existing-directory result",
        program: "folder = tempname();\nmkdir(folder);\n[status, msg, msgID] = mkdir(folder);\nrmdir(folder);",
        display_output: Some("status is true and msg and msgID describe the existing directory"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(status) && status); assert(~isempty(msg)); assert(~isempty(msgID));",
        },
    },
    BuiltinExample {
        id: "single-path-hierarchy",
        title: "Create intermediate directories from one path",
        program: "root = tempname();\nfolder = fullfile(root, 'data', 'archive');\nstatus = mkdir(folder);\ncreated = exist(folder, 'dir');\nrmdir(root, 's');",
        display_output: Some("status is logical true and every path component is created"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(status) && status); assert(created ~= 0);",
        },
    },
    BuiltinExample {
        id: "character-row",
        title: "Create a directory from a character row",
        program: "folder = char(tempname());\nstatus = mkdir(folder);\ncreated = exist(folder, 'dir');\nrmdir(folder);",
        display_output: Some("character-row paths use the same filesystem operation"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(status) && status); assert(created ~= 0);",
        },
    },
    BuiltinExample {
        id: "occupied-by-file",
        title: "Capture a target-file failure",
        program: "target = tempname();\nfid = fopen(target, 'w');\nfclose(fid);\n[status, msg, msgID] = mkdir(target);\nstillFile = exist(target, 'file');\ndelete(target);",
        display_output: Some("status is logical false, diagnostics are populated, and the file remains"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(status) && ~status); assert(~isempty(msg)); assert(~isempty(msgID)); assert(stillFile ~= 0);",
        },
    },
    BuiltinExample {
        id: "diagnostic-shape",
        title: "Inspect successful diagnostic outputs",
        program: "folder = tempname();\n[status, msg, msgID] = mkdir(folder);\nmessageSize = size(msg);\nidSize = size(msgID);\nrmdir(folder);",
        display_output: Some("successful creation returns empty character rows"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(status); assert(ischar(msg) && isequal(messageSize, [1 0])); assert(ischar(msgID) && isequal(idSize, [1 0]));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does mkdir create intermediate directories?",
        answer: "Yes. Both the single-path form and a missing parent in the two-input form create the required directory chain.",
    },
    BuiltinDocumentationFaq {
        question: "What happens when the target exists as a file?",
        answer: "The file is never overwritten. Output forms return logical false and diagnostics; a no-output call raises an error.",
    },
    BuiltinDocumentationFaq {
        question: "Can mkdir run on a GPU?",
        answer: "No. Directory creation is a filesystem operation. Provider-resident arguments are gathered before validation and execution.",
    },
    BuiltinDocumentationFaq {
        question: "Which path value types does mkdir accept?",
        answer: "A character row, scalar string, or one-element string array is accepted. Other values reject before filesystem access.",
    },
    BuiltinDocumentationFaq {
        question: "How do I handle creation failures programmatically?",
        answer: "Request status, message, and message ID. Status is logical false after an operational failure; the other outputs describe it. A no-output call raises an error.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "rmdir",
        target: BuiltinDocumentationLinkTarget::Builtin("rmdir"),
    },
    BuiltinDocumentationLink {
        label: "fullfile",
        target: BuiltinDocumentationLinkTarget::Builtin("fullfile"),
    },
    BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/directory_lifecycle/mkdir/mod.rs"),
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("mkdir"),
    slug: Some("mkdir"),
    summary: "Create directories and return logical status with optional diagnostics.",
    description: "`mkdir` creates a directory hierarchy through the active filesystem service.",
    keywords: &[
        "mkdir",
        "create directory",
        "folder",
        "filesystem",
        "status",
        "message",
        "messageid",
    ],
    related: &["rmdir", "fullfile", "pwd", "cd", "exist"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Directory creation runtime",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/directory_lifecycle/mkdir/mod.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Directory creation, diagnostics, output, and virtual-filesystem behavior",
                location: "builtins::io::repl_fs::directory_lifecycle::mkdir::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Executable filesystem examples",
                location: "scripts/runtime/verify-builtin-examples.mjs",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
