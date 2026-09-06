use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Remove folders from the session search path",
        paragraphs: &[
            "`rmpath` removes one or more folders from the ordered path used for runtime function, script, class, package, MEX, and file resolution. Arguments may be character vectors, strings, string arrays, or cells containing those text forms. Multi-row character arrays contribute one folder per row, and path-list text is split with the platform separator.",
            "Each requested folder is processed once. A directly matching stored entry can be removed even if the folder has since disappeared. Otherwise relative text is normalized from the current working folder and checked so missing folders, non-folders, and folders not on the path produce distinct errors.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Atomic host operation",
        paragraphs: &[
            "All arguments are decoded and every removal is validated before the session path changes. If any requested folder fails, the original path remains active. Successful changes affect later resolution immediately and remain isolated to the current session.",
            "Numeric inputs are not path text and reject before provider access. The returned character row contains the previous path and can be passed to `path` to restore it.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "single", title: "Remove one folder", program: "mkdir('toolbox');\noriginal = path;\naddpath('toolbox');\nbefore = path;\nold = rmpath('toolbox');\nassert(strcmp(old, before));\nassert(~contains(path, 'toolbox'));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "multiple", title: "Remove several folders", program: "mkdir('alpha');\nmkdir('beta');\noriginal = path;\naddpath('alpha', 'beta');\nrmpath('alpha', 'beta');\nassert(~contains(path, 'alpha'));\nassert(~contains(path, 'beta'));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "path-list", title: "Remove a path-list argument", program: "mkdir('tree');\nmkdir(fullfile('tree', 'child'));\noriginal = path;\ntreePath = genpath('tree');\naddpath(treePath);\nrmpath(treePath);\nassert(~contains(path, 'tree'));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "cell", title: "Remove folders from a cell", program: "mkdir('one');\nmkdir('two');\noriginal = path;\nfolders = {'one', 'two'};\naddpath(folders);\nrmpath(folders);\nassert(~contains(path, 'one'));\nassert(~contains(path, 'two'));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "restore", title: "Restore the path after removal", program: "mkdir('analysis');\noriginal = path;\naddpath('analysis');\nold = rmpath('analysis');\npath(old);\nassert(contains(path, 'analysis'));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "deleted-folder", title: "Remove a stored folder after it is deleted", program: "mkdir('ephemeral');\nfolder = fullfile(pwd, 'ephemeral');\noriginal = path;\naddpath(folder);\nrmdir(folder);\nrmpath(folder);\nassert(~contains(path, 'ephemeral'));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Are relative folders supported?", answer: "Yes. If there is no direct stored match, relative text is normalized from the current working folder before matching." },
    BuiltinDocumentationFaq { question: "What if a folder was deleted after it was added?", answer: "Its stored path entry can still be removed by an exact match. Filesystem validation is used only when normalization is needed." },
    BuiltinDocumentationFaq { question: "What happens with duplicate arguments?", answer: "Each normalized requested folder is processed once." },
    BuiltinDocumentationFaq { question: "Can I pass genpath output?", answer: "Yes. Path-list text is split with the platform separator." },
    BuiltinDocumentationFaq { question: "Does a failed call partially change the path?", answer: "No. Every requested removal is validated before the new path is committed." },
    BuiltinDocumentationFaq { question: "Can I pass numeric or resident values?", answer: "No. `rmpath` accepts text containers; numeric host and resident values reject before provider access." },
    BuiltinDocumentationFaq { question: "What value does rmpath return?", answer: "It returns the previous path as a character row so it can be restored with `path(oldpath)`." },
];

const RELATED: &[&str] = &[
    "path", "addpath", "genpath", "which", "exist", "cd", "copyfile", "delete", "dir", "fullfile",
    "getenv", "ls", "mkdir", "movefile", "pwd", "rmdir", "savepath", "setenv", "tempdir",
    "tempname",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "path", target: BuiltinDocumentationLinkTarget::Builtin("path") },
    BuiltinDocumentationLink { label: "addpath", target: BuiltinDocumentationLinkTarget::Builtin("addpath") },
    BuiltinDocumentationLink { label: "genpath", target: BuiltinDocumentationLinkTarget::Builtin("genpath") },
    BuiltinDocumentationLink { label: "which", target: BuiltinDocumentationLinkTarget::Builtin("which") },
    BuiltinDocumentationLink { label: "exist", target: BuiltinDocumentationLinkTarget::Builtin("exist") },
    BuiltinDocumentationLink { label: "cd", target: BuiltinDocumentationLinkTarget::Builtin("cd") },
    BuiltinDocumentationLink { label: "copyfile", target: BuiltinDocumentationLinkTarget::Builtin("copyfile") },
    BuiltinDocumentationLink { label: "delete", target: BuiltinDocumentationLinkTarget::Builtin("delete") },
    BuiltinDocumentationLink { label: "dir", target: BuiltinDocumentationLinkTarget::Builtin("dir") },
    BuiltinDocumentationLink { label: "fullfile", target: BuiltinDocumentationLinkTarget::Builtin("fullfile") },
    BuiltinDocumentationLink { label: "getenv", target: BuiltinDocumentationLinkTarget::Builtin("getenv") },
    BuiltinDocumentationLink { label: "ls", target: BuiltinDocumentationLinkTarget::Builtin("ls") },
    BuiltinDocumentationLink { label: "mkdir", target: BuiltinDocumentationLinkTarget::Builtin("mkdir") },
    BuiltinDocumentationLink { label: "movefile", target: BuiltinDocumentationLinkTarget::Builtin("movefile") },
    BuiltinDocumentationLink { label: "pwd", target: BuiltinDocumentationLinkTarget::Builtin("pwd") },
    BuiltinDocumentationLink { label: "rmdir", target: BuiltinDocumentationLinkTarget::Builtin("rmdir") },
    BuiltinDocumentationLink { label: "savepath", target: BuiltinDocumentationLinkTarget::Builtin("savepath") },
    BuiltinDocumentationLink { label: "setenv", target: BuiltinDocumentationLinkTarget::Builtin("setenv") },
    BuiltinDocumentationLink { label: "tempdir", target: BuiltinDocumentationLinkTarget::Builtin("tempdir") },
    BuiltinDocumentationLink { label: "tempname", target: BuiltinDocumentationLinkTarget::Builtin("tempname") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/rmpath/mod.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rmpath"), slug: Some("rmpath"),
    summary: "Remove folders from the active session search path.",
    description: "`rmpath` removes validated folder entries from the ordered path used by runtime callable and file resolution.",
    keywords: &["rmpath", "search path", "function resolution", "remove folder"],
    related: RELATED, sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS,
    links: LINKS, media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Search-path mutation runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/rmpath/mod.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Text admission, removal, errors, atomicity, and path-list parsing", location: "builtins::io::repl_fs::rmpath::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Callable reselection after removal", location: "runmat-core path precedence tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Isolated filesystem examples", location: "scripts/runtime/verify-builtin-examples.mjs" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
