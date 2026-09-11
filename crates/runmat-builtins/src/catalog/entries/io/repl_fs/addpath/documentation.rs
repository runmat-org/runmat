use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Add folders to the session search path",
        paragraphs: &[
            "`addpath` validates folders and adds them to the ordered path used to resolve functions, scripts, classes, packages, MEX files, and data files. The default position is the beginning; `'-end'` appends and `'-begin'` selects the beginning explicitly.",
            "Arguments may be character vectors, string values, string arrays, or cells containing those text forms. A multi-row character array contributes one folder per row. A path-list string is split with the platform separator, which lets `addpath` consume `genpath` output directly.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Ordering, validation, and compatibility",
        paragraphs: &[
            "Relative folders are resolved from the current working folder, `~` is expanded, and every folder must exist. Existing occurrences are removed before insertion, so each normalized folder appears once. A call either validates all requested folders and updates the path or leaves it unchanged.",
            "The `'-frozen'` option is accepted for source compatibility but does not create a separate frozen tier. RunMat mode also accepts a dense real numeric row of Unicode scalar values; MATLAB compatibility modes require text. Resident numeric rows pass compatibility admission before one host gather.",
            "The returned character row is the previous path. Changes apply immediately to later resolution in the same session and do not alter another session's path.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "prepend", title: "Add a folder at the beginning", program: "mkdir('toolbox');\noriginal = path;\nold = addpath('toolbox');\nassert(strcmp(old, original));\nassert(contains(path, 'toolbox'));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "append", title: "Append a folder", program: "mkdir('shared');\noriginal = path;\naddpath('shared', '-end');\nassert(contains(path, 'shared'));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "command", title: "Use command syntax", program: "mkdir('SourceCode');\noriginal = path;\naddpath ./SourceCode\nassert(contains(path, 'SourceCode'));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "multiple", title: "Add several folders", program: "mkdir('alpha');\nmkdir('beta');\noriginal = path;\naddpath('alpha', 'beta', '-end');\nassert(contains(path, 'alpha'));\nassert(contains(path, 'beta'));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "string-array", title: "Add folders from a string array", program: "mkdir('algorithms');\nmkdir('visualization');\noriginal = path;\nfolders = [\"algorithms\", \"visualization\"];\naddpath(folders);\nassert(contains(path, 'algorithms'));\nassert(contains(path, 'visualization'));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "generated", title: "Add a generated folder tree", program: "mkdir('tree');\nmkdir(fullfile('tree', 'child'));\noriginal = path;\naddpath(genpath('tree'));\nassert(contains(path, 'child'));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "frozen", title: "Accept the frozen compatibility option", program: "mkdir('vendor');\noriginal = path;\naddpath('vendor', '-frozen');\nassert(contains(path, 'vendor'));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "restore", title: "Restore the previous path", program: "mkdir('temporary');\noriginal = path;\nold = addpath('temporary');\nassert(strcmp(old, original));\npath(old);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "numeric-codes", title: "Use numeric character codes in RunMat mode", program: "mkdir('codes');\noriginal = path;\naddpath(uint16('codes'));\nassert(contains(path, 'codes'));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Are relative folders supported?", answer: "Yes. They are resolved from the current working folder and stored in normalized absolute form." },
    BuiltinDocumentationFaq { question: "How do I append instead of prepend?", answer: "Pass `'-end'`. The default and `'-begin'` place folders at the beginning." },
    BuiltinDocumentationFaq { question: "What happens to duplicate entries?", answer: "Existing occurrences are removed before the requested ordering is applied, so a normalized folder appears once." },
    BuiltinDocumentationFaq { question: "Can I pass genpath output?", answer: "Yes. Path-list text is split using the platform separator." },
    BuiltinDocumentationFaq { question: "What does -frozen do?", answer: "RunMat accepts the option for source compatibility. It currently has no separate path-state effect." },
    BuiltinDocumentationFaq { question: "Can I pass numeric character codes?", answer: "RunMat mode accepts a dense real row, including a resident row after compatibility admission. MATLAB compatibility modes require text." },
    BuiltinDocumentationFaq { question: "What value does addpath return?", answer: "It returns the previous path as a character row so it can be restored with `path(oldpath)`." },
];

const RELATED: &[&str] = &[
    "path", "rmpath", "genpath", "which", "exist", "cd", "copyfile", "delete", "dir", "fullfile",
    "getenv", "ls", "mkdir", "movefile", "pwd", "rmdir", "savepath", "setenv", "tempdir",
    "tempname",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "path", target: BuiltinDocumentationLinkTarget::Builtin("path") },
    BuiltinDocumentationLink { label: "rmpath", target: BuiltinDocumentationLinkTarget::Builtin("rmpath") },
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
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/addpath/mod.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("addpath"), slug: Some("addpath"),
    summary: "Add validated folders to the active session search path.",
    description: "`addpath` prepends or appends folders to the ordered path used by runtime callable and file resolution.",
    keywords: &["addpath", "search path", "function resolution", "-begin", "-end", "-frozen"],
    related: RELATED, sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS,
    links: LINKS, media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Search-path mutation runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/addpath/mod.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Arguments, ordering, validation, compatibility, and atomicity", location: "builtins::io::repl_fs::addpath::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Immediate callable resolution and session isolation", location: "runmat-core search-path tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Isolated filesystem examples", location: "scripts/runtime/verify-builtin-examples.mjs" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
