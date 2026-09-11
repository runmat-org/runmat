use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Query or change the working folder",
        paragraphs: &[
            "`cd` with no input returns the current working folder as a character row vector. `cd(folder)` changes the working folder and returns the previous folder, which can be passed back to `cd` when the work is complete.",
            "The folder may be a character row vector or string scalar. Relative paths are resolved from the current folder. `~` and `~/...` expand from the user's home folder when that environment value is available.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Runtime environment",
        paragraphs: &[
            "Native execution updates the process working folder. Browser and embedded execution update the current folder maintained by the active virtual filesystem provider. Later relative file and source lookups observe the change.",
            "Folder text is host data. `cd` does not accept accelerator-resident numeric buffers, launch a device kernel, or participate in fusion.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "query", title: "Read the current folder", program: "current = cd;\nassert(ischar(current));\nassert(size(current, 1) == 1);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(current)); assert(size(current, 1) == 1);" } },
    BuiltinExample { id: "project-subfolder", title: "Work in a project subfolder and restore the original folder", program: "project_root = pwd;\nmkdir('data');\nmkdir(fullfile('data', 'logs'));\nold = cd(fullfile('data', 'logs'));\nassert(strcmp(old, project_root));\ncd(old);\nassert(strcmp(pwd, project_root));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(pwd, project_root));" } },
    BuiltinExample { id: "parent", title: "Navigate to the parent folder", program: "start_dir = pwd;\nold = cd('..');\nassert(strcmp(old, start_dir));\ncd(old);\nassert(strcmp(pwd, start_dir));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(pwd, start_dir));" } },
    BuiltinExample { id: "command-form", title: "Use command syntax", program: "project_root = pwd;\nmkdir('scratch');\ncd scratch\ncd ..\nassert(strcmp(pwd, project_root));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(pwd, project_root));" } },
    BuiltinExample { id: "home", title: "Change to the home folder", program: "start_dir = pwd;\nold = cd('~');\nassert(strcmp(old, start_dir));\ncd(old);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(pwd, start_dir));" } },
    BuiltinExample { id: "restore", title: "Capture and restore the previous folder", program: "start_dir = pwd;\nmkdir('results');\nold = cd('results');\nassert(strcmp(old, start_dir));\ncd(old);\nassert(strcmp(pwd, start_dir));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(pwd, start_dir));" } },
    BuiltinExample { id: "missing", title: "Handle a missing folder", program: "caught = false;\ntry\n    cd('missing-folder');\ncatch err\n    caught = true;\n    assert(~isempty(err.message));\nend\nassert(caught);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(caught);" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does cd return the new folder or the previous folder?", answer: "With an input, it returns the previous folder. Without an input, it returns the current folder." },
    BuiltinDocumentationFaq { question: "Can cd accept accelerator-resident values?", answer: "No. Folder arguments are character row vectors or string scalars, both of which are host values." },
    BuiltinDocumentationFaq { question: "Is tilde expansion supported?", answer: "Yes. `~` and paths beginning with `~/` or `~\\` use the current user's home folder when it can be resolved." },
    BuiltinDocumentationFaq { question: "How are relative paths resolved?", answer: "They are resolved from the current folder maintained by the active filesystem environment." },
    BuiltinDocumentationFaq { question: "What happens if the target does not exist?", answer: "RunMat raises a structured error containing the requested folder and the underlying filesystem detail." },
    BuiltinDocumentationFaq { question: "Can folder names contain spaces?", answer: "Yes. Pass the complete name as a string scalar or character row vector." },
];

const RELATED: &[&str] = &[
    "pwd", "ls", "dir", "fileread", "addpath", "copyfile", "delete", "exist", "fullfile",
    "genpath", "getenv", "mkdir", "movefile", "path", "rmdir", "rmpath", "savepath", "setenv",
    "tempdir", "tempname",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "pwd", target: BuiltinDocumentationLinkTarget::Builtin("pwd") },
    BuiltinDocumentationLink { label: "ls", target: BuiltinDocumentationLinkTarget::Builtin("ls") },
    BuiltinDocumentationLink { label: "dir", target: BuiltinDocumentationLinkTarget::Builtin("dir") },
    BuiltinDocumentationLink { label: "fileread", target: BuiltinDocumentationLinkTarget::Builtin("fileread") },
    BuiltinDocumentationLink { label: "addpath", target: BuiltinDocumentationLinkTarget::Builtin("addpath") },
    BuiltinDocumentationLink { label: "copyfile", target: BuiltinDocumentationLinkTarget::Builtin("copyfile") },
    BuiltinDocumentationLink { label: "delete", target: BuiltinDocumentationLinkTarget::Builtin("delete") },
    BuiltinDocumentationLink { label: "exist", target: BuiltinDocumentationLinkTarget::Builtin("exist") },
    BuiltinDocumentationLink { label: "fullfile", target: BuiltinDocumentationLinkTarget::Builtin("fullfile") },
    BuiltinDocumentationLink { label: "genpath", target: BuiltinDocumentationLinkTarget::Builtin("genpath") },
    BuiltinDocumentationLink { label: "getenv", target: BuiltinDocumentationLinkTarget::Builtin("getenv") },
    BuiltinDocumentationLink { label: "mkdir", target: BuiltinDocumentationLinkTarget::Builtin("mkdir") },
    BuiltinDocumentationLink { label: "movefile", target: BuiltinDocumentationLinkTarget::Builtin("movefile") },
    BuiltinDocumentationLink { label: "path", target: BuiltinDocumentationLinkTarget::Builtin("path") },
    BuiltinDocumentationLink { label: "rmdir", target: BuiltinDocumentationLinkTarget::Builtin("rmdir") },
    BuiltinDocumentationLink { label: "rmpath", target: BuiltinDocumentationLinkTarget::Builtin("rmpath") },
    BuiltinDocumentationLink { label: "savepath", target: BuiltinDocumentationLinkTarget::Builtin("savepath") },
    BuiltinDocumentationLink { label: "setenv", target: BuiltinDocumentationLinkTarget::Builtin("setenv") },
    BuiltinDocumentationLink { label: "tempdir", target: BuiltinDocumentationLinkTarget::Builtin("tempdir") },
    BuiltinDocumentationLink { label: "tempname", target: BuiltinDocumentationLinkTarget::Builtin("tempname") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/cd/mod.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("cd"), slug: Some("cd"),
    summary: "Query or change the current working folder.",
    description: "`cd` returns the current folder or changes it and returns the previous folder.",
    keywords: &["cd", "change directory", "current folder", "working directory", "pwd"], related: RELATED,
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS, media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Working-folder runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/cd/mod.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Query, mutation, path admission, and structured errors", location: "builtins::io::repl_fs::cd::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Native filesystem and browser examples", location: "scripts/runtime/verify-builtin-examples.mjs" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
