use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Generate a recursive folder list",
        paragraphs: &[
            "`genpath(folder)` returns a path-list character row containing the root and readable descendants in depth-first lexical order. Entries are canonical absolute paths separated by `pathsep`, so the result can be passed directly to `addpath` or `rmpath`.",
            "Folders named `private` or `resources`, class folders beginning with `@`, and namespace folders beginning with `+` are omitted with their descendants. If the supplied root itself has one of those names, it remains the root and is included.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Traversal, exclusions, and input policy",
        paragraphs: &[
            "Canonical folder identity prevents symbolic-link cycles and duplicate output entries. A descendant that cannot be inspected is skipped without discarding the rest of the result. A missing root or a root that is not a folder is an error.",
            "The no-input form traverses the current RunMat working folder. RunMat mode additionally accepts `genpath(folder, excludes)`, where `excludes` is a `pathsep`-delimited list resolved from the root, and dense real numeric character-code rows. MATLAB compatibility modes accept only the documented character-row and string-scalar inputs and reject the RunMat-only two-input form before filesystem or provider work.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "tree", title: "Generate a path for a folder tree", program: "mkdir('project');\nmkdir(fullfile('project', 'analysis'));\nmkdir(fullfile('project', 'analysis', 'helpers'));\np = genpath('project');\nassert(ischar(p));\nassert(contains(p, 'project'));\nassert(contains(p, 'analysis'));\nassert(contains(p, 'helpers'));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(p)); assert(contains(p, 'helpers'));" } },
    BuiltinExample { id: "add", title: "Add a generated tree to the search path", program: "mkdir('toolbox');\nmkdir(fullfile('toolbox', 'signal'));\noriginal = path;\naddpath(genpath('toolbox'));\nassert(contains(path, 'toolbox'));\nassert(contains(path, 'signal'));\npath(original);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(path, original));" } },
    BuiltinExample { id: "reserved", title: "Omit special folder categories", program: "mkdir('source');\nmkdir(fullfile('source', 'public'));\nmkdir(fullfile('source', 'private'));\nmkdir(fullfile('source', '+package'));\np = genpath('source');\nassert(contains(p, fullfile('source', 'public')));\nassert(~contains(p, fullfile('source', 'private')));\nassert(~contains(p, fullfile('source', '+package')));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(contains(p, fullfile('source', 'public'))); assert(~contains(p, fullfile('source', 'private'))); assert(~contains(p, fullfile('source', '+package')));" } },
    BuiltinExample { id: "string", title: "Use a string-scalar root", program: "mkdir('models');\nmkdir(fullfile('models', 'shared'));\np = genpath(\"models\");\nassert(contains(p, 'models'));\nassert(contains(p, 'shared'));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(contains(p, 'shared'));" } },
    BuiltinExample { id: "excludes", title: "Exclude a subtree in RunMat mode", program: "mkdir('solver');\nmkdir(fullfile('solver', 'keep'));\nmkdir(fullfile('solver', 'build'));\np = genpath('solver', 'build');\nassert(contains(p, 'keep'));\nassert(~contains(p, 'build'));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(contains(p, 'keep')); assert(~contains(p, 'build'));" } },
    BuiltinExample { id: "numeric-codes", title: "Use exact numeric character codes in RunMat mode", program: "mkdir('encoded');\np = genpath(uint16('encoded'));\nassert(contains(p, 'encoded'));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(contains(p, 'encoded'));" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Which folders are omitted automatically?", answer: "`private`, `resources`, class folders beginning with `@`, and namespace folders beginning with `+` are omitted with their descendants." },
    BuiltinDocumentationFaq { question: "What order does the result use?", answer: "The root appears first, followed by descendants in depth-first lexical order." },
    BuiltinDocumentationFaq { question: "How are duplicate folders handled?", answer: "Canonical folder identity removes duplicates and prevents symbolic-link cycles." },
    BuiltinDocumentationFaq { question: "Can I exclude additional folders?", answer: "In RunMat mode, pass a `pathsep`-delimited exclusion list as the second argument. Relative exclusions are resolved from the traversal root." },
    BuiltinDocumentationFaq { question: "Can the result be passed to addpath?", answer: "Yes. `genpath` returns a platform path list intended for `addpath` and `rmpath`." },
    BuiltinDocumentationFaq { question: "Does genpath run on an accelerator?", answer: "No. Traversal uses the active host or browser filesystem provider. RunMat-only resident numeric character codes are gathered after compatibility and argument admission." },
];

const RELATED: &[&str] = &[
    "addpath", "rmpath", "path", "pathsep", "which", "exist", "cd", "dir", "fullfile", "mkdir",
    "pwd",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "addpath", target: BuiltinDocumentationLinkTarget::Builtin("addpath") },
    BuiltinDocumentationLink { label: "rmpath", target: BuiltinDocumentationLinkTarget::Builtin("rmpath") },
    BuiltinDocumentationLink { label: "path", target: BuiltinDocumentationLinkTarget::Builtin("path") },
    BuiltinDocumentationLink { label: "pathsep", target: BuiltinDocumentationLinkTarget::Builtin("pathsep") },
    BuiltinDocumentationLink { label: "which", target: BuiltinDocumentationLinkTarget::Builtin("which") },
    BuiltinDocumentationLink { label: "exist", target: BuiltinDocumentationLinkTarget::Builtin("exist") },
    BuiltinDocumentationLink { label: "cd", target: BuiltinDocumentationLinkTarget::Builtin("cd") },
    BuiltinDocumentationLink { label: "dir", target: BuiltinDocumentationLinkTarget::Builtin("dir") },
    BuiltinDocumentationLink { label: "fullfile", target: BuiltinDocumentationLinkTarget::Builtin("fullfile") },
    BuiltinDocumentationLink { label: "mkdir", target: BuiltinDocumentationLinkTarget::Builtin("mkdir") },
    BuiltinDocumentationLink { label: "pwd", target: BuiltinDocumentationLinkTarget::Builtin("pwd") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/genpath/mod.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("genpath"),
    slug: Some("genpath"),
    summary: "Generate a path list from a folder tree.",
    description: "`genpath` traverses a folder tree and returns its eligible folders as a platform path-list character row.",
    keywords: &["genpath", "recursive path", "search path", "folder traversal", "addpath"],
    related: RELATED,
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Folder traversal runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/genpath/mod.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Inputs, traversal, exclusions, compatibility, and provider admission", location: "builtins::io::repl_fs::genpath::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Generated path function resolution", location: "runmat-core path-state tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Isolated filesystem examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
