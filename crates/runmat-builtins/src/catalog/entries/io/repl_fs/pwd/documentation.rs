use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Working-folder query",
        paragraphs: &[
            "`pwd` returns the current working folder as a character row vector. The path uses the active platform's native representation and separators.",
            "A successful `cd` call changes the value returned by subsequent `pwd` calls. Reading the folder does not change workspace values, the search path, or the working folder itself.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution and placement",
        paragraphs: &[
            "The runtime asks its filesystem environment for the current directory. Native hosts query the process environment; browser hosts query the virtual filesystem environment supplied to the WebAssembly runtime.",
            "`pwd` is a host operation and does not invoke an accelerator provider, gather device values, or participate in fusion.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "current-folder",
        title: "Read the current working folder",
        program: "current = pwd;\nassert(ischar(current));\nassert(size(current, 1) == 1);\ndisp(current)",
        display_output: Some("The absolute path of the current working folder."),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(ischar(current)); assert(size(current, 1) == 1);",
        },
    },
    BuiltinExample {
        id: "restore-folder",
        title: "Restore a folder after completing work",
        program: "start_dir = pwd;\nmkdir('results');\ncd('results');\nassert(strcmp(pwd, fullfile(start_dir, 'results')));\ncd(start_dir);\nassert(strcmp(pwd, start_dir));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(pwd, start_dir));",
        },
    },
    BuiltinExample {
        id: "cd-previous-folder",
        title: "Use the previous folder returned by cd",
        program: "start_dir = pwd;\nprevious = cd('..');\nassert(strcmp(previous, start_dir));\ncd(previous);\nassert(strcmp(pwd, start_dir));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(pwd, start_dir));",
        },
    },
    BuiltinExample {
        id: "script-location",
        title: "Print the folder where a script starts",
        program: "fprintf('Script started in %s\\n', pwd)",
        display_output: Some("Script started in <current folder>"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Succeeds,
    },
    BuiltinExample {
        id: "write-location-log",
        title: "Write the working folder to a log",
        program: "current = pwd;\nfid = fopen('run.log', 'w');\nfprintf(fid, 'Working folder: %s\\n', current);\nfclose(fid);\ncontents = fileread('run.log');\nassert(contains(contents, current));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(contains(contents, current));",
        },
    },
    BuiltinExample {
        id: "handle-query-error",
        title: "Handle an unavailable working folder",
        program: "try\n    location = pwd;\n    assert(ischar(location));\ncatch err\n    disp(err.message);\nend",
        display_output: Some("Returns a path or reports why the host could not resolve it."),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Succeeds,
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Why does pwd return a character vector instead of a string scalar?", answer: "The compatible return type is a character row vector. Use `string(pwd)` when a string scalar is more convenient." },
    BuiltinDocumentationFaq { question: "Does pwd reflect changes made with cd?", answer: "Yes. Each call reads the current folder from the active runtime environment." },
    BuiltinDocumentationFaq { question: "Can pwd fail?", answer: "Yes. The host can report that the current directory is unavailable, for example after an external process removes it. RunMat returns a structured runtime error containing the host detail." },
    BuiltinDocumentationFaq { question: "Does pwd normalize the path?", answer: "RunMat returns the path supplied by the active filesystem environment. Platform-native separators and representation are preserved." },
    BuiltinDocumentationFaq { question: "Does pwd affect GPU-resident values?", answer: "No. It has no data inputs and performs no accelerator operation or transfer." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "cd", target: BuiltinDocumentationLinkTarget::Builtin("cd") },
    BuiltinDocumentationLink { label: "addpath", target: BuiltinDocumentationLinkTarget::Builtin("addpath") },
    BuiltinDocumentationLink { label: "copyfile", target: BuiltinDocumentationLinkTarget::Builtin("copyfile") },
    BuiltinDocumentationLink { label: "delete", target: BuiltinDocumentationLinkTarget::Builtin("delete") },
    BuiltinDocumentationLink { label: "dir", target: BuiltinDocumentationLinkTarget::Builtin("dir") },
    BuiltinDocumentationLink { label: "exist", target: BuiltinDocumentationLinkTarget::Builtin("exist") },
    BuiltinDocumentationLink { label: "fullfile", target: BuiltinDocumentationLinkTarget::Builtin("fullfile") },
    BuiltinDocumentationLink { label: "genpath", target: BuiltinDocumentationLinkTarget::Builtin("genpath") },
    BuiltinDocumentationLink { label: "getenv", target: BuiltinDocumentationLinkTarget::Builtin("getenv") },
    BuiltinDocumentationLink { label: "ls", target: BuiltinDocumentationLinkTarget::Builtin("ls") },
    BuiltinDocumentationLink { label: "mkdir", target: BuiltinDocumentationLinkTarget::Builtin("mkdir") },
    BuiltinDocumentationLink { label: "movefile", target: BuiltinDocumentationLinkTarget::Builtin("movefile") },
    BuiltinDocumentationLink { label: "path", target: BuiltinDocumentationLinkTarget::Builtin("path") },
    BuiltinDocumentationLink { label: "rmdir", target: BuiltinDocumentationLinkTarget::Builtin("rmdir") },
    BuiltinDocumentationLink { label: "rmpath", target: BuiltinDocumentationLinkTarget::Builtin("rmpath") },
    BuiltinDocumentationLink { label: "savepath", target: BuiltinDocumentationLinkTarget::Builtin("savepath") },
    BuiltinDocumentationLink { label: "setenv", target: BuiltinDocumentationLinkTarget::Builtin("setenv") },
    BuiltinDocumentationLink { label: "tempdir", target: BuiltinDocumentationLinkTarget::Builtin("tempdir") },
    BuiltinDocumentationLink { label: "tempname", target: BuiltinDocumentationLinkTarget::Builtin("tempname") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/pwd/mod.rs") },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Working-folder runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/pwd/mod.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Current-folder values, shape, mutation visibility, and errors", location: "builtins::io::repl_fs::pwd::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Native and browser documentation examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pwd"),
    slug: Some("pwd"),
    summary: "Return the current working folder.",
    description: "`pwd` returns the active runtime environment's current working folder as a character row vector.",
    keywords: &["pwd", "current directory", "working folder", "present working directory"],
    related: &[
        "cd", "addpath", "copyfile", "delete", "dir", "exist", "fullfile", "genpath",
        "getenv", "ls", "mkdir", "movefile", "path", "rmdir", "rmpath", "savepath", "setenv",
        "tempdir", "tempname",
    ],
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
