use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Installation root", paragraphs: &["`matlabroot()` returns the root of the active RunMat installation as a character row. Code that uses the installation root to locate bundled resources can compose paths with `fullfile`.", "A native session first honors an explicitly configured `RUNMAT_ROOT`, then uses the directory containing the running executable. Browser sessions use the root exposed by their virtual filesystem. The current working directory is a final fallback when the host cannot expose an installation location."] },
    BuiltinDocumentationSection { heading: "Execution", paragraphs: &["The query accepts no inputs, does not invoke an acceleration provider, and is available in interpreted, JIT-compiled, AOT-compiled, native, and browser execution."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "query",
        title: "Read the installation root",
        program: "root = matlabroot();",
        display_output: Some("root is a nonempty character row"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(ischar(root)); assert(size(root, 1) == 1); assert(~isempty(root));",
        },
    },
    BuiltinExample {
        id: "compose",
        title: "Build a path below the installation root",
        program: "resource = fullfile(matlabroot(), 'toolbox');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(ischar(resource)); assert(endsWith(resource, 'toolbox'));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does matlabroot return the current project folder?", answer: "No. It reports the RunMat installation root. Use `pwd` for the current working directory." },
    BuiltinDocumentationFaq { question: "Can a deployment select the installation root?", answer: "Yes. Native hosts can set `RUNMAT_ROOT` when the executable directory is not the desired resource root." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "pwd", target: BuiltinDocumentationLinkTarget::Builtin("pwd") },
    BuiltinDocumentationLink { label: "fullfile", target: BuiltinDocumentationLinkTarget::Builtin("fullfile") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/installation_path/matlabroot/mod.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlabroot"),
    slug: Some("matlabroot"),
    summary: "Return the active RunMat installation root.",
    description: "`matlabroot` returns the installation root as a character row for source compatibility with code that locates installation-relative resources.",
    keywords: &["matlabroot", "installation root", "resource path", "fullfile"],
    related: &["pwd", "fullfile", "path"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Installation-root runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/installation_path/matlabroot/mod.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Resolution order, output representation, and arity", location: "builtins::io::repl_fs::installation_path::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable native and browser examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
