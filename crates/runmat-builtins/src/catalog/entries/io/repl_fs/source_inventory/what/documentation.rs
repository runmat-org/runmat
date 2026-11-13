use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("what"),
    slug: Some("what"),
    summary: "Summarize recognized source and data artifacts in a folder.",
    description: "`what` inspects the current folder or one named folder and returns a scalar structure that groups recognized source files, data files, binary extensions, class folders, and package folders.",
    keywords: &["what", "folder", "source files", "MAT files", "classes", "packages"],
    related: &["dir", "which", "path", "fullfile"],
    sections: &[
        BuiltinDocumentationSection { heading: "Selected folder", paragraphs: &["With no input, `what` inspects the current folder. The optional folder must be a character row or string scalar. Relative paths use the current folder and a leading `~` uses the active environment."] },
        BuiltinDocumentationSection { heading: "Returned groups", paragraphs: &["The scalar structure has `path`, `m`, `mat`, `mex`, `classes`, and `packages` fields. `path` identifies the inspected folder. The remaining fields are column cell arrays of character rows, sorted by name.", "Class and package names omit their leading `@` and `+` markers. The inventory covers direct children only; unrelated files and nested contents are not included."] },
        BuiltinDocumentationSection { heading: "Execution boundary", paragraphs: &["Enumeration uses RunMat's filesystem service on native and browser/WASM targets. Numeric and provider-resident folder values reject before filesystem or accelerator-provider access."] },
    ],
    examples: super::examples::EXAMPLES,
    example_exemption: None,
    faqs: super::faqs::FAQS,
    links: &[
        BuiltinDocumentationLink { label: "dir", target: BuiltinDocumentationLinkTarget::Builtin("dir") },
        BuiltinDocumentationLink { label: "which", target: BuiltinDocumentationLinkTarget::Builtin("which") },
        BuiltinDocumentationLink { label: "path", target: BuiltinDocumentationLinkTarget::Builtin("path") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Source inventory runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/source_inventory/what") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Classification, ordering, input, and provider behavior", location: "builtins::io::repl_fs::source_inventory::what::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable filesystem examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
