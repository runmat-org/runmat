use crate::*;

pub(in super::super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("eval"),
    slug: Some("eval"),
    summary: "Evaluate source text in the current workspace.",
    description: "`eval` executes scalar source text in the current workspace through the active dynamic-source execution service. Its result and effects depend on that source and cannot be inferred from the text value's type alone.",
    keywords: &["eval", "workspace", "dynamic source"],
    related: &["evalc", "evalin", "assignin"],
    sections: &[BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &["The source may read or modify current-workspace bindings. Requested outputs are forwarded to the evaluated expression; no output class or shape is promised by `eval` itself.", "An active workspace execution service is required. A direct runtime builtin dispatch without one raises RunMat:DynamicWorkspaceRequiresVm."],
    }],
    examples: &[BuiltinExample {
        id: "current-workspace-assignment",
        title: "Update a current-workspace variable",
        program: "current_value = 4; eval('current_value = current_value + 3;');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(current_value == 7);" },
    }],
    example_exemption: None,
    faqs: &[],
    links: &[],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "VM dynamic workspace dispatch", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-vm/src/call/builtins.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Current-workspace evaluation", location: "crates/runmat-core/src/tests.rs dynamic workspace execution tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
