use crate::*;

pub(in super::super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("evalin"),
    slug: Some("evalin"),
    summary: "Evaluate source text in a selected workspace.",
    description: "`evalin` selects the `base` or `caller` workspace and evaluates scalar source text there. Output facts and side effects depend on that source.",
    keywords: &["evalin", "base workspace", "caller workspace"],
    related: &["eval", "evalc", "assignin"],
    sections: &[BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &["The workspace selector is scalar text. `evalin` may read or modify bindings in the selected workspace; it does not prove that a dynamically named binding exists.", "An active workspace execution service is required. The selected source determines output count, class, shape, and further effects."],
    }],
    examples: &[BuiltinExample {
        id: "base-expression",
        title: "Read a base-workspace expression",
        program: "base_value = 7; selected = evalin('base', 'base_value + 1');",
        display_output: Some("selected = 8"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(selected == 8);" },
    }],
    example_exemption: None,
    faqs: &[],
    links: &[],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "VM selected-workspace dispatch", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-vm/src/call/builtins.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Base and caller workspace selection", location: "crates/runmat-core/src/tests.rs dynamic workspace evaluation tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
