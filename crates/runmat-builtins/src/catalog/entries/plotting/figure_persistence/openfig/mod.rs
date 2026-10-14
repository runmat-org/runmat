use crate::*;
use runmat_types::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fig",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Imported figure handle or row vector of handles.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Filename plus copy and visibility options.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "fig = openfig(filename)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "fig = openfig(filename, copies)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "fig = openfig(filename, visibility)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "fig = openfig(filename, copies, visibility)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
];
pub const OPENFIG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &super::FIGURE_PERSISTENCE_ERRORS,
};
pub const OPENFIG_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "openfig accepts a textual filename plus textual copy/visibility options and returns opaque figure handles; it has no direct integer data or control role, and resident numeric arguments reject before provider access or file I/O." };
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("openfig"), slug: Some("openfig"), summary: "Open figures saved in RunMat FIG files.",
    description: "`openfig(filename)` loads a RunMat FIG scene file and returns the imported figure handle or a row vector of handles.",
    keywords: &["openfig", "figure open", "fig", "plotting"], related: &["savefig", "hgload"],
    sections: &[
        BuiltinDocumentationSection { heading: "Forms", paragraphs: &[
            "The `.fig` extension is appended when absent. `new` imports another copy. `visible`/`invisible` (and `on`/`off`) override restored visibility.",
            "`new` imports another live copy. `reuse` returns the session-owned open copy for the same provider and canonical path when it still exists; otherwise it imports a new copy. Visibility options apply to both paths.",
        ]},
        BuiltinDocumentationSection { heading: "Execution and residency", paragraphs: &[
            "Filename, copy mode, and visibility are textual roles. Numeric and resident numeric arguments reject before provider access. The call reads host or virtual-filesystem bytes, creates host plotting state, launches no numeric GPU kernel, and can suspend during I/O/import.",
            "RunMat FIG stores RunMat scene archives; the interface does not claim MATLAB FIG binary interchange.",
        ]},
    ],
    examples: &[BuiltinExample { id: "visible-override", title: "Open an invisible figure as visible", program: "plot(1:5);\nset(gcf,\"Visible\",\"off\");\nsavefig(\"hidden_plot.fig\");\nfig=openfig(\"hidden_plot.fig\",\"visible\");", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: super::PLOT_CORE, verification: BuiltinExampleVerification::Figure { minimum_figures: 2, assertions: "assert(strcmp(get(fig,'Visible'),'on'));" } }],
    example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "savefig", target: BuiltinDocumentationLinkTarget::Builtin("savefig") },
        BuiltinDocumentationLink { label: "hgload", target: BuiltinDocumentationLinkTarget::Builtin("hgload") },
    ], media: &[], evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "openfig runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/figure_persistence/openfig/mod.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Single and multiple figure round-trip tests", location: "builtins::plotting::figure_persistence::tests" }],
        notes: &["The complete archive is decoded and validated before any figure becomes visible. Import failures retire every provisional handle and restore the previous current figure."],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::infer_fixed(request, entry, vec![super::handle_fact()], 1, 3)
}
pub const OPENFIG_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "openfig" },
    category: "plotting",
    documentation: DOCUMENTATION,
    descriptor: &OPENFIG_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Filesystem,
        workspace_effect: None,
        environment_effect: None,
        effects: super::LOAD_EFFECTS,
        capabilities: &[
            CapabilityRequirement::Filesystem,
            CapabilityRequirement::UserInterface,
        ],
    },
    placement: super::PLACEMENT,
    link: super::LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&OPENFIG_INTEGER_AUDIT),
    suppress_auto_output: true,
};
