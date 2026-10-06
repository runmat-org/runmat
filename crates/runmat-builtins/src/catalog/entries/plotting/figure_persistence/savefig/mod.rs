use crate::*;
use runmat_types::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ok",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical completion status returned by the current RunMat implementation.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional figures, filename, and version token.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "ok = savefig(filename)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ok = savefig(fig, filename)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ok = savefig(fig, filename, version)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
];
pub const SAVEFIG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &super::FIGURE_PERSISTENCE_ERRORS,
};
pub const SAVEFIG_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "savefig accepts host Figure objects, textual filenames, and textual version tokens; typed-integer arrays are not figure handles and resident numeric arguments reject before provider access or file I/O." };
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("savefig"), slug: Some("savefig"),
    summary: "Save one or more figures as RunMat FIG files.",
    description: "`savefig` writes RunMat FIG files that can be reopened by `openfig` and `hgload`.",
    keywords: &["savefig", "figure save", "fig", "plotting"], related: &["openfig", "saveas", "hgsave"],
    sections: &[
        BuiltinDocumentationSection { heading: "Forms", paragraphs: &[
            "`savefig(filename)` saves the current figure and appends `.fig` when needed. `savefig(fig,filename)` accepts one figure or an array of figures. Non-`.fig` extensions are rejected.",
            "The current implementation accepts `-v6`, `-v7`, `-v7.3`, and `compact` tokens while writing the same RunMat scene archive. These tokens do not imply MATLAB FIG binary interchange; exact version semantics remain incomplete.",
        ]},
        BuiltinDocumentationSection { heading: "Execution and residency", paragraphs: &[
            "The call serializes plotting state, gathers only metadata arguments, launches no numeric GPU kernel, and returns a host logical status in the current implementation. Direct typed-integer or gpuArray arguments reject before file I/O.",
        ]},
    ],
    examples: &[BuiltinExample { id: "round-trip", title: "Save and reopen the current figure", program: "plot(1:4,[1 3 2 4]);\nsavefig(\"session_plot.fig\");\nfig=openfig(\"session_plot.fig\");", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: super::PLOT_CORE, verification: BuiltinExampleVerification::Figure { minimum_figures: 2, assertions: "assert(ishandle(fig));" } }],
    example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "openfig", target: BuiltinDocumentationLinkTarget::Builtin("openfig") },
        BuiltinDocumentationLink { label: "saveas", target: BuiltinDocumentationLinkTarget::Builtin("saveas") },
        BuiltinDocumentationLink { label: "hgsave", target: BuiltinDocumentationLinkTarget::Builtin("hgsave") },
    ],
    media: &[], evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "savefig runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/figure_persistence/savefig/mod.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Single and multiple figure round-trip tests", location: "builtins::plotting::figure_persistence::tests" }],
        notes: &["Legacy documentation is retained; maturity stays incomplete until default filename, output, version, staging, and multi-scene snapshot semantics are qualified."],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::infer_fixed(request, entry, vec![super::status_fact()], 1, 3)
}
pub const SAVEFIG_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "savefig" },
    category: "plotting",
    documentation: DOCUMENTATION,
    descriptor: &SAVEFIG_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Filesystem,
        workspace_effect: None,
        environment_effect: None,
        effects: super::SAVE_EFFECTS,
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
    integer_audit: Some(SAVEFIG_INTEGER_AUDIT),
    suppress_auto_output: true,
};
