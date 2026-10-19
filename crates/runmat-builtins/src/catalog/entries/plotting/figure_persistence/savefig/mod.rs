use crate::*;
use runmat_types::*;

const NO_OUTPUT: [BuiltinParamDescriptor; 0] = [];
const STATUS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ok",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat-only logical completion status when one output is requested.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional figures, filename, and version token.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "savefig(filename)",
        inputs: &INPUTS,
        outputs: &NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "savefig(fig, filename)",
        inputs: &INPUTS,
        outputs: &NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "savefig(fig, filename, version)",
        inputs: &INPUTS,
        outputs: &NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ok = savefig(___)",
        inputs: &INPUTS,
        outputs: &STATUS_OUTPUT,
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
            "`savefig` saves the current figure as `Untitled.fig`. `savefig(filename)` saves the current figure and appends `.fig` when needed. `savefig(fig,filename)` accepts one figure or an array of figures. Non-`.fig` extensions are rejected.",
            "`-v6`, `-v7`, `-v7.3`, and `compact` select bounded encoding profiles inside RunMat's versioned scene archive. The profiles control archive compression; they do not claim MATLAB FIG binary interchange.",
        ]},
        BuiltinDocumentationSection { heading: "Execution and residency", paragraphs: &[
            "The call serializes plotting state, reads host metadata arguments, and launches no numeric GPU kernel. Direct typed-integer or gpuArray arguments reject before file I/O. In RunMat compatibility mode, requesting one output returns a host logical completion status; MATLAB compatibility mode rejects that extension before file effects.",
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
        notes: &["The implementation captures every selected figure before its first suspension and publishes the staged archive atomically where the filesystem provider supports staged replacement. Remote staged publication remains a provider capability gap."],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::infer_fixed(request, entry, vec![super::status_fact()], 0, 3)
}
pub const SAVEFIG_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
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
    extensions: &super::FIGURE_PERSISTENCE_EXTENSIONS,
    integer_capabilities: &[],
    integer_audit: Some(&SAVEFIG_INTEGER_AUDIT),
    suppress_auto_output: true,
};
