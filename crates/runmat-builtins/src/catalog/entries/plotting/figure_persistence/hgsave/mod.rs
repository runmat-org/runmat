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
    description: "Graphics hierarchy, filename, and optional version.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "hgsave(filename)",
        inputs: &INPUTS,
        outputs: &NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "hgsave(h, filename)",
        inputs: &INPUTS,
        outputs: &NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "hgsave(..., version)",
        inputs: &INPUTS,
        outputs: &NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ok = hgsave(___)",
        inputs: &INPUTS,
        outputs: &STATUS_OUTPUT,
    },
];
pub const HGSAVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &super::FIGURE_PERSISTENCE_ERRORS,
};
pub const HGSAVE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "hgsave accepts host graphics handles, a textual filename, and textual version tokens; numeric array contents are not an integer computation surface and resident tensors are rejected without gather." };
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("hgsave"), slug: Some("hgsave"), summary: "Save figures, or parent figures of graphics handles, as RunMat FIG files.",
    description: "`hgsave` saves the current figure, a figure handle, or the parent figures of graphics handles to RunMat FIG files. RunMat FIG persistence is figure-scene based; arbitrary graphics-subtree FIG files are not emitted yet.",
    keywords: &["hgsave", "graphics save", "fig", "plotting"], related: &["hgload", "savefig"],
    sections: &[
        BuiltinDocumentationSection { heading: "Forms", paragraphs: &[
            "`hgsave(filename)` saves the current figure. `hgsave(h,filename)` accepts scalar double graphics handles and handle arrays. An array cannot contain an object together with one of its ancestors or descendants. The typed roots retain input order during admission. The current archive still collapses each accepted root to its parent figure and stores the first occurrence of each parent, so arbitrary top-level graphics-subtree persistence remains incomplete.",
            "`-v6`, `-v7`, `-v7.3`, and `compact` select bounded encoding profiles inside RunMat's versioned scene archive. They do not establish MATLAB FIG binary interchange.",
        ]},
        BuiltinDocumentationSection { heading: "Execution and residency", paragraphs: &[
            "The operation is an I/O sink and fusion boundary. Typed-integer handle spellings and gpuArray numeric arguments reject before file effects. No numeric GPU kernel is launched; scene data uses the same exporter as `savefig`.",
            "In RunMat compatibility mode, requesting one output returns a host logical completion status. MATLAB compatibility mode rejects that extension before file effects.",
        ]},
    ],
    examples: &[BuiltinExample { id: "hierarchy", title: "Save a graphics hierarchy", program: "plot(1:4);\nhgsave(gcf,\"legacy_plot.fig\");", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: super::PLOT_CORE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(exist('legacy_plot.fig','file') == 2);" } }],
    example_exemption: None, faqs: &[], links: &[
        BuiltinDocumentationLink { label: "hgload", target: BuiltinDocumentationLinkTarget::Builtin("hgload") },
        BuiltinDocumentationLink { label: "savefig", target: BuiltinDocumentationLinkTarget::Builtin("savefig") },
    ], media: &[], evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "hgsave runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/figure_persistence/hgsave/mod.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "hgsave and hgload round-trip tests", location: "builtins::plotting::figure_persistence::tests" }],
        notes: &["Arbitrary graphics-subtree persistence remains incomplete. Local providers use staged replacement; remote publication depends on the provider acquiring an equivalent staged-output capability."],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::infer_fixed(request, entry, vec![super::status_fact()], 1, 3)
}
pub const HGSAVE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "hgsave" },
    category: "plotting",
    documentation: DOCUMENTATION,
    descriptor: &HGSAVE_DESCRIPTOR,
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
    integer_audit: Some(&HGSAVE_INTEGER_AUDIT),
    suppress_auto_output: true,
};
