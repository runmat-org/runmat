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
    description: "Figure, filename, and optional format.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "saveas(fig, filename)",
        inputs: &INPUTS,
        outputs: &NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "saveas(fig, filename, formattype)",
        inputs: &INPUTS,
        outputs: &NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ok = saveas(___)",
        inputs: &INPUTS,
        outputs: &STATUS_OUTPUT,
    },
];
pub const SAVEAS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &super::FIGURE_PERSISTENCE_ERRORS,
};
pub const SAVEAS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "saveas accepts a host graphics object, textual filename, and textual format token; numeric figure handles are opaque double identifiers rather than integer computation inputs, and resident numeric arguments reject before provider access or file I/O.",
};
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("saveas"), slug: Some("saveas"),
    summary: "Save a figure to a file.",
    description: "`saveas(fig, filename)` saves the specified figure or graphics object parent figure. RunMat supports RunMat FIG scene files and PNG image export.",
    keywords: &["saveas", "figure export", "figure save", "png", "fig"],
    related: &["savefig", "openfig", "print"],
    sections: &[
        BuiltinDocumentationSection { heading: "Forms and formats", paragraphs: &[
            "`saveas(fig,filename)` infers `.fig` or `.png`; a name without an extension uses FIG. `saveas(fig,filename,formattype)` selects the explicit format and appends its standard extension when none is present.",
            "Figure, axes, text, legend, ruler, and plot-child handles select their parent figure. PDF, EPS, SVG, JPEG, TIFF, and MATLAB code-generation formats are recognized but unavailable in the current exporter.",
        ]},
        BuiltinDocumentationSection { heading: "Execution and residency", paragraphs: &[
            "The call is an I/O and fusion boundary. It gathers handle, filename, and format metadata; integer plot data nested in the scene remains scene data. Direct typed-integer or gpuArray arguments reject before file I/O.",
            "PNG uses the shared plotting renderer. FIG uses RunMat's scene archive rather than claiming MATLAB FIG binary interchange. Unsupported export formats fail before any partial output is written. In RunMat compatibility mode, requesting one output returns a host logical completion status. MATLAB compatibility mode rejects that extension before file effects.",
        ]},
    ],
    examples: &[
        BuiltinExample { id: "figure-scene", title: "Save a figure scene", program: "plot(1:5,[1 4 2 5 3]);\nsaveas(gcf,\"plot_state.fig\");", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: super::PLOT_CORE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(exist('plot_state.fig','file') == 2);" } },
        BuiltinExample { id: "png", title: "Save a PNG image", program: "x=0:0.1:1;\nplot(x,x.^2);\nsaveas(gcf,\"quadratic\",\"png\");", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: super::PLOT_CORE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(exist('quadratic.png','file') == 2);" } },
    ],
    example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "savefig", target: BuiltinDocumentationLinkTarget::Builtin("savefig") },
        BuiltinDocumentationLink { label: "openfig", target: BuiltinDocumentationLinkTarget::Builtin("openfig") },
        BuiltinDocumentationLink { label: "print", target: BuiltinDocumentationLinkTarget::Builtin("print") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "saveas runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/figure_persistence/saveas/mod.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "FIG round-trip and PNG export regression coverage", location: "builtins::plotting::figure_persistence::tests" }],
        notes: &["All legacy sidecar descriptions, behaviors, GPU notes, examples, and related links are retained; maturity remains incomplete until the compatibility and staged-publication work executes."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::infer_fixed(request, entry, vec![super::status_fact()], 2, 3)
}
pub const SAVEAS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "saveas" },
    category: "plotting",
    documentation: DOCUMENTATION,
    descriptor: &SAVEAS_DESCRIPTOR,
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
    integer_audit: Some(&SAVEAS_INTEGER_AUDIT),
    suppress_auto_output: true,
};
