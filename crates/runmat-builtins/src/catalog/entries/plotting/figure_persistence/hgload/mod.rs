use crate::*;
use runmat_types::*;

const HANDLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Imported top-level graphics handles.",
}];
const TWO_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Imported top-level graphics handles.",
    },
    BuiltinParamDescriptor {
        name: "old_prop_values",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Cell array of previous top-level property values replaced by the override structure.",
    },
];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Filename and optional top-level property override structure.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "h = hgload(filename)",
        inputs: &INPUTS,
        outputs: &HANDLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[h, old_prop_values] = hgload(filename, property_structure)",
        inputs: &INPUTS,
        outputs: &TWO_OUTPUTS,
    },
];
pub const HGLOAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &super::FIGURE_PERSISTENCE_ERRORS,
};
pub const HGLOAD_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "hgload accepts a textual filename and an optional property structure and returns opaque graphics handles plus property metadata; it has no direct integer data or control argument." };
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("hgload"), slug: Some("hgload"), summary: "Load graphics object hierarchies from RunMat FIG files.",
    description: "`hgload(filename)` loads top-level graphics objects from a RunMat FIG file. The two-output form returns previous property values for overridden top-level figure properties.",
    keywords: &["hgload", "graphics load", "fig", "plotting"], related: &["hgsave", "openfig"],
    sections: &[
        BuiltinDocumentationSection { heading: "Forms", paragraphs: &[
            "A missing `.fig` extension is appended. `[h,old]=hgload(filename,property_structure)` applies recognized top-level figure overrides after import. Unrecognized fields are ignored and omitted from `old`, matching the current documented behavior.",
        ]},
        BuiltinDocumentationSection { heading: "Execution and residency", paragraphs: &[
            "The operation reads host or virtual-filesystem bytes and creates host plotting state. Numeric and GPU-resident arguments reject without gather; no numeric GPU kernel is launched. RunMat FIG is a RunMat scene archive rather than a claim of MATLAB FIG binary interchange.",
            "The operation constructs a transient zero-, one-, or two-value result before committing imported figures. Invalid output cardinality or property restoration failure leaves no partially imported hierarchy.",
        ]},
    ],
    examples: &[BuiltinExample { id: "override", title: "Load and override a figure name", program: "plot(1:3);\nhgsave(gcf,\"named.fig\");\n[h,old]=hgload(\"named.fig\",struct(\"Name\",\"Loaded\"));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: super::PLOT_CORE, verification: BuiltinExampleVerification::Figure { minimum_figures: 2, assertions: "assert(ishandle(h)); assert(strcmp(get(h,'Name'),'Loaded'));" } }],
    example_exemption: None, faqs: &[], links: &[
        BuiltinDocumentationLink { label: "hgsave", target: BuiltinDocumentationLinkTarget::Builtin("hgsave") },
        BuiltinDocumentationLink { label: "openfig", target: BuiltinDocumentationLinkTarget::Builtin("openfig") },
    ], media: &[], evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "hgload runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/figure_persistence/hgload/mod.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "hgload property override regression coverage", location: "builtins::plotting::figure_persistence::tests" }],
        notes: &["The internal operation uses `ValueSequence`; the registered callable retains the workspace's legacy value projection until the shared callable ABI migration reaches this builtin."],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::infer_fixed(
        request,
        entry,
        vec![
            super::handle_fact(),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        1,
        2,
    )
}
pub const HGLOAD_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "hgload" },
    category: "plotting",
    documentation: DOCUMENTATION,
    descriptor: &HGLOAD_DESCRIPTOR,
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
    integer_audit: Some(&HGLOAD_INTEGER_AUDIT),
    suppress_auto_output: false,
};
