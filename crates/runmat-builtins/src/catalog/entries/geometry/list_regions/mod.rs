use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};
mod inference;
#[cfg(test)]
mod tests;

const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "asset",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A host geometry.Asset returned by geometry.load.",
}];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "regions",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar struct with a regions field.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "regions = geometry.listRegions(asset)",
    inputs: &INPUT,
    outputs: &OUTPUT,
}];
// Existing listRegions execution uses these inspect-family identifiers for
// malformed asset payload and operation conversion. Preserve them at cutover.
pub const GEOMETRY_LIST_REGIONS_ERROR_OPERATION: BuiltinErrorDescriptor =
    super::inspect::GEOMETRY_INSPECT_ERROR_OPERATION;
pub const GEOMETRY_LIST_REGIONS_ERROR_INTERNAL: BuiltinErrorDescriptor =
    super::inspect::GEOMETRY_INSPECT_ERROR_INTERNAL;
pub const GEOMETRY_LIST_REGIONS_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.LIST_REGIONS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:geometry:listRegions:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "geometry.listRegions: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    GEOMETRY_LIST_REGIONS_ERROR_OPERATION,
    GEOMETRY_LIST_REGIONS_ERROR_INTERNAL,
    GEOMETRY_LIST_REGIONS_ERROR_OUTPUTS,
];
pub const GEOMETRY_LIST_REGIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const GEOMETRY_LIST_REGIONS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "geometry.listRegions accepts a geometry.Asset object and returns imported-region metadata; no direct integer input or control is admitted." };

const STL: BuiltinExampleFixture = BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId { local_name: "triangle-stl" }, root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[BuiltinFilesystemEntry::File { relative_path: "part.stl", content: BuiltinFixtureContent::Utf8("solid tri\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid tri\n") }],
});
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("geometry.listRegions"), slug: Some("geometry.listRegions"),
    summary: "List regions retained by an imported geometry asset.",
    description: "`geometry.listRegions(asset)` projects the imported regions into a scalar host struct with a `regions` field.",
    keywords: &["geometry", "regions", "mesh", "FEA"], related: &["geometry.load", "geometry.meshes", "fea.model"],
    sections: &[BuiltinDocumentationSection { heading: "Asset metadata", paragraphs: &[
        "The receiver must be a typed `geometry.Asset`; arbitrary structs or resident numeric arrays are not accepted. The returned scalar struct contains the asset's imported region records. Its region collection shape and fields depend on file contents, not on the static receiver class alone.",
        "The method form `asset.listRegions()` dispatches to the same canonical builtin. It reads the retained host asset payload but performs no filesystem read or numeric provider work. Invalid payload and conversion errors retain their underlying cause.",
    ] }],
    examples: &[BuiltinExample { id: "triangle-regions", title: "List imported STL regions", program: "asset = geometry.load('part.stl');\nregions = geometry.listRegions(asset);\nregions.regions.region_id", display_output: Some("ans = region_default"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: STL, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isstruct(regions)); assert(isfield(regions,'regions')); assert(isequal(numel(regions.regions),1)); assert(strcmp(regions.regions.region_id,'region_default'));" } }],
    example_exemption: None, faqs: &[BuiltinDocumentationFaq { question: "Does this reread the geometry file?", answer: "No. It projects region records from the supplied geometry.Asset." }],
    links: &[BuiltinDocumentationLink { label: "geometry.load", target: BuiltinDocumentationLinkTarget::Builtin("geometry.load") }], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Region projection", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/geometry/entrypoints.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Imported region projection", location: "crates/runmat-runtime/src/builtins/geometry/tests.rs" }], notes: &[] }, introduced: None, status: None,
};

pub const GEOMETRY_LIST_REGIONS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "geometry.listRegions",
    },
    category: "geometry/assets",
    documentation: DOCUMENTATION,
    descriptor: &GEOMETRY_LIST_REGIONS_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::RunMat,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[EffectKind::MayThrow],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::Host,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&GEOMETRY_LIST_REGIONS_INTEGER_AUDIT),
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&GEOMETRY_LIST_REGIONS_CATALOG_ENTRY];
