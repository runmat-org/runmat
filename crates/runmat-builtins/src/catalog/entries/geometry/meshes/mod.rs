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
    name: "meshes",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-by-N cell of surface mesh structs.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "meshes = geometry.meshes(asset)",
    inputs: &INPUT,
    outputs: &OUTPUT,
}];
// The old runtime uses inspect-family operation/internal identifiers for this
// projection. Keep those public errors while moving the declaration.
pub const GEOMETRY_MESHES_ERROR_OPERATION: BuiltinErrorDescriptor =
    super::inspect::GEOMETRY_INSPECT_ERROR_OPERATION;
pub const GEOMETRY_MESHES_ERROR_INTERNAL: BuiltinErrorDescriptor =
    super::inspect::GEOMETRY_INSPECT_ERROR_INTERNAL;
pub const GEOMETRY_MESHES_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.MESHES.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:geometry:meshes:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "geometry.meshes: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    GEOMETRY_MESHES_ERROR_OPERATION,
    GEOMETRY_MESHES_ERROR_INTERNAL,
    GEOMETRY_MESHES_ERROR_OUTPUTS,
];
pub const GEOMETRY_MESHES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const GEOMETRY_MESHES_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "geometry.meshes accepts a geometry.Asset object; vertices and one-based triangle/range indices are double-valued host output, not typed-integer input." };

const STL: BuiltinExampleFixture = BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId { local_name: "triangle-stl" }, root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[BuiltinFilesystemEntry::File { relative_path: "part.stl", content: BuiltinFixtureContent::Utf8("solid tri\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid tri\n") }],
});
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("geometry.meshes"), slug: Some("geometry.meshes"),
    summary: "Project an imported asset's renderable surface meshes.",
    description: "`geometry.meshes(asset)` returns a one-by-N host cell array of mesh structs ready for patch-style rendering or topology inspection.",
    keywords: &["geometry", "meshes", "surface", "triangles", "vertices"], related: &["geometry.load", "geometry.listRegions", "patch"],
    sections: &[BuiltinDocumentationSection { heading: "Surface topology", paragraphs: &[
        "Each cell contains `mesh_id`, an M-by-3 double `vertices` matrix, N-by-3 one-based double `triangles` and `faces` matrices, and a `region_mappings` cell. Each mapping includes `region_id`, `entity_kind`, and one-based start/count `ranges`. The outer cell is 1-by-number-of-surface-meshes, including a 1-by-0 empty result.",
            "The receiver is a host `geometry.Asset`. This projection reads its retained topology; it does not reread the source file, invoke CAD import, or run a numeric provider kernel. `asset.meshes()` uses the same builtin through typed class method registration.",
        ] }],
    examples: &[BuiltinExample { id: "triangle-mesh", title: "Inspect imported STL topology", program: "asset = geometry.load('part.stl');\nmeshes = geometry.meshes(asset);\nsize(meshes)", display_output: Some("ans = [1 1]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: STL, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(iscell(meshes)); assert(isequal(size(meshes),[1 1])); assert(isequal(size(meshes{1}.vertices),[3 3])); assert(isequal(meshes{1}.faces,[1 2 3]));" } }],
    example_exemption: None, faqs: &[BuiltinDocumentationFaq { question: "Are face indices zero-based?", answer: "No. Public triangle and face indices, and mapping starts, are one-based doubles." }],
    links: &[BuiltinDocumentationLink { label: "geometry.load", target: BuiltinDocumentationLinkTarget::Builtin("geometry.load") }], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Mesh projection", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/geometry/conversion.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "STL patch topology and one-based faces", location: "crates/runmat-runtime/src/builtins/geometry/tests.rs" }], notes: &[] }, introduced: None, status: None,
};

pub const GEOMETRY_MESHES_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "geometry.meshes",
    },
    category: "geometry/assets",
    documentation: DOCUMENTATION,
    descriptor: &GEOMETRY_MESHES_DESCRIPTOR,
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
    integer_audit: Some(&GEOMETRY_MESHES_INTEGER_AUDIT),
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&GEOMETRY_MESHES_CATALOG_ENTRY];
