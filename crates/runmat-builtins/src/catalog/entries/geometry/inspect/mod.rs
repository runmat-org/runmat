use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};
mod inference;
#[cfg(test)]
mod tests;

pub const GEOMETRY_INSPECT_RESULT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("geometry.InspectResult");
pub const GEOMETRY_INSPECT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "geometry.inspect accepts only a scalar text path; byte_count is double-valued output metadata.",
    };
const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "path",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar path to the geometry file.",
}];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "geometry.InspectResult with format and byte_count.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "info = geometry.inspect(path)",
    inputs: &INPUT,
    outputs: &OUTPUT,
}];
pub const GEOMETRY_INSPECT_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.INSPECT.IO",
    identifier: Some("RunMat:geometry:inspect:IoFailure"),
    when: "The geometry file cannot be read.",
    message: "geometry.inspect: failed to read geometry file",
};
pub const GEOMETRY_INSPECT_ERROR_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.INSPECT.INVALID_PATH",
    identifier: Some("RunMat:geometry:inspect:InvalidPath"),
    when: "The path is not scalar string text or a character row.",
    message: "geometry.inspect: invalid path",
};
pub const GEOMETRY_INSPECT_ERROR_OPERATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.INSPECT.OPERATION_FAILED",
    identifier: Some("RunMat:geometry:inspect:OperationFailed"),
    when: "The geometry inspection operation fails.",
    message: "geometry.inspect: operation failed",
};
pub const GEOMETRY_INSPECT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.INSPECT.INTERNAL",
    identifier: Some("RunMat:geometry:inspect:Internal"),
    when: "The inspection result cannot be converted to a RunMat value.",
    message: "geometry.inspect: internal error",
};
pub const GEOMETRY_INSPECT_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.INSPECT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:geometry:inspect:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "geometry.inspect: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    GEOMETRY_INSPECT_ERROR_INPUT,
    GEOMETRY_INSPECT_ERROR_IO,
    GEOMETRY_INSPECT_ERROR_OPERATION,
    GEOMETRY_INSPECT_ERROR_INTERNAL,
    GEOMETRY_INSPECT_ERROR_OUTPUTS,
];
pub const GEOMETRY_INSPECT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const STL: BuiltinExampleFixture = BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId { local_name: "triangle-stl" }, root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[BuiltinFilesystemEntry::File { relative_path: "part.stl", content: BuiltinFixtureContent::Utf8("solid tri\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid tri\n") }],
});
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("geometry.inspect"), slug: Some("geometry.inspect"),
    summary: "Inspect file format and byte count without importing a geometry asset.",
    description: "`geometry.inspect(path)` asynchronously reads bytes and reports the detected format and byte count in a host `geometry.InspectResult` object.",
    keywords: &["geometry", "inspect", "format", "bytes", "STL"], related: &["geometry.load", "geometry.meshes"],
    sections: &[BuiltinDocumentationSection { heading: "File inspection", paragraphs: &[
        "The path is scalar text. The operation reads the entire file, identifies its format from path and content, and returns `format` and `byte_count`; it does not build a mesh or load CAD topology. The byte count is a numeric metadata property, not an integer array input.",
        "The asynchronous filesystem read may suspend and requires an available filesystem service. It does not use a numeric accelerator. Browser execution depends on a filesystem bridge; native registration is not proof of access to browser-local files.",
    ] }],
    examples: &[BuiltinExample { id: "inspect-stl", title: "Inspect a triangle STL", program: "info = geometry.inspect('part.stl');\ninfo.format", display_output: Some("ans = stl"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: STL, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(info,'geometry.InspectResult')); assert(strcmp(info.format,'stl')); assert(info.byte_count > 0);" } }],
    example_exemption: None, faqs: &[BuiltinDocumentationFaq { question: "Does inspect import the geometry?", answer: "No. It reads bytes to detect format and size; use geometry.load to build an asset." }],
    links: &[BuiltinDocumentationLink { label: "geometry.load", target: BuiltinDocumentationLinkTarget::Builtin("geometry.load") }], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Inspection builtin", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/geometry/entrypoints.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Closed STL inspection", location: "crates/runmat-runtime/src/builtins/geometry/tests.rs" }], notes: &[] },
    introduced: None, status: None,
};

pub const GEOMETRY_INSPECT_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "geometry.inspect",
    },
    category: "geometry/assets",
    documentation: DOCUMENTATION,
    descriptor: &GEOMETRY_INSPECT_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::RunMat,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::DeterministicReadOnly,
        semantic_kind: BuiltinSemanticKind::Filesystem,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::FilesystemRead,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[CapabilityRequirement::Filesystem],
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
    integer_audit: Some(&GEOMETRY_INSPECT_INTEGER_AUDIT),
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&GEOMETRY_INSPECT_CATALOG_ENTRY];
