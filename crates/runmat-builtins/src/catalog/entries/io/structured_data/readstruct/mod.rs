use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

mod documentation;
mod inference;
#[cfg(test)]
mod tests;

const FILE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "XML file in the active filesystem provider.",
}];
const FILE_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "XML file in the active filesystem provider.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "FileType, AttributeSuffix, or TextNodeName.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::PropertyValue,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "The corresponding scalar text value.",
    },
];
const RESULT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar struct projected from the XML root.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "s = readstruct(filename)",
        inputs: &FILE,
        outputs: &RESULT,
    },
    BuiltinSignatureDescriptor {
        label: "s = readstruct(filename, Name, Value, ...)",
        inputs: &FILE_OPTIONS,
        outputs: &RESULT,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 7] = [
    super::diagnostics::error(
        "RM.READSTRUCT.INVALID_ARGUMENT",
        "RunMat:readstruct:InvalidArgument",
        "The filename, output count, or name-value arity is invalid.",
        "readstruct: invalid argument",
    ),
    super::diagnostics::error(
        "RM.READSTRUCT.INVALID_OPTION",
        "RunMat:readstruct:InvalidOption",
        "An option name or value is unsupported.",
        "readstruct: unsupported option",
    ),
    super::diagnostics::error(
        "RM.READSTRUCT.MALFORMED_XML",
        "RunMat:readstruct:MalformedXml",
        "The source is not a supported well-formed XML document.",
        "readstruct: malformed XML",
    ),
    super::diagnostics::error(
        "RM.READSTRUCT.PROJECTION",
        "RunMat:readstruct:ProjectionFailed",
        "An XML node cannot be projected to a scalar struct.",
        "readstruct: XML projection failed",
    ),
    super::diagnostics::error(
        "RM.READSTRUCT.IO",
        "RunMat:readstruct:IoFailed",
        "Filesystem capture, read, or authority validation fails.",
        "readstruct: filesystem operation failed",
    ),
    super::diagnostics::error(
        "RM.READSTRUCT.CANCELLED",
        "RunMat:readstruct:Cancelled",
        "The invocation is cancelled during XML work or filesystem I/O.",
        "readstruct: execution was cancelled",
    ),
    super::diagnostics::error(
        "RM.READSTRUCT.EXECUTION",
        "RunMat:readstruct:ExecutionFailed",
        "The invocation execution service fails while processing XML.",
        "readstruct: execution service failed",
    ),
];
pub const READSTRUCT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const READSTRUCT_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "readstruct" },
    category: "io/repl_fs",
    documentation: documentation::DOCUMENTATION,
    descriptor: &READSTRUCT_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::DeterministicReadOnly,
        semantic_kind: BuiltinSemanticKind::Filesystem,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::EnvironmentRead,
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
        residency: BuiltinResidencyPolicy::GatherToHost,
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
    integer_audit: None,
    suppress_auto_output: false,
};
