use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

mod documentation;
mod inference;
#[cfg(test)]
mod tests;

const DOCUMENT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "document",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "XML text, lightweight document/element, or scalar struct.",
}];
const FILE_DOCUMENT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Destination in the active filesystem provider.",
    },
    BuiltinParamDescriptor {
        name: "document",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "XML text, lightweight document/element, or scalar struct.",
    },
];
const TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "xml",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Serialized XML character row.",
}];
const BYTES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bytes",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of UTF-8 bytes written.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "xml = xmlwrite(document)",
        inputs: &DOCUMENT,
        outputs: &TEXT,
    },
    BuiltinSignatureDescriptor {
        label: "bytes = xmlwrite(filename, document)",
        inputs: &FILE_DOCUMENT,
        outputs: &BYTES,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    super::diagnostics::error(
        "RM.XMLWRITE.INVALID_ARGUMENT",
        "RunMat:xmlwrite:InvalidArgument",
        "The document, filename, input count, or output count is invalid.",
        "xmlwrite: invalid argument",
    ),
    super::diagnostics::error(
        "RM.XMLWRITE.INVALID_DOCUMENT",
        "RunMat:xmlwrite:InvalidDocument",
        "A document, element, or struct field cannot be serialized.",
        "xmlwrite: invalid XML document",
    ),
    super::diagnostics::error(
        "RM.XMLWRITE.IO",
        "RunMat:xmlwrite:IoFailed",
        "Filesystem capture, write, or authority validation fails.",
        "xmlwrite: filesystem operation failed",
    ),
    super::diagnostics::error(
        "RM.XMLWRITE.CANCELLED",
        "RunMat:xmlwrite:Cancelled",
        "The invocation is cancelled during XML work or filesystem I/O.",
        "xmlwrite: execution was cancelled",
    ),
    super::diagnostics::error(
        "RM.XMLWRITE.EXECUTION",
        "RunMat:xmlwrite:ExecutionFailed",
        "The invocation execution service fails while processing XML.",
        "xmlwrite: execution service failed",
    ),
];
pub const XMLWRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const XMLWRITE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "xmlwrite" },
    category: "io/repl_fs",
    documentation: documentation::DOCUMENTATION,
    descriptor: &XMLWRITE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Filesystem,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::EnvironmentRead,
            EffectKind::FilesystemWrite,
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
