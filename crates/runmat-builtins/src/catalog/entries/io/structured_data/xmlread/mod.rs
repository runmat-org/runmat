use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

mod documentation;
mod inference;
#[cfg(test)]
mod tests;

const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "XML file in the active filesystem provider.",
}];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "document",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Lightweight org.w3c.dom.Document-compatible object.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "document = xmlread(filename)",
    inputs: &INPUT,
    outputs: &OUTPUT,
}];
const ERRORS: [BuiltinErrorDescriptor; 6] = [
    super::diagnostics::error(
        "RM.XMLREAD.INVALID_ARGUMENT",
        "RunMat:xmlread:InvalidArgument",
        "The filename, input count, or output count is invalid.",
        "xmlread: invalid argument",
    ),
    super::diagnostics::error(
        "RM.XMLREAD.MALFORMED_XML",
        "RunMat:xmlread:MalformedXml",
        "The source is not a supported well-formed XML document.",
        "xmlread: malformed XML",
    ),
    super::diagnostics::error(
        "RM.XMLREAD.PROJECTION",
        "RunMat:xmlread:ProjectionFailed",
        "A parsed XML node cannot become a document object.",
        "xmlread: XML projection failed",
    ),
    super::diagnostics::error(
        "RM.XMLREAD.IO",
        "RunMat:xmlread:IoFailed",
        "Filesystem capture, read, or authority validation fails.",
        "xmlread: filesystem operation failed",
    ),
    super::diagnostics::error(
        "RM.XMLREAD.CANCELLED",
        "RunMat:xmlread:Cancelled",
        "The invocation is cancelled during XML work or filesystem I/O.",
        "xmlread: execution was cancelled",
    ),
    super::diagnostics::error(
        "RM.XMLREAD.EXECUTION",
        "RunMat:xmlread:ExecutionFailed",
        "The invocation execution service fails while processing XML.",
        "xmlread: execution service failed",
    ),
];
pub const XMLREAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const XMLREAD_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "xmlread accepts a scalar text filename; native integer filename classes reject before filesystem or provider access.",
};
pub const XMLREAD_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "xmlread" },
    category: "io/repl_fs",
    documentation: documentation::DOCUMENTATION,
    descriptor: &XMLREAD_DESCRIPTOR,
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
    integer_audit: Some(&XMLREAD_INTEGER_AUDIT),
    suppress_auto_output: false,
};
