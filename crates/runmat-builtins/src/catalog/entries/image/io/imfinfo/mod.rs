mod documentation;
mod inference;
#[cfg(test)]
mod tests;

use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

const INPUT_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar text path to an image file.",
}];
const INPUT_FORMAT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar text path to an image file.",
    },
    BuiltinParamDescriptor {
        name: "fmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Optional supported image format hint.",
    },
];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar metadata struct with file, format, dimensions and color fields.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "info = imfinfo(filename)",
        inputs: &INPUT_FILENAME,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "info = imfinfo(filename, fmt)",
        inputs: &INPUT_FORMAT,
        outputs: &OUTPUT,
    },
];
pub const IMFINFO_ERROR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMFINFO.INVALID_ARGUMENT",
    identifier: Some("RunMat:imfinfo:InvalidArgument"),
    when: "The arity or scalar text arguments are invalid.",
    message: "imfinfo: invalid argument",
};
pub const IMFINFO_ERROR_FORMAT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMFINFO.INVALID_FORMAT",
    identifier: Some("RunMat:imfinfo:InvalidFormat"),
    when: "An explicit image format hint is empty or unsupported.",
    message: "imfinfo: invalid format",
};
pub const IMFINFO_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMFINFO.IO",
    identifier: Some("RunMat:imfinfo:IoFailure"),
    when: "The image file cannot be read.",
    message: "imfinfo: file read failed",
};
pub const IMFINFO_ERROR_DECODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMFINFO.DECODE",
    identifier: Some("RunMat:imfinfo:DecodeFailure"),
    when: "The image bytes or format cannot be decoded.",
    message: "imfinfo: image decode failed",
};
pub const IMFINFO_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMFINFO.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:imfinfo:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "imfinfo: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    IMFINFO_ERROR_ARGUMENT,
    IMFINFO_ERROR_FORMAT,
    IMFINFO_ERROR_IO,
    IMFINFO_ERROR_DECODE,
    IMFINFO_ERROR_OUTPUTS,
];
pub const IMFINFO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const IMFINFO_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "imfinfo" },
    category: "image/io",
    documentation: documentation::DOCUMENTATION,
    descriptor: &IMFINFO_DESCRIPTOR,
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
    integer_audit: None,
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&IMFINFO_CATALOG_ENTRY];
