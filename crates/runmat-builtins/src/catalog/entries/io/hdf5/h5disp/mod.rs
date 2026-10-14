use super::common::*;
use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

mod documentation;
mod inference;
#[cfg(test)]
mod tests;

pub const H5DISP_CAPTURE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "h5disp-captured-text-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "capturing h5disp display text is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:H5dispCapturedTextExtension"),
};

const FILENAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "HDF5 file name.",
};
const LOCATION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "location",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Group or dataset path.",
};
const MODE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "'simple' or 'min' display mode.",
};
const TEXT_ENCODING: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "TextEncoding",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "TextEncoding option name.",
};
const UTF8: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "UTF-8",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "UTF-8 name decoding.",
};
const IN_FILE_LOCATION_MODE: [BuiltinParamDescriptor; 3] = [FILENAME, LOCATION, MODE];
const IN_FILE_ENCODING: [BuiltinParamDescriptor; 3] = [FILENAME, TEXT_ENCODING, UTF8];
const IN_FILE_LOCATION_ENCODING: [BuiltinParamDescriptor; 4] =
    [FILENAME, LOCATION, TEXT_ENCODING, UTF8];
const IN_FILE_LOCATION_MODE_ENCODING: [BuiltinParamDescriptor; 5] =
    [FILENAME, LOCATION, MODE, TEXT_ENCODING, UTF8];

const SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "h5disp(filename)",
        inputs: &IN_FILE_ONLY,
        outputs: &OUT_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "h5disp(filename, location)",
        inputs: &IN_FILE_OPTIONAL_LOCATION,
        outputs: &OUT_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "h5disp(filename, location, mode)",
        inputs: &IN_FILE_LOCATION_MODE,
        outputs: &OUT_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "h5disp(filename, 'TextEncoding', 'UTF-8')",
        inputs: &IN_FILE_ENCODING,
        outputs: &OUT_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "h5disp(filename, location, 'TextEncoding', 'UTF-8')",
        inputs: &IN_FILE_LOCATION_ENCODING,
        outputs: &OUT_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "h5disp(filename, location, mode, 'TextEncoding', 'UTF-8')",
        inputs: &IN_FILE_LOCATION_MODE_ENCODING,
        outputs: &OUT_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "text = h5disp(filename)",
        inputs: &IN_FILE_ONLY,
        outputs: &OUT_TEXT,
    },
    BuiltinSignatureDescriptor {
        label: "text = h5disp(filename, location)",
        inputs: &IN_FILE_OPTIONAL_LOCATION,
        outputs: &OUT_TEXT,
    },
    BuiltinSignatureDescriptor {
        label: "text = h5disp(filename, location, mode)",
        inputs: &IN_FILE_LOCATION_MODE,
        outputs: &OUT_TEXT,
    },
    BuiltinSignatureDescriptor {
        label: "text = h5disp(filename, 'TextEncoding', 'UTF-8')",
        inputs: &IN_FILE_ENCODING,
        outputs: &OUT_TEXT,
    },
    BuiltinSignatureDescriptor {
        label: "text = h5disp(filename, location, 'TextEncoding', 'UTF-8')",
        inputs: &IN_FILE_LOCATION_ENCODING,
        outputs: &OUT_TEXT,
    },
    BuiltinSignatureDescriptor {
        label: "text = h5disp(filename, location, mode, 'TextEncoding', 'UTF-8')",
        inputs: &IN_FILE_LOCATION_MODE_ENCODING,
        outputs: &OUT_TEXT,
    },
];

pub const H5DISP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Only scalar text file and object paths are inputs; the result is display text.",
};

pub const H5DISP_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "h5disp" },
    category: "io/hdf5",
    documentation: documentation::DOCUMENTATION,
    descriptor: &H5DISP_DESCRIPTOR,
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
            EffectKind::FilesystemRead,
            EffectKind::UserInterface,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[
            CapabilityRequirement::Filesystem,
            CapabilityRequirement::NativeCode,
        ],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeOnly,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::NativeSymbol,
        execution_stack: ExecutionStackRequirement::Process,
        artifact_dependencies: &["hdf5-metno"],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[H5DISP_CAPTURE_EXTENSION],
    integer_capabilities: &[],
    integer_audit: Some(&INTEGER_AUDIT),
    suppress_auto_output: false,
};
