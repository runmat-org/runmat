mod documentation;
pub(in crate::catalog::entries::io::repl_fs) mod inference;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor { name: "folder", ty: BuiltinParamType::StringScalar, arity: BuiltinParamArity::Required, default: None, description: "Current folder for query form or previous folder for mutation form, as a character row vector." }];
const FOLDER: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "folder",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target folder as a character vector or string scalar.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "folder = cd",
        inputs: &[],
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "previous = cd(folder)",
        inputs: &FOLDER,
        outputs: &OUTPUTS,
    },
];

pub const CD_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CD.TOO_MANY_INPUTS",
    identifier: Some("RunMat:cd:TooManyInputs"),
    when: "More than one input is provided.",
    message: "cd: too many input arguments",
};
pub const CD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CD.INVALID_INPUT",
    identifier: Some("RunMat:cd:InvalidFolder"),
    when: "The folder is not a character row vector or string scalar.",
    message: "cd: folder name must be a character vector or string scalar",
};
pub const CD_ERROR_EMPTY_FOLDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CD.EMPTY_FOLDER",
    identifier: Some("RunMat:cd:EmptyFolder"),
    when: "The folder text is empty.",
    message: "cd: folder name must not be empty",
};
pub const CD_ERROR_CHANGE_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CD.CHANGE_FAILED",
    identifier: Some("RunMat:cd:ChangeFailed"),
    when: "The target folder cannot be entered.",
    message: "cd: unable to change directory",
};
pub const CD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CD.INTERNAL",
    identifier: Some("RunMat:cd:CurrentDirectoryUnavailable"),
    when: "The current folder cannot be resolved.",
    message: "cd: unable to determine current directory",
};
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    CD_ERROR_TOO_MANY_INPUTS,
    CD_ERROR_INVALID_INPUT,
    CD_ERROR_EMPTY_FOLDER,
    CD_ERROR_CHANGE_FAILED,
    CD_ERROR_INTERNAL,
];
pub const CD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const EFFECTS: [EffectKind; 5] = [
    EffectKind::EnvironmentRead,
    EffectKind::EnvironmentWrite,
    EffectKind::FilesystemRead,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];

pub const CD_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "cd" },
    category: "io/repl_fs",
    documentation: documentation::DOCUMENTATION,
    descriptor: &CD_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Io(IoInferenceRule::ReplFs(
            IoReplFsInferenceRule::ChangeDirectory,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Workspace,
        workspace_effect: None,
        environment_effect: Some(BuiltinEnvironmentEffect::WorkingDirectoryMutation),
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
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
    bindings: &crate::REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: None,
    suppress_auto_output: true,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&CD_CATALOG_ENTRY];
