mod documentation;
pub(in crate::catalog::entries::io::repl_fs) mod inference;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

const OLD_PATH_OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "oldpath",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Current path for query form or previous path for mutation forms, as a character row vector.",
};
const OUTPUTS: [BuiltinParamDescriptor; 1] = [OLD_PATH_OUTPUT];
const PATH1: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "path1",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Replacement path as a character row, string scalar, or RunMat numeric character-code row.",
}];
const PATH1_PATH2: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "path1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First path fragment.",
    },
    BuiltinParamDescriptor {
        name: "path2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second path fragment.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "oldpath = path",
        inputs: &[],
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "oldpath = path(path1)",
        inputs: &PATH1,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "oldpath = path(path1, path2)",
        inputs: &PATH1_PATH2,
        outputs: &OUTPUTS,
    },
];

pub const PATH_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PATH.INVALID_INPUT",
    identifier: Some("RunMat:path:InvalidInput"),
    when: "A path argument is not an admitted character row, string scalar, or numeric character-code row.",
    message: "path: arguments must be character vectors or string scalars",
};
pub const PATH_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PATH.TOO_MANY_INPUTS",
    identifier: Some("RunMat:path:TooManyInputs"),
    when: "More than two input arguments are provided.",
    message: "path: too many input arguments",
};
pub const PATH_ERROR_PROVIDER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PATH.PROVIDER_FAILED",
    identifier: Some("RunMat:path:ProviderFailed"),
    when: "An admitted resident numeric character-code row cannot be gathered from its owner.",
    message: "path: unable to read resident character codes",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    PATH_ERROR_INVALID_INPUT,
    PATH_ERROR_TOO_MANY_INPUTS,
    PATH_ERROR_PROVIDER_FAILED,
];
pub const PATH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const PATH_NUMERIC_CHARACTER_CODES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "path-numeric-character-codes",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "path with a numeric character-code row is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PathNumericCharacterCodesExtension"),
    };
const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [PATH_NUMERIC_CHARACTER_CODES_EXTENSION];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "path",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "A dense real integer row may encode Unicode scalar values in RunMat mode.",
}];
const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "oldpath = path(integer_character_codes)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer code points are decoded exactly, invalid Unicode rejects, and the previous path is returned as a character row.",
    }];
const EFFECTS: [EffectKind; 4] = [
    EffectKind::EnvironmentRead,
    EffectKind::EnvironmentWrite,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];

pub const PATH_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "path" },
    category: "io/repl_fs",
    documentation: documentation::DOCUMENTATION,
    descriptor: &PATH_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Io(IoInferenceRule::ReplFs(
            IoReplFsInferenceRule::SearchPath(SearchPathInferenceRule::QueryOrReplace),
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Workspace,
        workspace_effect: None,
        environment_effect: Some(BuiltinEnvironmentEffect::PathMutation),
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
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
    extensions: &EXTENSIONS,
    integer_capabilities: &INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&PATH_CATALOG_ENTRY];
