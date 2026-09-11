mod documentation;

use crate::*;
use documentation::DOCUMENTATION;
use runmat_types::{EffectKind, ExecutionStackRequirement};

const FUNCTION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "func",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function applied once to each group.",
};
const DATA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X1,...,XN",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "One or more data arrays, or a table whose variables supply function inputs.",
};
const GROUPS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "G",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive consecutive group numbers, with NaN used to omit observations.",
};
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Vertically concatenated callback results, one output for each requested callback output.",
}];
const INPUTS: [BuiltinParamDescriptor; 3] = [FUNCTION, DATA, GROUPS];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[Y1,...,YM] = splitapply(func,X1,...,XN,G)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const SPLITAPPLY_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLITAPPLY.INVALID_INPUT",
    identifier: Some("RunMat:splitapply:InvalidInput"),
    when: "The function, data inputs, or group numbers do not satisfy the splitapply contract.",
    message: "splitapply: invalid input",
};
pub const SPLITAPPLY_ERROR_CALLBACK: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLITAPPLY.CALLBACK",
    identifier: Some("RunMat:splitapply:CallbackFailed"),
    when: "The group function fails or returns the wrong number of outputs.",
    message: "splitapply: callback failed",
};
pub const SPLITAPPLY_ERROR_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLITAPPLY.OUTPUT",
    identifier: Some("RunMat:splitapply:IncompatibleOutput"),
    when: "Callback results cannot be concatenated consistently across groups.",
    message: "splitapply: incompatible callback outputs",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    SPLITAPPLY_ERROR_INVALID_INPUT,
    SPLITAPPLY_ERROR_CALLBACK,
    SPLITAPPLY_ERROR_OUTPUT,
];
pub const SPLITAPPLY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const SPLITAPPLY_INTEGER_GROUP_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "splitapply-integer-group-vector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "splitapply with fixed-width integer group-number storage is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:SplitapplyIntegerGroupVectorExtension"),
    };
pub const SPLITAPPLY_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [SPLITAPPLY_INTEGER_GROUP_EXTENSION];

const INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X1,...,XN",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Group slicing preserves each data input's fixed-width integer class and exact payload.",
}];
const INTEGER_GROUPS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "G",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Fixed-width group numbers are decoded exactly and require RunMat compatibility mode.",
}];
pub const SPLITAPPLY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = splitapply(func,integer_X1,...,G)",
        inputs: &INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The callback receives native-class group slices; its declared result controls output class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = splitapply(func,X1,...,integer_G)",
        inputs: &INTEGER_GROUPS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Group values are validated as the consecutive sequence 1 through N before callback execution.",
    },
];

const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 3] = [
    EffectKind::HostCallback,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
pub const SPLITAPPLY_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "splitapply" },
    category: "array/grouping",
    documentation: DOCUMENTATION,
    descriptor: &SPLITAPPLY_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::DynamicByDesign,
        inference_rule: BuiltinInferenceRule::Array(ArrayInferenceRule::Grouping(
            GroupingInferenceRule::GroupedApply,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Never,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Process,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &SPLITAPPLY_EXTENSIONS,
    integer_capabilities: &SPLITAPPLY_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
