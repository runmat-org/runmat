use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor,
    BuiltinDistributedPolicy, BuiltinErrorDescriptor, BuiltinFusionPolicy, BuiltinInferenceRule,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, MathInferenceRule, RemainderFunction, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;
use documentation::REM_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Element-wise remainder result, retaining the supported numeric class or container.",
}];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real numeric, logical, character, table, timetable, or duration dividend.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Compatible real divisor or scalar expansion operand.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "R = rem(A, B)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const REM_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REM.INVALID_INPUT",
    identifier: Some("RunMat:rem:InvalidInput"),
    when: "An operand is complex or is not a supported real numeric, logical, character, table, timetable, or duration value.",
    message: "rem: invalid input",
};
pub const REM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REM.INVALID_ARGUMENT",
    identifier: Some("RunMat:rem:InvalidArgument"),
    when: "The invocation does not contain exactly two inputs.",
    message: "rem: invalid argument",
};
pub const REM_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REM.SIZE_MISMATCH",
    identifier: Some("RunMat:rem:SizeMismatch"),
    when: "Numeric operands are not broadcast-compatible or tabular operands do not have compatible variables.",
    message: "rem: array sizes are not compatible for implicit expansion",
};
pub const REM_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REM.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:rem:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "rem: too many output arguments",
};
pub const REM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REM.INTERNAL",
    identifier: Some("RunMat:rem:Internal"),
    when: "Internal conversion, allocation, provider execution, or residency restoration fails.",
    message: "rem: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    REM_ERROR_INVALID_INPUT,
    REM_ERROR_INVALID_ARGUMENT,
    REM_ERROR_SIZE_MISMATCH,
    REM_ERROR_TOO_MANY_OUTPUTS,
    REM_ERROR_INTERNAL,
];
pub const REM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer pairs use one matching class; a compatible scalar double may supply the other operand where documented.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer pairs use one matching class; a compatible scalar double may supply the other operand where documented.",
    },
];
pub const REM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "R = rem(A, B) with a fixed-width integer operand",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Exact native storage applies truncating-remainder semantics without a binary64 conversion.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const REM_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "rem" },
    category: "math/rounding",
    documentation: REM_DOCUMENTATION,
    descriptor: &REM_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Remainder(
            RemainderFunction::Remainder,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Candidate,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &[],
    integer_capabilities: &REM_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
