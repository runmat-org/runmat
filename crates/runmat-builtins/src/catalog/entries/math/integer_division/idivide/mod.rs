use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;

use documentation::DOCUMENTATION;

pub const IDIVIDE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BITWISE.INVALID_INPUT",
    identifier: Some("RunMat:bitwise:InvalidInput"),
    when: "Inputs are not supported integer arrays or compatible integer-valued scalar doubles.",
    message: "bitwise operation: invalid input",
};
pub const IDIVIDE_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BITWISE.SIZE_MISMATCH",
    identifier: Some("RunMat:bitwise:SizeMismatch"),
    when: "Operand shapes are not compatible for implicit expansion.",
    message: "bitwise operation: array sizes are not compatible",
};
pub const IDIVIDE_ERROR_DIVIDE_BY_ZERO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IDIVIDE.DIVIDE_BY_ZERO",
    identifier: Some("RunMat:idivide:DivideByZero"),
    when: "The divisor contains zero.",
    message: "idivide: division by zero",
};
pub const IDIVIDE_ERROR_OVERFLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IDIVIDE.OVERFLOW",
    identifier: Some("RunMat:idivide:Overflow"),
    when: "A rounded quotient cannot be represented in the dividend integer class.",
    message: "idivide: quotient overflows output class",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    IDIVIDE_ERROR_INVALID_INPUT,
    IDIVIDE_ERROR_SIZE_MISMATCH,
    IDIVIDE_ERROR_DIVIDE_BY_ZERO,
    IDIVIDE_ERROR_OVERFLOW,
];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Rounded quotient in the nondouble integer input class.",
}];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer dividend.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer divisor.",
    },
];
const ROUNDING_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer dividend.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer divisor.",
    },
    BuiltinParamDescriptor {
        name: "rounding",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"fix\""),
        description: "Rounding mode: fix, floor, ceil, or round.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = idivide(A, B)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = idivide(A, B, rounding)",
        inputs: &ROUNDING_INPUTS,
        outputs: &OUTPUTS,
    },
];
pub const IDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
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
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "A is an integer array or a compatible integer-valued scalar double.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "B is an integer array or a compatible integer-valued scalar double.",
    },
];
pub const IDIVIDE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "C = idivide(A, B, roundingMode)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Supports fix, floor, ceil, and round. Division by zero and the unrepresentable signed minimum divided by -1 produce structured errors.",
    }];
const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const IDIVIDE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "idivide" },
    category: "math/integer-division",
    documentation: DOCUMENTATION,
    descriptor: &IDIVIDE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::IntegerDivide),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::Elementwise,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::PreserveInputs,
        fusion: BuiltinFusionPolicy::Boundary,
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
    integer_capabilities: &IDIVIDE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
