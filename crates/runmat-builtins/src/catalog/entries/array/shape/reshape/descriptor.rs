use crate::*;

pub const RESHAPE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array data with reshaped dimensions.",
}];

pub const RESHAPE_INPUTS_SZ: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array/value.",
    },
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Row/column size vector. May include one auto-sized dimension ([]).",
    },
];

pub const RESHAPE_INPUTS_DIMS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array/value.",
    },
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First output dimension size (or [] for auto).",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second output dimension size (or [] for auto).",
    },
];

pub const RESHAPE_INPUTS_DIMS_VARIADIC: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array/value.",
    },
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First output dimension size (or [] for auto).",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second output dimension size (or [] for auto).",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional output dimension sizes (or [] for auto).",
    },
];

pub const RESHAPE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "B = reshape(A, sz)",
        inputs: &RESHAPE_INPUTS_SZ,
        outputs: &RESHAPE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = reshape(A, m, n)",
        inputs: &RESHAPE_INPUTS_DIMS,
        outputs: &RESHAPE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = reshape(A, m, n, p...)",
        inputs: &RESHAPE_INPUTS_DIMS_VARIADIC,
        outputs: &RESHAPE_OUTPUT,
    },
];

pub const RESHAPE_ERROR_SIZE_MISSING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RESHAPE.SIZE_MISSING",
    identifier: Some("RunMat:reshape:SizeMissing"),
    when: "No size vector/scalars are provided after A.",
    message: "reshape: size information missing",
};

pub const RESHAPE_ERROR_INVALID_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RESHAPE.INVALID_SIZE",
    identifier: Some("RunMat:reshape:InvalidSize"),
    when:
        "Size arguments are malformed, non-scalar where scalar required, or contain invalid values.",
    message: "reshape: invalid size specification",
};

pub const RESHAPE_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RESHAPE.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:reshape:UnsupportedInput"),
    when: "Input value type is unsupported for reshape.",
    message: "reshape: unsupported input type",
};

pub const RESHAPE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    RESHAPE_ERROR_SIZE_MISSING,
    RESHAPE_ERROR_INVALID_SIZE,
    RESHAPE_ERROR_UNSUPPORTED_INPUT,
];

pub const RESHAPE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RESHAPE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RESHAPE_ERRORS,
};

pub const RESHAPE_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents all eight integer classes and preserves the input datatype.",
    }];

pub const RESHAPE_INTEGER_SIZE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sz/m/n/dims",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed size controls are parsed exactly as nonnegative dimensions, with one optional inferred dimension.",
    }];

pub const RESHAPE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = reshape(integer_A, sz)",
        inputs: &RESHAPE_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Reshape changes metadata only; authoritative integer payload and resident handle ownership remain unchanged.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = reshape(A, integer_sz) or reshape(A, integer_m,integer_n,...)",
        inputs: &RESHAPE_INTEGER_SIZE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Size controls remain exact and checked multiplication proves element-count equality without binary64 conversion.",
    },
];
