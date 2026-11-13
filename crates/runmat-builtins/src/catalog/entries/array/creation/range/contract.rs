//! Canonical range signatures, errors, extensions, and integer forms.
use crate::*;

const RANGE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Range result (max-min).",
}];

const RANGE_SIG_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input data.",
}];

const RANGE_SIG_DIM_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input data.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Reduction dimension.",
    },
];

const RANGE_SIG_VECDIM_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input data.",
    },
    BuiltinParamDescriptor {
        name: "vecdim",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Vector of reduction dimensions.",
    },
];

const RANGE_SIG_ALL_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input data.",
    },
    BuiltinParamDescriptor {
        name: "all_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"all\""),
        description: "Reduce over all elements.",
    },
];

const RANGE_SIG_NANFLAG_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input data.",
    },
    BuiltinParamDescriptor {
        name: "nanflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"includenan\""),
        description: "NaN handling flag ('omitnan'|'includenan').",
    },
];

const RANGE_SIG_DIM_NANFLAG_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input data.",
    },
    BuiltinParamDescriptor {
        name: "dim_or_vecdim",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Dimension selector.",
    },
    BuiltinParamDescriptor {
        name: "nanflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"includenan\""),
        description: "NaN handling flag ('omitnan'|'includenan').",
    },
];

const RANGE_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "y = range(X)",
        inputs: &RANGE_SIG_INPUT,
        outputs: &RANGE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "y = range(X, dim)",
        inputs: &RANGE_SIG_DIM_INPUTS,
        outputs: &RANGE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "y = range(X, vecdim)",
        inputs: &RANGE_SIG_VECDIM_INPUTS,
        outputs: &RANGE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "y = range(X, \"all\")",
        inputs: &RANGE_SIG_ALL_INPUTS,
        outputs: &RANGE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "y = range(X, nanflag)",
        inputs: &RANGE_SIG_NANFLAG_INPUTS,
        outputs: &RANGE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "y = range(X, dim_or_vecdim, nanflag)",
        inputs: &RANGE_SIG_DIM_NANFLAG_INPUTS,
        outputs: &RANGE_OUTPUT,
    },
];

const RANGE_ERRORS: [BuiltinErrorDescriptor; 5] = [
    BuiltinErrorDescriptor {
        code: "RM.RANGE.ALL_WITH_DIM",
        identifier: None,
        when: "'all' is combined with an explicit dimension.",
        message: "range: 'all' cannot be combined with an explicit dimension",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANGE.TOO_MANY_DIMS",
        identifier: None,
        when: "More than one explicit dimension selector is provided.",
        message: "range: too many dimension arguments",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANGE.INVALID_DIM",
        identifier: None,
        when: "Dimension arguments are non-integer, non-finite, or < 1.",
        message: "range: dimension must be >= 1",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANGE.DIM_ON_GPU",
        identifier: None,
        when: "Dimension selector is GPU-resident.",
        message: "range: dimension arguments must reside on the host (numeric or string)",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANGE.INVALID_DIM_TYPE",
        identifier: None,
        when: "Dimension selector has unsupported value type or shape.",
        message: "range: unsupported dimension argument type",
    },
];

pub const RANGE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RANGE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RANGE_ERRORS,
};

pub const INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "range-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "range with typed-integer input data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RangeIntegerDataExtension"),
};

pub const TYPED_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "range-typed-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "range with a typed-integer dimension control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RangeTypedIntegerControlExtension"),
    };

pub const EXPLICIT_NANFLAG_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "range-explicit-nanflag",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "range with an explicit NaN handling flag is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RangeExplicitNanflagExtension"),
};

pub const GPU_MULTI_AXIS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "range-gpu-all-or-vecdim",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "range with all or vecdim on a GPU input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RangeGpuMultiAxisExtension"),
};

pub const RANGE_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    INTEGER_DATA_EXTENSION,
    TYPED_INTEGER_CONTROL_EXTENSION,
    EXPLICIT_NANFLAG_EXTENSION,
    GPU_MULTI_AXIS_EXTENSION,
];

const INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented data domain is single, double, logical, datetime, or duration; RunMat mode additionally accepts all eight real integer classes.",
    }];

const INTEGER_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim_or_vecdim",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented dimension domain is single or double; exact typed-integer dimensions are a mode-gated RunMat extension.",
    }];

pub const RANGE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "y = range(X, dim_or_vecdim_or_all) with integer X",
        inputs: &INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat's integer-data extension selects exact same-class extrema and subtracts them in a widened signed or unsigned domain before producing double output; resident integer input gathers rather than using floating provider hooks.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = range(X, integer_dim_or_vecdim) with noninteger X",
        inputs: &INTEGER_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed-integer dimensions are decoded exactly and independently gated; range otherwise omits NaN by definition without an explicit flag.",
    },
];
