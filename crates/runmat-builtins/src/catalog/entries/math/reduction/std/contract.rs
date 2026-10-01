use crate::*;
pub const STD_TYPED_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "std-typed-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "std with a typed-integer normalization or dimension control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StdTypedIntegerControlExtension"),
    };

pub const STD_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [STD_TYPED_INTEGER_CONTROL_EXTENSION];

const STD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Standard deviation reduction result.",
}];

const STD_PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single- or double-precision input array.",
};

const STD_PARAM_W: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "w",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("0"),
    description: "Normalization flag or empty placeholder ([]).",
};

const STD_PARAM_AXES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "axes",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("[]"),
    description: "Dimension selector, vector of dimensions, or \"all\".",
};

const STD_PARAM_NANFLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nanflag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"includemissing\""),
    description:
        "Missing-value mode: \"includemissing\"/\"includenan\" or \"omitmissing\"/\"omitnan\".",
};

const STD_INPUTS_A: [BuiltinParamDescriptor; 1] = [STD_PARAM_A];
const STD_INPUTS_A_W: [BuiltinParamDescriptor; 2] = [STD_PARAM_A, STD_PARAM_W];
const STD_INPUTS_A_W_AXES: [BuiltinParamDescriptor; 3] = [STD_PARAM_A, STD_PARAM_W, STD_PARAM_AXES];
const STD_INPUTS_A_NANFLAG: [BuiltinParamDescriptor; 2] = [STD_PARAM_A, STD_PARAM_NANFLAG];
const STD_INPUTS_A_W_NANFLAG: [BuiltinParamDescriptor; 3] =
    [STD_PARAM_A, STD_PARAM_W, STD_PARAM_NANFLAG];
const STD_INPUTS_A_W_AXES_NANFLAG: [BuiltinParamDescriptor; 4] =
    [STD_PARAM_A, STD_PARAM_W, STD_PARAM_AXES, STD_PARAM_NANFLAG];
const STD_INPUTS_A_W_NANFLAG_AXES: [BuiltinParamDescriptor; 4] =
    [STD_PARAM_A, STD_PARAM_W, STD_PARAM_NANFLAG, STD_PARAM_AXES];
const STD_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "S = std(A)",
        inputs: &STD_INPUTS_A,
        outputs: &STD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = std(A, w)",
        inputs: &STD_INPUTS_A_W,
        outputs: &STD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = std(A, w, dim)",
        inputs: &STD_INPUTS_A_W_AXES,
        outputs: &STD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = std(A, w, vecdim)",
        inputs: &STD_INPUTS_A_W_AXES,
        outputs: &STD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = std(A, w, \"all\")",
        inputs: &STD_INPUTS_A_W_AXES,
        outputs: &STD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = std(A, nanflag)",
        inputs: &STD_INPUTS_A_NANFLAG,
        outputs: &STD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = std(A, w, nanflag)",
        inputs: &STD_INPUTS_A_W_NANFLAG,
        outputs: &STD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = std(A, w, dim, nanflag)",
        inputs: &STD_INPUTS_A_W_AXES_NANFLAG,
        outputs: &STD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = std(A, w, nanflag, dim)",
        inputs: &STD_INPUTS_A_W_NANFLAG_AXES,
        outputs: &STD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = std(A, w, [], nanflag)",
        inputs: &STD_INPUTS_A_W_AXES_NANFLAG,
        outputs: &STD_OUTPUT,
    },
];

pub const STD_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STD.INVALID_ARGUMENT",
    identifier: Some("RunMat:std:InvalidArgument"),
    when: "Argument grammar, dimensions, normalization, or missing-value selectors are invalid.",
    message: "std: invalid argument",
};

pub const STD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STD.INVALID_INPUT",
    identifier: Some("RunMat:std:InvalidInput"),
    when: "Input values are outside the supported floating std reduction domains, including integer data.",
    message: "std: invalid input",
};

pub const STD_ERROR_COMPLEX_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STD.COMPLEX_UNSUPPORTED",
    identifier: Some("RunMat:std:ComplexUnsupported"),
    when: "Input value is complex for a path that currently supports only real reductions.",
    message: "std: complex inputs are not supported yet",
};

pub const STD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STD.INTERNAL",
    identifier: Some("RunMat:std:Internal"),
    when: "Execution fails due to conversion, provider, upload, allocation, or fallback internals.",
    message: "std: internal reduction failure",
};

const STD_ERRORS: [BuiltinErrorDescriptor; 4] = [
    STD_ERROR_INVALID_ARGUMENT,
    STD_ERROR_INVALID_INPUT,
    STD_ERROR_COMPLEX_UNSUPPORTED,
    STD_ERROR_INTERNAL,
];

pub const STD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STD_ERRORS,
};

const REJECTED_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented data domain is single, double, datetime, duration, table, or timetable; every typed-integer data class is rejected before host or provider dispatch.",
    }];

const INTEGER_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "normalization_or_dimension",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode accepts exact host typed-integer normalization and dimension controls; MATLAB-compatible modes retain the documented floating control surface.",
    }];

pub const STD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "S = std(A, args...) with integer A",
        inputs: &REJECTED_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer data is invalid in every compatibility mode, including resident integer data, and no integer output contract exists.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "S = std(A, normalization_or_dimension, ...) with floating A",
        inputs: &INTEGER_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed-integer controls are a mode-gated RunMat extension; values are decoded exactly on the host, while the floating data reduction may still execute on a provider. The result class follows the floating input and execution path, not the integer control.",
    },
];
