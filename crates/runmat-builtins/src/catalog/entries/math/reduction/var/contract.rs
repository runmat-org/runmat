use crate::*;

pub const VAR_TYPED_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "var-typed-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "var with a typed-integer normalization or dimension control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:VarTypedIntegerControlExtension"),
    };

pub const VAR_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [VAR_TYPED_INTEGER_CONTROL_EXTENSION];

const VAR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "V",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Variance reduction result.",
}];

const VAR_PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single- or double-precision input array.",
};

const VAR_PARAM_W: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "w",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("0"),
    description: "Normalization flag (0 sample, 1 population) or [] placeholder.",
};

const VAR_PARAM_AXES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "axes",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("[]"),
    description: "Dimension selector, vector of dimensions, or \"all\".",
};

const VAR_PARAM_NANFLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nanflag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"includemissing\""),
    description:
        "Missing-value mode: \"includemissing\"/\"includenan\" or \"omitmissing\"/\"omitnan\".",
};

const VAR_INPUTS_A: [BuiltinParamDescriptor; 1] = [VAR_PARAM_A];
const VAR_INPUTS_A_W: [BuiltinParamDescriptor; 2] = [VAR_PARAM_A, VAR_PARAM_W];
const VAR_INPUTS_A_W_AXES: [BuiltinParamDescriptor; 3] = [VAR_PARAM_A, VAR_PARAM_W, VAR_PARAM_AXES];
const VAR_INPUTS_A_NANFLAG: [BuiltinParamDescriptor; 2] = [VAR_PARAM_A, VAR_PARAM_NANFLAG];
const VAR_INPUTS_A_W_NANFLAG: [BuiltinParamDescriptor; 3] =
    [VAR_PARAM_A, VAR_PARAM_W, VAR_PARAM_NANFLAG];
const VAR_INPUTS_A_W_AXES_NANFLAG: [BuiltinParamDescriptor; 4] =
    [VAR_PARAM_A, VAR_PARAM_W, VAR_PARAM_AXES, VAR_PARAM_NANFLAG];
const VAR_INPUTS_A_W_NANFLAG_AXES: [BuiltinParamDescriptor; 4] =
    [VAR_PARAM_A, VAR_PARAM_W, VAR_PARAM_NANFLAG, VAR_PARAM_AXES];

const VAR_SIGNATURES: [BuiltinSignatureDescriptor; 11] = [
    BuiltinSignatureDescriptor {
        label: "V = var(A)",
        inputs: &VAR_INPUTS_A,
        outputs: &VAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "V = var(A, w)",
        inputs: &VAR_INPUTS_A_W,
        outputs: &VAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "V = var(A, w, dim)",
        inputs: &VAR_INPUTS_A_W_AXES,
        outputs: &VAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "V = var(A, w, vecdim)",
        inputs: &VAR_INPUTS_A_W_AXES,
        outputs: &VAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "V = var(A, w, \"all\")",
        inputs: &VAR_INPUTS_A_W_AXES,
        outputs: &VAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "V = var(A, nanflag)",
        inputs: &VAR_INPUTS_A_NANFLAG,
        outputs: &VAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "V = var(A, w, nanflag)",
        inputs: &VAR_INPUTS_A_W_NANFLAG,
        outputs: &VAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "V = var(A, w, dim, nanflag)",
        inputs: &VAR_INPUTS_A_W_AXES_NANFLAG,
        outputs: &VAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "V = var(A, w, nanflag, dim)",
        inputs: &VAR_INPUTS_A_W_NANFLAG_AXES,
        outputs: &VAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "V = var(A, w, [], nanflag)",
        inputs: &VAR_INPUTS_A_W_AXES_NANFLAG,
        outputs: &VAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "V = var(A, w, nanflag, \"all\")",
        inputs: &VAR_INPUTS_A_W_NANFLAG_AXES,
        outputs: &VAR_OUTPUT,
    },
];

pub const VAR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VAR.INVALID_ARGUMENT",
    identifier: Some("RunMat:var:InvalidArgument"),
    when: "Argument grammar, dimensions, normalization, or missing-value selectors are invalid.",
    message: "var: invalid argument",
};

pub const VAR_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VAR.INVALID_INPUT",
    identifier: Some("RunMat:var:InvalidInput"),
    when: "Input values are outside the supported floating var reduction domains, including integer data.",
    message: "var: invalid input",
};

pub const VAR_ERROR_COMPLEX_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VAR.COMPLEX_UNSUPPORTED",
    identifier: Some("RunMat:var:ComplexUnsupported"),
    when: "Input value is complex for a path that currently supports only real reductions.",
    message: "var: complex inputs are not supported yet",
};

pub const VAR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VAR.INTERNAL",
    identifier: Some("RunMat:var:Internal"),
    when: "Execution fails due to provider, conversion, gather, or tensor allocation internals.",
    message: "var: internal reduction failure",
};

const VAR_ERRORS: [BuiltinErrorDescriptor; 4] = [
    VAR_ERROR_INVALID_ARGUMENT,
    VAR_ERROR_INVALID_INPUT,
    VAR_ERROR_COMPLEX_UNSUPPORTED,
    VAR_ERROR_INTERNAL,
];

pub const VAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &VAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &VAR_ERRORS,
};

const REJECTED_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented data domain is single, double, table, or timetable; every typed-integer data class is rejected before host or provider dispatch.",
    }];

const INTEGER_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "normalization_or_dimension",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode accepts exact host typed-integer normalization and dimension controls; MATLAB-compatible modes retain the documented floating control surface.",
    }];

pub const VAR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "V = var(A, args...) with integer A",
        inputs: &REJECTED_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer data is invalid in every compatibility mode, including resident integer data, and no integer output contract exists.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "V = var(A, normalization_or_dimension, ...) with floating A",
        inputs: &INTEGER_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed-integer controls are a mode-gated RunMat extension; values are decoded exactly on the host, while the floating data reduction may still execute on a provider. The result class follows the floating data and execution path, not the integer control.",
    },
];
