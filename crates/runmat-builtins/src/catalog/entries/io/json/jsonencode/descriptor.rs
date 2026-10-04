use crate::*;

pub const JSONENCODE_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "jsonencode-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "jsonencode with explicit gpuArray input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:JsonencodeResidentInputExtension"),
    };
pub const JSONENCODE_TYPED_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "jsonencode-typed-integer-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "jsonencode with a typed-integer boolean option is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:JsonencodeTypedIntegerOptionExtension"),
    };
pub const JSONENCODE_NUMERIC_OPTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "jsonencode-numeric-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "jsonencode with a floating numeric boolean option is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:JsonencodeNumericOptionExtension"),
    };
pub const JSONENCODE_TEXT_OPTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "jsonencode-text-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "jsonencode with a textual boolean option is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:JsonencodeTextOptionExtension"),
    };
pub const JSONENCODE_COMPLEX_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "jsonencode-complex-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "jsonencode complex-number object encoding is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:JsonencodeComplexInputExtension"),
    };
pub const JSONENCODE_SPARSE_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "jsonencode-sparse-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "jsonencode sparse-array densification is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:JsonencodeSparseInputExtension"),
    };
pub const JSONENCODE_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    JSONENCODE_RESIDENT_INPUT_EXTENSION,
    JSONENCODE_TYPED_INTEGER_OPTION_EXTENSION,
    JSONENCODE_NUMERIC_OPTION_EXTENSION,
    JSONENCODE_TEXT_OPTION_EXTENSION,
    JSONENCODE_COMPLEX_INPUT_EXTENSION,
    JSONENCODE_SPARSE_INPUT_EXTENSION,
];

const JSONENCODE_REAL_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "value",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight real integer classes serialize from authoritative storage to exact signed or unsigned JSON decimal literals.",
    }];
const JSONENCODE_INTEGER_OPTION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "PrettyPrint_or_ConvertInfAndNaN",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "MATLAB-compatible mode requires logical option values; RunMat mode treats exact integer zero as false and every nonzero integer as true.",
    }];
const JSONENCODE_COMPLEX_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "value",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public jsonencode contract rejects complex numeric data. RunMat mode retains its exact {real,imag} object encoding for paired integer storage.",
    }];
pub const JSONENCODE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "jsonText = jsonencode(full_real_integer_value)",
        inputs: &JSONENCODE_REAL_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host and automatically resident integer values serialize exactly. Explicit gpuArray input is independently compatibility-gated before owner access; accepted resident values gather through their owner at the serialization sink.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "jsonText = jsonencode(value, option, integer_flag)",
        inputs: &JSONENCODE_INTEGER_OPTION_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Typed numeric option coercion is a RunMat convenience and is gated before resident access.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "jsonText = jsonencode(complex_integer_value)",
        inputs: &JSONENCODE_COMPLEX_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Complex JSON encoding is outside the public compatibility surface and requires the complex-input extension; integer components remain exact in RunMat mode.",
    },
];

const JSONENCODE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "jsonText",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "JSON text encoded as a character row vector.",
}];
const JSONENCODE_INPUTS_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value to encode as JSON.",
}];
const JSONENCODE_INPUTS_VALUE_OPTIONS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value to encode as JSON.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Options struct with fields such as PrettyPrint and ConvertInfAndNaN.",
    },
];
const JSONENCODE_INPUTS_VALUE_NAME_VALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value to encode as JSON.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option name (for example \"PrettyPrint\" or \"ConvertInfAndNaN\").",
    },
    BuiltinParamDescriptor {
        name: "optionValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option value for the preceding option name.",
    },
];
const JSONENCODE_INPUTS_VALUE_NAME_VALUE_VARIADIC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value to encode as JSON.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs...",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option pairs.",
    },
];
const JSONENCODE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "jsonText = jsonencode(value)",
        inputs: &JSONENCODE_INPUTS_VALUE,
        outputs: &JSONENCODE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "jsonText = jsonencode(value, options)",
        inputs: &JSONENCODE_INPUTS_VALUE_OPTIONS,
        outputs: &JSONENCODE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "jsonText = jsonencode(value, name, optionValue)",
        inputs: &JSONENCODE_INPUTS_VALUE_NAME_VALUE,
        outputs: &JSONENCODE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "jsonText = jsonencode(value, nameValuePairs...)",
        inputs: &JSONENCODE_INPUTS_VALUE_NAME_VALUE_VARIADIC,
        outputs: &JSONENCODE_OUTPUT,
    },
];
pub const JSONENCODE_ERROR_OPTIONS_CONFIG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JSONENCODE.OPTIONS_CONFIG",
    identifier: None,
    when: "Single options argument is provided but is not a struct.",
    message: "jsonencode: expected name/value pairs or options struct",
};
pub const JSONENCODE_ERROR_NAME_VALUE_PAIRS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JSONENCODE.NAME_VALUE_PAIRS",
    identifier: None,
    when: "Name-value options do not come in pairs.",
    message: "jsonencode: name/value pairs must come in pairs",
};
pub const JSONENCODE_ERROR_OPTION_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JSONENCODE.OPTION_NAME",
    identifier: None,
    when: "Option name is not a character vector or string scalar.",
    message: "jsonencode: option names must be character vectors or strings",
};
pub const JSONENCODE_ERROR_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JSONENCODE.OPTION_VALUE",
    identifier: None,
    when: "Option value is not a scalar logical/numeric or boolean-like text.",
    message: "jsonencode: option value must be scalar logical or numeric",
};
pub const JSONENCODE_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JSONENCODE.UNKNOWN_OPTION",
    identifier: None,
    when: "Option name is not recognized.",
    message: "jsonencode: unknown option name",
};
pub const JSONENCODE_ERROR_INF_NAN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JSONENCODE.INF_NAN",
    identifier: None,
    when: "Input contains NaN/Inf while ConvertInfAndNaN is false.",
    message: "jsonencode: ConvertInfAndNaN must be true to encode NaN or Inf values",
};
pub const JSONENCODE_ERROR_UNSUPPORTED_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JSONENCODE.UNSUPPORTED_TYPE",
    identifier: None,
    when: "Input value type is not supported for JSON encoding.",
    message:
        "jsonencode: unsupported input type; expected numeric, logical, string, struct, cell, or object data",
};
pub const JSONENCODE_ERROR_UNEXPECTED_GPU: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JSONENCODE.UNEXPECTED_GPU",
    identifier: None,
    when: "A GPU tensor handle reaches encoding after gather pass.",
    message: "jsonencode: unexpected gpuArray handle after gather pass",
};
pub const JSONENCODE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JSONENCODE.INTERNAL",
    identifier: None,
    when: "Internal JSON conversion or container materialization fails.",
    message: "jsonencode: internal conversion failed",
};
const JSONENCODE_ERRORS: [BuiltinErrorDescriptor; 9] = [
    JSONENCODE_ERROR_OPTIONS_CONFIG,
    JSONENCODE_ERROR_NAME_VALUE_PAIRS,
    JSONENCODE_ERROR_OPTION_NAME,
    JSONENCODE_ERROR_OPTION_VALUE,
    JSONENCODE_ERROR_UNKNOWN_OPTION,
    JSONENCODE_ERROR_INF_NAN,
    JSONENCODE_ERROR_UNSUPPORTED_TYPE,
    JSONENCODE_ERROR_UNEXPECTED_GPU,
    JSONENCODE_ERROR_INTERNAL,
];
pub const JSONENCODE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &JSONENCODE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &JSONENCODE_ERRORS,
};
