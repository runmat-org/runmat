use crate::*;

pub const DB_NONFLOATING_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "db-nonfloating-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "db with integer, logical, or complex-integer computation input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DbNonfloatingInputExtension"),
};
pub const DB_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [DB_NONFLOATING_INPUT_EXTENSION];

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "yDb",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real single or double decibel output, following the input or broadcast shape.",
}];
const INPUT_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input signal magnitude or power quantity.",
};
const INPUT_MODE_OR_R: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "modeOrR",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("\"voltage\""),
    description: "Mode string ('voltage' or 'power') or positive real resistance reference.",
};
const INPUTS_Y: [BuiltinParamDescriptor; 1] = [INPUT_Y];
const INPUTS_Y_MODE: [BuiltinParamDescriptor; 2] = [INPUT_Y, INPUT_MODE_OR_R];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "yDb = db(y)",
        inputs: &INPUTS_Y,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "yDb = db(y, \"voltage\")",
        inputs: &INPUTS_Y_MODE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "yDb = db(y, \"power\")",
        inputs: &INPUTS_Y_MODE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "yDb = db(y, R)",
        inputs: &INPUTS_Y_MODE,
        outputs: &OUTPUT,
    },
];

pub const DB_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DB.INVALID_ARGUMENT",
    identifier: Some("RunMat:db:InvalidArgument"),
    when: "Inputs do not match supported db invocation forms.",
    message: "db: invalid argument",
};
pub const DB_ERROR_INVALID_MODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DB.INVALID_MODE",
    identifier: Some("RunMat:db:InvalidMode"),
    when: "Mode string is not recognized or is not a scalar text value.",
    message: "db: invalid mode",
};
pub const DB_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DB.INVALID_INPUT",
    identifier: Some("RunMat:db:InvalidInput"),
    when: "Input signal cannot be interpreted as numeric magnitude data.",
    message: "db: invalid input",
};
pub const DB_ERROR_INVALID_RESISTANCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DB.INVALID_RESISTANCE",
    identifier: Some("RunMat:db:InvalidResistance"),
    when: "Resistance reference is non-numeric, complex, non-finite, or non-positive.",
    message: "db: invalid resistance",
};
pub const DB_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DB.SIZE_MISMATCH",
    identifier: Some("RunMat:db:SizeMismatch"),
    when: "Signal and resistance inputs are not broadcast compatible.",
    message: "db: array sizes are not compatible for broadcasting",
};
pub const DB_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DB.INTERNAL",
    identifier: Some("RunMat:db:Internal"),
    when: "Internal tensor conversion or allocation failed.",
    message: "db: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 6] = [
    DB_ERROR_INVALID_ARGUMENT,
    DB_ERROR_INVALID_MODE,
    DB_ERROR_INVALID_INPUT,
    DB_ERROR_INVALID_RESISTANCE,
    DB_ERROR_SIZE_MISMATCH,
    DB_ERROR_INTERNAL,
];
pub const DB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "y", classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Real and paired-complex integer computation input is gated and enters the decibel floating domain.",
    },
    BuiltinIntegerInputCapability {
        name: "R", classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer positive resistance is gated and broadcasts with the signal magnitude.",
    },
];
pub const DB_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "yDb = db(y, mode_or_R)", inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Nonfloating computation input is RunMat-only. Integer/logical input promotes to double except that a native-single signal or resistance selects single output; admitted resident inputs gather for host conversion.",
    }];
