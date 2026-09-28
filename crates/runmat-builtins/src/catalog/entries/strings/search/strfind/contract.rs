use crate::*;

const STRFIND_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "idx",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Match start indices as a numeric row vector or cell array of row vectors.",
}];

const STRFIND_INPUTS_BASE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text input (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "pat",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pattern text (string/char/cell/string array).",
    },
];

const STRFIND_INPUTS_FORCE_CELL_PAIR: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text input (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "pat",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pattern text (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"ForceCellOutput\""),
        description: "Option name (`\"ForceCellOutput\"`).",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical-ish option value controlling forced cell output.",
    },
];

const STRFIND_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "idx = strfind(str, pat)",
        inputs: &STRFIND_INPUTS_BASE,
        outputs: &STRFIND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "idx = strfind(str, pat, \"ForceCellOutput\", value)",
        inputs: &STRFIND_INPUTS_FORCE_CELL_PAIR,
        outputs: &STRFIND_OUTPUT,
    },
];

pub const STRFIND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRFIND.INVALID_INPUT",
    identifier: Some("RunMat:strfind:InvalidInput"),
    when: "Text or pattern input is not a supported text container.",
    message: "strfind: text and pattern inputs must be text values",
};

pub const STRFIND_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRFIND.INVALID_OPTION",
    identifier: Some("RunMat:strfind:InvalidOption"),
    when: "ForceCellOutput option arguments are invalid or malformed.",
    message: "strfind: invalid option arguments",
};

pub const STRFIND_ERROR_SHAPE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRFIND.SHAPE_MISMATCH",
    identifier: Some("RunMat:strfind:ShapeMismatch"),
    when: "Text and pattern inputs are not broadcast-compatible.",
    message: "strfind: input sizes are not broadcast-compatible",
};

pub const STRFIND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRFIND.INTERNAL",
    identifier: Some("RunMat:strfind:InternalError"),
    when: "Internal output assembly failed.",
    message: "strfind: internal error",
};

const STRFIND_ERRORS: [BuiltinErrorDescriptor; 4] = [
    STRFIND_ERROR_INVALID_INPUT,
    STRFIND_ERROR_INVALID_OPTION,
    STRFIND_ERROR_SHAPE_MISMATCH,
    STRFIND_ERROR_INTERNAL,
];

pub const STRFIND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRFIND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STRFIND_ERRORS,
};

pub const STRFIND_TYPED_FORCE_CELL_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "strfind-typed-force-cell-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "strfind with a typed-integer ForceCellOutput value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StrfindTypedForceCellOutputExtension"),
    };

pub const STRFIND_NONBINARY_FORCE_CELL_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "strfind-nonbinary-force-cell-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "strfind with a numeric ForceCellOutput value other than zero or one is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StrfindNonbinaryForceCellOutputExtension"),
    };

pub const STRFIND_RESIDENT_FORCE_CELL_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "strfind-resident-force-cell-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "strfind with a resident ForceCellOutput value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StrfindResidentForceCellOutputExtension"),
    };

pub const STRFIND_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    STRFIND_TYPED_FORCE_CELL_OUTPUT_EXTENSION,
    STRFIND_NONBINARY_FORCE_CELL_OUTPUT_EXTENSION,
    STRFIND_RESIDENT_FORCE_CELL_OUTPUT_EXTENSION,
];

const STRFIND_INTEGER_FORCE_CELL_OUTPUT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "cellOutput",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target explicitly documents false, true, 0, and 1 but does not enumerate typed integer storage classes. RunMat mode accepts every exact scalar integer class; MATLAB-compatible mode retains only logical and double zero/one.",
    }];

pub const STRFIND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "k = strfind(str, pat, 'ForceCellOutput', integer_cellOutput)",
        inputs: &STRFIND_INTEGER_FORCE_CELL_OUTPUT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The typed integer flag controls only whether double index vectors are wrapped in cells. Typed storage acceptance is mode-gated because the public page lists numeric literals but no storage classes; values other than zero and one remain a separately gated convenience.",
    }];
