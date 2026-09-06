use crate::*;

const NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Environment name or equally shaped container of names.",
};
const VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Text, scalar numeric value, or equally shaped container of values.",
};
const DICTIONARY: BuiltinParamDescriptor = BuiltinParamDescriptor { name: "environment", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "Dictionary whose keys are environment names and whose values are admitted environment values." };
const STATUS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat extension: zero on success and one on validation or host rejection.",
};
const MESSAGE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "message",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat extension: diagnostic text, or an empty character row on success.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "setenv(name)",
        inputs: &[NAME],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "setenv(name, value)",
        inputs: &[NAME, VALUE],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "setenv(environment)",
        inputs: &[DICTIONARY],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "status = setenv(name, value)",
        inputs: &[NAME, VALUE],
        outputs: &[STATUS],
    },
    BuiltinSignatureDescriptor {
        label: "[status, message] = setenv(name, value)",
        inputs: &[NAME, VALUE],
        outputs: &[STATUS, MESSAGE],
    },
];
pub const SETENV_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETENV.ARITY",
    identifier: Some("RunMat:setenv:Arity"),
    when: "The call has no inputs or more than two inputs.",
    message: "setenv: expected one or two input arguments",
};
pub const SETENV_ERROR_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETENV.INVALID_NAME",
    identifier: Some("RunMat:setenv:InvalidName"),
    when: "An environment name is not valid text.",
    message:
        "setenv: names must be character vectors, string values, or cells of character vectors",
};
pub const SETENV_ERROR_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETENV.INVALID_VALUE",
    identifier: Some("RunMat:setenv:InvalidValue"),
    when: "An environment value has an unsupported class or shape.",
    message: "setenv: values must be text, scalar numeric values, or missing strings",
};
pub const SETENV_ERROR_SHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETENV.SHAPE",
    identifier: Some("RunMat:setenv:SizeMismatch"),
    when: "Nonscalar name and value containers do not have the same size.",
    message: "setenv: names and values must have the same size unless one side is scalar",
};
pub const SETENV_ERROR_DICTIONARY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETENV.DICTIONARY",
    identifier: Some("RunMat:setenv:InvalidDictionary"),
    when: "The dictionary storage or a key/value entry is invalid.",
    message: "setenv: environment dictionary is invalid",
};
pub const SETENV_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETENV.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:setenv:TooManyOutputs"),
    when: "More than two outputs are requested.",
    message: "setenv: too many output arguments",
};
pub const SETENV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        SETENV_ERROR_ARITY,
        SETENV_ERROR_NAME,
        SETENV_ERROR_VALUE,
        SETENV_ERROR_SHAPE,
        SETENV_ERROR_DICTIONARY,
        SETENV_ERROR_TOO_MANY_OUTPUTS,
    ],
};
pub const SETENV_STATUS_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "setenv-status-outputs",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "setenv status and message outputs are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SetenvStatusOutputsExtension"),
};
pub(super) const EXTENSIONS: &[BuiltinExtensionDescriptor] = &[SETENV_STATUS_OUTPUT_EXTENSION];
const INTEGER_INPUTS: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "value",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "A scalar integer value is converted directly from exact storage to decimal text.",
}];
pub(super) const INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor {
        form: "setenv(name, integer_value)",
        inputs: INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer values remain exact through decimal conversion; integer names reject.",
    }];
