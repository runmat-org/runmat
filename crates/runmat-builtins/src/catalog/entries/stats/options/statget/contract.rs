use crate::*;

const OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "val",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Selected option or supplied default.",
}];
const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Statistics options structure.",
};
const FIELD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "field",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Canonical name or unique case-insensitive prefix.",
};
const DEFAULT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "defaultData",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Returned unchanged when a recognized field is empty.",
};
pub const STATGET_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STATGET.INVALID_ARGUMENT",
    identifier: Some("RunMat:statget:InvalidArgument"),
    when: "The argument grammar or required resident materialization fails.",
    message: "statget: invalid argument",
};
pub const STATGET_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STATGET.INVALID_OPTION",
    identifier: Some("RunMat:statget:InvalidOption"),
    when: "The options argument is not a struct or field name is not text.",
    message: "statget: invalid option",
};

pub const STATGET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "val = statget(options, field)",
            inputs: &[OPTIONS, FIELD],
            outputs: &OUT,
        },
        BuiltinSignatureDescriptor {
            label: "val = statget(options, field, defaultData)",
            inputs: &[OPTIONS, FIELD, DEFAULT],
            outputs: &OUT,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[STATGET_ERROR_INVALID_ARGUMENT, STATGET_ERROR_INVALID_OPTION],
};
pub const STATGET_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "val = statget(options, field, integer_defaultData)",
    inputs: &[BuiltinIntegerInputCapability { name: "defaultData", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Opaque default data retains exact native class, shape, and payload if the selected field is empty." }],
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::PreserveInput,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "The default is not numerically converted; an automatically resident selected default may gather through its owning provider.",
}];
