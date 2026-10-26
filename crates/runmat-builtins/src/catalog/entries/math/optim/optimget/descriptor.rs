use crate::*;
pub const OPTIMGET_INTEGER_CAPABILITIES:&[BuiltinIntegerCapabilityDescriptor]=&[BuiltinIntegerCapabilityDescriptor {
    form:"value = optimget(options,parameter,default)",
    inputs:&[BuiltinIntegerInputCapability{name:"stored value or default payload",classes:&ALL_INTEGER_CLASSES,
        availability:BuiltinIntegerInputAvailability::Documented,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:"The chosen value retains native integer storage; no numeric normalization occurs."}],
    computation_domain:BuiltinIntegerComputationDomain::Structural,output_class:BuiltinIntegerOutputClassRule::PreserveInput,
    overflow:BuiltinIntegerOverflowRule::NotApplicable,backend:BuiltinIntegerBackendRule::HostAndGpu,
    overload:BuiltinIntegerOverloadKind::Multiple,notes:"Opaque option-value selection neither executes a numeric kernel nor gathers resident data.",
}];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Unchanged stored value, supplied default or [].",
}];
const CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar optimization options struct or numeric [].",
    },
    BuiltinParamDescriptor {
        name: "parameter",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Case-insensitive option name or unambiguous leading abbreviation.",
    },
];
const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar optimization options struct or numeric [].",
    },
    BuiltinParamDescriptor {
        name: "parameter",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Case-insensitive option name or unambiguous leading abbreviation.",
    },
    BuiltinParamDescriptor {
        name: "default",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Value returned when the parameter is absent or numeric [].",
    },
];
pub const OPTIMGET_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMGET.INVALID_ARGUMENT",
    identifier: Some("RunMat:optimget:InvalidArgument"),
    when: "Invalid argument/output count, options container or unknown/ambiguous parameter name.",
    message: "optimget: invalid argument",
};
pub const OPTIMGET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "value = optimget(options,parameter)",
            inputs: &CORE,
            outputs: &OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "value = optimget(options,parameter,default)",
            inputs: &INPUTS,
            outputs: &OUTPUT,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[OPTIMGET_ERROR_INVALID_ARGUMENT],
};
