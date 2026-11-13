use crate::*;

const BUILTIN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Output value(s) from the selected builtin implementation.",
}];

const BUILTIN_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Builtin function name.",
    },
    BuiltinParamDescriptor {
        name: "varargin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Builtin function arguments.",
    },
];

const BUILTIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[varargout] = builtin(name, varargin)",
    inputs: &BUILTIN_INPUTS,
    outputs: &BUILTIN_OUTPUT,
}];

const BUILTIN_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "varargin",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are passed unchanged when they are valid inputs for the builtin selected by name; mixed-class and scalar-double rules belong to that target.",
    }];

pub const BUILTIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[varargout] = builtin(name,varargin) with integer target arguments or results",
        inputs: &BUILTIN_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "builtin performs no numeric conversion or arithmetic of its own; input admission, output count and class, overflow, shape, and host or provider behavior are inherited from the selected registered builtin.",
    }];

pub const BUILTIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BUILTIN_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
