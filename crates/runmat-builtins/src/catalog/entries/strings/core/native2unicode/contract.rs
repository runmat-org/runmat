use crate::*;

const OUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output value.",
}];

const NATIVE2UNICODE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "bytes",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric byte vector with values in the range 0 through 255, or a character vector or string scalar returned unchanged.",
    },
    BuiltinParamDescriptor {
        name: "encoding",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"UTF-8\""),
        description: "Source character encoding.",
    },
];
pub const NATIVE2UNICODE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NATIVE2UNICODE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:native2unicode:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "native2unicode: too many output arguments",
};

pub const NATIVE2UNICODE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "text = native2unicode(bytes, encoding)",
        inputs: &NATIVE2UNICODE_INPUTS,
        outputs: &OUT_ANY,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[NATIVE2UNICODE_ERROR_TOO_MANY_OUTPUTS],
};

const NATIVE2UNICODE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "bytes",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented for byte vectors; every element must be exactly within 0 through 255.",
    }];
pub const NATIVE2UNICODE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "text = native2unicode(integer_bytes[, encoding])",
        inputs: &NATIVE2UNICODE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Native byte extraction is exact and range checked before decoding to a host character vector; explicit gpuArray input is unsupported and automatic residency gathers transparently.",
    }];
