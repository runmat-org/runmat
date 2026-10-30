use crate::*;

const SIZE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sizes",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "RunMat-only nonnegative size values or one size vector.",
}];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String array containing typed missing elements.",
}];
pub const MISSING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "S = missing",
            inputs: &[],
            outputs: &OUTPUTS,
        },
        BuiltinSignatureDescriptor {
            label: "S = missing(sizes)",
            inputs: &SIZE_INPUTS,
            outputs: &OUTPUTS,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        BuiltinErrorDescriptor {
            code: "RM.MISSING.INVALID_ARGUMENT",
            identifier: Some("RunMat:missing:InvalidArgument"),
            when: "Size controls are malformed, negative, fractional, or exceed allocation limits.",
            message: "missing-value builtin: invalid argument",
        },
        BuiltinErrorDescriptor {
            code: "RM.MISSING.INTERNAL",
            identifier: Some("RunMat:missing:InternalError"),
            when: "String-array construction or provider result admission fails.",
            message: "missing-value builtin: internal error",
        },
    ],
};

pub const MISSING_SHAPED_ARRAY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "missing-shaped-array",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "missing(size...) is a RunMat convenience; the documented MATLAB missing function accepts no input arguments",
        error_identifier: Some("RunMat:compatibility:MissingShapedArrayExtension"),
    };
pub const MISSING_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [MISSING_SHAPED_ARRAY_EXTENSION];

const INTEGER_SIZE: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "size arguments",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Native integer sizes are decoded exactly and checked against nonnegative platform allocation limits.",
}];
pub const MISSING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "missing(integer_size, ...)",
        inputs: &INTEGER_SIZE,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "MATLAB-compatible mode rejects all arguments before provider access. RunMat mode reads admitted size controls exactly and creates a host string array.",
    }];
