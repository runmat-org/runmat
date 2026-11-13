use crate::*;
const COPYOBJ_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "hnew",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Copied graphics object handle or handle array.",
}];

const COPYOBJ_INPUTS_SOURCE_PARENT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Graphics object handle or handle array to copy.",
    },
    BuiltinParamDescriptor {
        name: "parent",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Destination axes handle compatible with the plot child.",
    },
];

const COPYOBJ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "hnew = copyobj(h, parent)",
    inputs: &COPYOBJ_INPUTS_SOURCE_PARENT,
    outputs: &COPYOBJ_OUTPUT,
}];

pub const COPYOBJ_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COPYOBJ.INVALID_ARGUMENT",
    identifier: Some("RunMat:copyobj:InvalidArgument"),
    when: "Argument count, source handle, or target parent handle is invalid or unsupported.",
    message: "copyobj: invalid argument",
};

pub const COPYOBJ_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COPYOBJ.INTERNAL",
    identifier: Some("RunMat:copyobj:Internal"),
    when: "Internal graphics state lookup or copy registration fails.",
    message: "copyobj: internal operation failed",
};

const COPYOBJ_ERRORS: [BuiltinErrorDescriptor; 3] = [
    COPYOBJ_ERROR_INVALID_ARGUMENT,
    COPYOBJ_ERROR_INTERNAL,
    COPYOBJ_ERROR_TOO_MANY_OUTPUTS,
];

pub const COPYOBJ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COPYOBJ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COPYOBJ_ERRORS,
};

pub const COPYOBJ_INTEGER_HANDLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "copyobj-integer-handle-aliases",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "copyobj with typed-integer numeric graphics-handle aliases is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CopyobjIntegerHandleExtension"),
    };

pub const COPYOBJ_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [COPYOBJ_INTEGER_HANDLE_EXTENSION];

const COPYOBJ_INTEGER_SOURCE: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "h",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode treats exactly representable nonnegative host integers as aliases for its numeric graphics-handle representation; the public contract accepts graphics objects, not integer arrays.",
    }];
const COPYOBJ_INTEGER_PARENT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "parent",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode treats exactly representable nonnegative host integers as aliases for numeric axes handles; resident numeric arrays are not graphics objects.",
    }];
const COPYOBJ_RESIDENT_INTEGER: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident h or parent",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Resident integer arrays are numeric data rather than graphics objects and reject without provider gather.",
    }];

pub const COPYOBJ_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "hnew = copyobj(integer_h, parent)", inputs: &COPYOBJ_INTEGER_SOURCE, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::SameSizeOrScalar, notes: "A separately gated RunMat alias reads authoritative host integer storage, rejects values that cannot be represented exactly by RunMat's f64 graphics registry, validates every source before mutation, and returns fresh host graphics handles." },
    BuiltinIntegerCapabilityDescriptor { form: "hnew = copyobj(h, integer_parent)", inputs: &COPYOBJ_INTEGER_PARENT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::SameSizeOrScalar, notes: "A separately gated RunMat alias resolves authoritative host integers as axes parents before any copy is made; output handles follow the controlling source/parent shape." },
    BuiltinIntegerCapabilityDescriptor { form: "copyobj(resident_integer_handle_alias, ...)", inputs: &COPYOBJ_RESIDENT_INTEGER, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::Multiple, notes: "GPU-resident numeric arrays are not graphics-object handles; copyobj performs no provider dispatch or implicit gather." },
];

pub const COPYOBJ_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COPYOBJ.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:copyobj:TooManyOutputs"),
    when: "More than 1 output arguments are requested.",
    message: "copyobj: too many output arguments",
};
