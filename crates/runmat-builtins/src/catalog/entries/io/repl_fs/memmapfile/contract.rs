use crate::*;

pub const MEMMAPFILE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("memmapfile");

const FILENAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File to read through the selected filesystem provider.",
};
const OPTION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Writable, Offset, Format, or Repeat; every name requires one value.",
};
const VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Typed property value; Format can be a scalar type or an n-by-3 record table.",
};
const RESULT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "m",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Materialized memmapfile object with Filename, Writable, Offset, Format, Repeat, and Data.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "m = memmapfile(filename)",
        inputs: &[FILENAME],
        outputs: &[RESULT],
    },
    BuiltinSignatureDescriptor {
        label: "m = memmapfile(filename, Name, Value, ...)",
        inputs: &[FILENAME, OPTION, VALUE],
        outputs: &[RESULT],
    },
];
pub const MEMMAPFILE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[MEMMAPFILE_ERROR_TOO_MANY_OUTPUTS],
};
pub const MEMMAPFILE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMMAPFILE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:memmapfile:TooManyOutputs"),
    when: "More than one result is requested.",
    message: "memmapfile: too many output arguments",
};

pub const MEMMAPFILE_INTEGER_CONTROLS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "memmapfile-integer-property-controls",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Typed-integer Writable, Offset, Repeat, and Format-shape controls are a RunMat extension.",
        error_identifier: Some("RunMat:compatibility:MemmapfileIntegerControlsExtension"),
    };
pub const MEMMAPFILE_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "memmapfile-explicit-gpu-argument",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "An explicit gpuArray structural argument to host-side memmapfile is a RunMat extension.",
        error_identifier: Some("RunMat:compatibility:MemmapfileExplicitGpuExtension"),
    };
pub const MEMMAPFILE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    MEMMAPFILE_INTEGER_CONTROLS_EXTENSION,
    MEMMAPFILE_EXPLICIT_GPU_EXTENSION,
];

const SHAPE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Format.shape",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Typed integer dimensions are parsed exactly and checked before byte geometry.",
}];
const CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Writable, Offset, or Repeat",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Typed controls are parsed exactly; compatibility mode retains logical/double controls.",
}];
pub const MEMMAPFILE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "m.Data = memmapfile(..., 'Format', integer_typename)",
        inputs: &[],
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "All eight integer Format classes decode directly into native integer storage.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "m = memmapfile(..., 'Format', {type, integer_shape, field})",
        inputs: &SHAPE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes:
            "Record dimensions are exact platform integers; payload class follows the type name.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "m = memmapfile(..., integer_property_control)",
        inputs: &CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes:
            "Integer controls are exact; automatic resident controls gather from their exact owner.",
    },
];
