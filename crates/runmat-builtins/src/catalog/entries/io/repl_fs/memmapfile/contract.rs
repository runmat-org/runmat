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
    errors: &[
        MEMMAPFILE_ERROR_TOO_MANY_OUTPUTS,
        MEMMAPFILE_ERROR_INVALID_CALL,
        MEMMAPFILE_ERROR_INVALID_FILENAME,
        MEMMAPFILE_ERROR_INVALID_OPTION,
        MEMMAPFILE_ERROR_INVALID_CONTROL,
        MEMMAPFILE_ERROR_INVALID_FORMAT,
        MEMMAPFILE_ERROR_READ_FAILED,
        MEMMAPFILE_ERROR_WRITE_FAILED,
        MEMMAPFILE_ERROR_PROVIDER_CONTROL,
        MEMMAPFILE_ERROR_CANCELLED,
        MEMMAPFILE_ERROR_INTEGER_COMPATIBILITY,
        MEMMAPFILE_ERROR_GPU_COMPATIBILITY,
    ],
};
pub const MEMMAPFILE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMMAPFILE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:memmapfile:TooManyOutputs"),
    when: "More than one result is requested.",
    message: "memmapfile: too many output arguments",
};

pub const MEMMAPFILE_ERROR_INVALID_CALL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMMAPFILE.INVALID_CALL",
    identifier: Some("RunMat:memmapfile:InvalidCall"),
    when: "The filename is missing or Name/Value pairs are incomplete.",
    message: "memmapfile: filename and complete Name/Value pairs are required",
};
pub const MEMMAPFILE_ERROR_INVALID_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMMAPFILE.INVALID_FILENAME",
    identifier: Some("RunMat:memmapfile:InvalidFilename"),
    when: "The filename is not a host text scalar or cannot be resolved.",
    message: "memmapfile: filename must resolve to a filesystem path",
};
pub const MEMMAPFILE_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMMAPFILE.INVALID_OPTION",
    identifier: Some("RunMat:memmapfile:InvalidOption"),
    when: "A property name is not a supported host text scalar.",
    message: "memmapfile: unsupported property name",
};
pub const MEMMAPFILE_ERROR_INVALID_CONTROL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMMAPFILE.INVALID_CONTROL",
    identifier: Some("RunMat:memmapfile:InvalidControl"),
    when: "Offset or Repeat has an invalid value or checked geometry fails.",
    message: "memmapfile: invalid structural control",
};
pub const MEMMAPFILE_ERROR_INVALID_FORMAT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMMAPFILE.INVALID_FORMAT",
    identifier: Some("RunMat:memmapfile:InvalidFormat"),
    when: "Format type, table, field, or decoded record geometry is invalid.",
    message: "memmapfile: invalid Format",
};
pub const MEMMAPFILE_ERROR_READ_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMMAPFILE.READ_FAILED",
    identifier: Some("RunMat:memmapfile:ReadFailed"),
    when: "The selected filesystem provider cannot read the captured file.",
    message: "memmapfile: file read failed",
};
pub const MEMMAPFILE_ERROR_WRITE_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMMAPFILE.WRITE_FAILED",
    identifier: Some("RunMat:memmapfile:WriteFailed"),
    when: "A mapped write fails before, during, or after bytes reach the selected backing; the typed source states whether bytes committed.",
    message: "memmapfile: mapped write failed",
};
pub const MEMMAPFILE_ERROR_PROVIDER_CONTROL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMMAPFILE.PROVIDER_CONTROL",
    identifier: None,
    when: "A provider-resident structural control fails to download; its original identifier and source are retained.",
    message: "memmapfile: provider control materialization failed",
};
pub const MEMMAPFILE_ERROR_CANCELLED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMMAPFILE.CANCELLED",
    identifier: Some("RunMat:Cancelled"),
    when: "The invocation is cancelled while a provider control is pending.",
    message: "memmapfile: invocation cancelled",
};
pub const MEMMAPFILE_ERROR_INTEGER_COMPATIBILITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMMAPFILE.INTEGER_COMPATIBILITY",
    identifier: Some("RunMat:compatibility:MemmapfileIntegerControlsExtension"),
    when: "A typed-integer structural control is used outside RunMat extension mode.",
    message: "memmapfile: typed-integer controls require the RunMat extension",
};
pub const MEMMAPFILE_ERROR_GPU_COMPATIBILITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMMAPFILE.GPU_COMPATIBILITY",
    identifier: Some("RunMat:compatibility:MemmapfileExplicitGpuExtension"),
    when: "An explicit gpuArray control is used outside RunMat extension mode.",
    message: "memmapfile: explicit gpuArray controls require the RunMat extension",
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
            "Integer controls are exact; automatic resident controls gather from their exact owner. Offset and Repeat object properties use UInt64 when a count exceeds exact Double range.",
    },
];
