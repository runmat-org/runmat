use crate::*;

pub const TIMERANGE_NUMERIC_BOUNDS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "timerange-numeric-bounds",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "timerange with numeric row-time bounds is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TimerangeNumericBoundsExtension"),
    };
pub const TIMERANGE_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "timerange-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "timerange with explicitly GPU-resident input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TimerangeExplicitGpuInputExtension"),
    };
pub const TIMERANGE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    TIMERANGE_NUMERIC_BOUNDS_EXTENSION,
    TIMERANGE_EXPLICIT_GPU_EXTENSION,
];
const TIMERANGE_INTEGER_BOUND_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "start or end numeric row-time bound",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Implemented compatibility bounds are datetime or duration. Numeric row-time bounds remain available only in RunMat mode; datetime-unit and event-filter overloads are not implemented.",
    }];
pub const TIMERANGE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "S = timerange(integer_start, integer_end [, interval_type])",
        inputs: &TIMERANGE_INTEGER_BOUND_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Numeric extension bounds retain their native scalar identity and compare with numeric timetable row times through exact mixed-class ordering rather than f64 conversion.",
    }];
const TIMERANGE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Timetable row-time subscript selector.",
}];
const TIMERANGE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "startTime",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Beginning of the selected row-time interval.",
    },
    BuiltinParamDescriptor {
        name: "endTime",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "End of the selected row-time interval.",
    },
    BuiltinParamDescriptor {
        name: "intervalType",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("openright"),
        description: "One of openright, openleft, open, or closed.",
    },
];
const TIMERANGE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "S = timerange(startTime, endTime, intervalType)",
    inputs: &TIMERANGE_INPUTS,
    outputs: &TIMERANGE_OUTPUT,
}];
pub const TIMERANGE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TIMERANGE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[super::super::super::shared::TABLE_ERROR_INVALID_ARGUMENT],
};
