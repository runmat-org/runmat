use crate::*;

pub const DATESHIFT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [DATETIME_GPU_INPUT_EXTENSION];
pub const DATESHIFT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "integer-valued weekday or occurrence rule",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The public API documents integer-valued numeric controls without a per-storage-class table. RunMat's settled compatibility coverage accepts all eight integer classes, reads them exactly before calendar shifting, and also accepts ordinary integer-valued doubles.",
}];
pub const DATESHIFT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "out = dateshift(t, boundary, integer_weekday_or_rule)",
    inputs: &DATESHIFT_INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
    notes: "Weekday indices must be in 1..7. Occurrence controls must fit int64 and the supported calendar range; day-target occurrences use bounded whole-week arithmetic. Exact native integer storage is read without a float round-trip. Automatic and explicit resident controls follow the existing declared datetime input extension.",
}];
pub const DATESHIFT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATESHIFT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:dateshift:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "dateshift: too many output arguments",
};
pub const DATESHIFT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "out = dateshift(t, boundary, unit, rule)",
            inputs: &[
                BuiltinParamDescriptor {name: "t",ty: BuiltinParamType::Any,arity: BuiltinParamArity::Required,default:None,description:"Host datetime value; numeric serials and relative text are not converted."},
                BuiltinParamDescriptor {name:"boundary",ty:BuiltinParamType::StringScalar,arity:BuiltinParamArity::Required,default:None,description:"'start' or 'end'."},
                BuiltinParamDescriptor {name:"unit",ty:BuiltinParamType::StringScalar,arity:BuiltinParamArity::Required,default:None,description:"Singular year, quarter, month, week, day, hour, minute, or second."},
                BuiltinParamDescriptor {name:"rule",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Optional,default:Some("'current'"),description:"current, next, previous, nearest, or an exact integer scalar/array occurrence offset."},
            ],
            outputs:&[OUTPUT],
        },
        BuiltinSignatureDescriptor {
            label: "out = dateshift(t, 'dayofweek', weekday, rule)",
            inputs: &[
                BuiltinParamDescriptor {name:"t",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Required,default:None,description:"Host datetime scalar or array."},
                BuiltinParamDescriptor {name:"boundary",ty:BuiltinParamType::StringScalar,arity:BuiltinParamArity::Required,default:Some("'dayofweek'"),description:"Day-target selection."},
                BuiltinParamDescriptor {name:"weekday",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Required,default:None,description:"A scalar weekday name, weekday/weekend group, or exact numeric indices 1..7."},
                BuiltinParamDescriptor {name:"rule",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Optional,default:Some("'next'"),description:"current, next, previous, nearest, or exact integer occurrence controls; scalar or same size."},
            ],
            outputs:&[OUTPUT],
        },
    ],
    output_mode:BuiltinOutputMode::Fixed,
    completion_policy:BuiltinCompletionPolicy::Public,
    errors:&[DATETIME_ERROR_INVALID_INPUT, DATETIME_ERROR_INVALID_ARGUMENT, DATESHIFT_ERROR_TOO_MANY_OUTPUTS],
};
const OUTPUT:BuiltinParamDescriptor=BuiltinParamDescriptor {
    name:"out",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Required,default:None,
    description:"Host datetime object, preserving input Format and exact non-scalar shape under scalar expansion, including empty geometry.",
};
