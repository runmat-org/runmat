use crate::*;
pub const GA_PLOT_HELPER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "state=gaplotFcn(options,state,flag)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "options",
                description:
                    "Scalar genetic options or []. Only the selected chart's controls are consumed.",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
            },
            BuiltinParamDescriptor {
                name: "state",
                description:
                    "Scalar genetic iteration state with the selected chart's numerical fields.",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
            },
            BuiltinParamDescriptor {
                name: "flag",
                description:
                    "init,iter,interrupt,done; predefined charts update at outer-generation events.",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
            },
        ],
        outputs: &[BuiltinParamDescriptor {
            name: "state",
            description: "The same state value, including unchanged user fields and StopFlag.",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
        }],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[super::GA_ERROR_INVALID_ARGUMENT],
};
pub(super) const INTEGERS:&[BuiltinIntegerCapabilityDescriptor]=&[
    BuiltinIntegerCapabilityDescriptor{form:"active chart data",inputs:&[BuiltinIntegerInputCapability{name:"selected state numeric fields",classes:&ALL_INTEGER_CLASSES,availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,notes:"The GA numeric extension admits exact binary64 conversion for consumed values."}],computation_domain:BuiltinIntegerComputationDomain::FloatingPoint,output_class:BuiltinIntegerOutputClassRule::NotApplicable,overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,overload:BuiltinIntegerOverloadKind::Multiple,notes:"Chart payloads use host double; the public output is the unchanged state structure."},
    BuiltinIntegerCapabilityDescriptor{form:"structural counters and selected options",inputs:&[BuiltinIntegerInputCapability{name:"Generation and selected stopping/structural fields",classes:&ALL_INTEGER_CLASSES,availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,notes:"Exact structural counts use the independent GA integer-option gate. StartTime is an existing opaque uint64 stopwatch token, not numerical plot data."}],computation_domain:BuiltinIntegerComputationDomain::Structural,output_class:BuiltinIntegerOutputClassRule::NotApplicable,overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,overload:BuiltinIntegerOverloadKind::StructuralParameter,notes:"No public integer result is manufactured and arbitrary user fields are not traversed."},
];
