use crate::*;
pub(super) const fn parameter(
    name: &'static str,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        description,
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
    }
}
const INPUTS: &[BuiltinParamDescriptor] = &[
    parameter(
        "fun",
        "The actual selected objective returning one finite real double scalar.",
    ),
    parameter(
        "x0",
        "Nonempty finite double vector, or one opaque custom decision object.",
    ),
    parameter("lb", "Double lower bounds; [] means unbounded."),
    parameter("ub", "Double upper bounds; [] means unbounded."),
    parameter("options", "Selected canonical annealing options or []."),
];
const OUTPUTS:&[BuiltinParamDescriptor]=&[
    parameter("x","Best actually measured point, in the original double shape or original custom representation."),
    parameter("fval","Actual objective at x; [] if no objective was evaluated."),
    parameter("exitflag","1 function stall,5 objective,0 budgets,-1 observer,-2 infeasible,-5 time."),
    parameter("output","iterations,funccount,message,initial rngstate,problemtype,temperature and totaltime."),
];
pub const SIMULANNEALBND_ERROR_INVALID_ARGUMENT:BuiltinErrorDescriptor=BuiltinErrorDescriptor{
    code:"RM.SIMULANNEALBND.INVALID_ARGUMENT",identifier:Some("RunMat:simulannealbnd:InvalidArgument"),
    when:"A problem, option, numerical result, callback return, or executable selected role is invalid.",
    message:"simulannealbnd: invalid argument",
};
pub const SIMULANNEALBND_DESCRIPTOR:BuiltinDescriptor=BuiltinDescriptor{
    signatures:&[
        BuiltinSignatureDescriptor{label:"[x,fval,exitflag,output] = simulannealbnd(fun,x0)",inputs:INPUTS.split_at(2).0,outputs:OUTPUTS},
        BuiltinSignatureDescriptor{label:"[...] = simulannealbnd(fun,x0,lb,ub)",inputs:INPUTS.split_at(4).0,outputs:OUTPUTS},
        BuiltinSignatureDescriptor{label:"[...] = simulannealbnd(fun,x0,lb,ub,options)",inputs:INPUTS,outputs:OUTPUTS},
        BuiltinSignatureDescriptor{label:"[...] = simulannealbnd(problem)",inputs:&[parameter("problem",
            "Complete objective,x0,lb,ub,solver='simulannealbnd',options and optional rngstate.")],outputs:OUTPUTS},
    ],
    output_mode:BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy:BuiltinCompletionPolicy::Public,errors:&[SIMULANNEALBND_ERROR_INVALID_ARGUMENT],
};
pub const SIMULANNEALBND_INTEGER_OPTIONS_EXTENSION:BuiltinExtensionDescriptor=BuiltinExtensionDescriptor{
    id:"simulannealbnd-integer-option",mode:BuiltinExtensionMode::RunMatOnly,
    description:"Exact native-integer controls before any selected option provider readback; not an integer decision-data overload.",
    error_identifier:Some("RunMat:compatibility:SimulannealbndIntegerOptionsExtension"),
};
pub const SIMULANNEALBND_RESIDENT_OPTIONS_EXTENSION:BuiltinExtensionDescriptor=BuiltinExtensionDescriptor{
    id:"simulannealbnd-explicit-gpu-option",mode:BuiltinExtensionMode::RunMatOnly,
    description:"Original-context host gather for selected resident numeric options, never callback captures or opaque custom data.",
    error_identifier:Some("RunMat:compatibility:SimulannealbndResidentOptionsExtension"),
};
pub(super) const EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    SIMULANNEALBND_INTEGER_OPTIONS_EXTENSION,
    SIMULANNEALBND_RESIDENT_OPTIONS_EXTENSION,
];
pub(super) const INTEGER_CAPABILITIES:&[BuiltinIntegerCapabilityDescriptor]=&[
    BuiltinIntegerCapabilityDescriptor{
        form:"active native-integer annealing controls",
        inputs:&[BuiltinIntegerInputCapability{
            name:"counts and continuous numeric controls",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes:"Counts retain exact structural usize range; continuous controls cross only exact binary64 boundaries.",
        }],
        computation_domain:BuiltinIntegerComputationDomain::Structural,
        output_class:BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::StructuralParameter,
        notes:"No native-integer problem-data or objective-output overload.",
    },
];
