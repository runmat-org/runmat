use crate::*;
const fn parameter(name: &'static str, description: &'static str) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        description,
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
    }
}
const INPUTS:&[BuiltinParamDescriptor]=&[
    parameter("fun","Actual objective function handle or scalar function name; scalar rows or vectorized population rows."),
    parameter("nvars","Positive integer-valued double number of variables."),
    parameter("lb","Real double lower bounds or []."),
    parameter("ub","Real double upper bounds or []."),
    parameter("options","Canonical particleswarm options or []."),
];
const OUTPUTS:&[BuiltinParamDescriptor]=&[
    parameter("x","Best actually measured finite real row, optionally improved by the selected hybrid; [] for inconsistent bounds."),
    parameter("fval","Actual objective value at x, without reporting replay; [] before evaluation."),
    parameter("exitflag","1 relative best-value stall; 0 iterations; -1 observer; -2 bounds; -3 objective limit; -4 stall time; -5 time."),
    parameter("output","iterations, funccount, message, hybridflag and original initial rngstate."),
    parameter("points","Actual final swarm: X has one row per particle; Fval contains its actual measurements, not personal bests or the hybrid point."),
];
pub const PARTICLESWARM_ERROR_INVALID_ARGUMENT:BuiltinErrorDescriptor=BuiltinErrorDescriptor{
    code:"RM.PARTICLESWARM.INVALID_ARGUMENT",identifier:Some("RunMat:particleswarm:InvalidArgument"),
    when:"A solver/control/objective carrier or geometry is invalid, or a selected numerical operation is not representable.",
    message:"particleswarm: invalid argument",
};
pub const PSWCREATIONUNIFORM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.PSWCREATIONUNIFORM.INVALID_ARGUMENT",
        identifier: Some("RunMat:pswcreationuniform:InvalidArgument"),
        when: "The problem or selected creation controls are invalid.",
        message: "pswcreationuniform: invalid argument",
    };
pub const PSWPLOTBESTF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PSWPLOTBESTF.INVALID_ARGUMENT",
    identifier: Some("RunMat:pswplotbestf:InvalidArgument"),
    when: "The two callback arguments, metrics, state or output count are invalid.",
    message: "pswplotbestf: invalid argument",
};
pub const PARTICLESWARM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "[x,fval,exitflag,output,points] = particleswarm(fun,nvars)",
            inputs: INPUTS.split_at(2).0,
            outputs: OUTPUTS,
        },
        BuiltinSignatureDescriptor {
            label: "[...] = particleswarm(fun,nvars,lb,ub)",
            inputs: INPUTS.split_at(4).0,
            outputs: OUTPUTS,
        },
        BuiltinSignatureDescriptor {
            label: "[...] = particleswarm(fun,nvars,lb,ub,options)",
            inputs: INPUTS,
            outputs: OUTPUTS,
        },
        BuiltinSignatureDescriptor {
            label: "[...] = particleswarm(problem)",
            inputs: &[parameter(
                "problem",
                "objective,nvars,lb,ub,options,solver='particleswarm', and optional rngstate.",
            )],
            outputs: OUTPUTS,
        },
    ],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[PARTICLESWARM_ERROR_INVALID_ARGUMENT],
};
pub const PSWCREATIONUNIFORM_DESCRIPTOR:BuiltinDescriptor=BuiltinDescriptor{
    signatures:&[BuiltinSignatureDescriptor{label:"swarm = pswcreationuniform(problem)",
        inputs:&[parameter("problem","The particleswarm creation problem; only nvars, bounds and creation controls are active.")],
        outputs:&[parameter("swarm","SwarmSize-by-nvars double initial positions; no fitness evaluation or initial velocity draws.")]}],
    output_mode:BuiltinOutputMode::Fixed,completion_policy:BuiltinCompletionPolicy::Public,
    errors:&[PSWCREATIONUNIFORM_ERROR_INVALID_ARGUMENT,PARTICLESWARM_ERROR_INVALID_ARGUMENT],
};
pub const PSWPLOTBESTF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "stop = pswplotbestf(optimValues,state)",
        inputs: &[
            parameter(
                "optimValues",
                "Scalar structure with iteration and bestfval.",
            ),
            parameter("state", "init, iter or done."),
        ],
        outputs: &[parameter(
            "stop",
            "Logical false after successful original-receiver scene update.",
        )],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[PSWPLOTBESTF_ERROR_INVALID_ARGUMENT],
};
pub const PARTICLESWARM_INTEGER_OPTIONS_EXTENSION:BuiltinExtensionDescriptor=BuiltinExtensionDescriptor{
    id:"particleswarm-integer-option",mode:BuiltinExtensionMode::RunMatOnly,
    description:"Exact native-integer scalar controls, gated by the original compatibility policy before provider readback.",
    error_identifier:Some("RunMat:compatibility:ParticleswarmIntegerOptionExtension"),
};
pub const PARTICLESWARM_RESIDENT_OPTIONS_EXTENSION:BuiltinExtensionDescriptor=BuiltinExtensionDescriptor{
    id:"particleswarm-explicit-gpu-option",mode:BuiltinExtensionMode::RunMatOnly,
    description:"Selected explicit-resident numeric controls and InitialPoints children use an original-context host gather; callback captures and inactive controls are not gathered.",
    error_identifier:Some("RunMat:compatibility:ParticleswarmResidentOptionsExtension"),
};
pub(super) const EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    PARTICLESWARM_INTEGER_OPTIONS_EXTENSION,
    PARTICLESWARM_RESIDENT_OPTIONS_EXTENSION,
];

pub(super) const INTEGER_CAPABILITIES:&[BuiltinIntegerCapabilityDescriptor]=&[
    BuiltinIntegerCapabilityDescriptor{
        form:"selected swarm native-integer controls",
        inputs:&[BuiltinIntegerInputCapability{
            name:"active scalar counts, flags and continuous controls",
            classes:&ALL_INTEGER_CLASSES,availability:BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes:"Original compatibility policy admits typed controls before any provider work; this is not a native-integer problem-data overload.",
        }],
        computation_domain:BuiltinIntegerComputationDomain::Structural,
        output_class:BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::StructuralParameter,
        notes:"Counts preserve exact native values and fit usize. Integer continuous controls cross only an exactly representable binary64 boundary.",
    },
];
