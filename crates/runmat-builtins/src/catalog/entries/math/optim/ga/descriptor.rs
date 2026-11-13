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
const INPUTS: &[BuiltinParamDescriptor] = &[
    parameter(
        "fun",
        "Selected scalar fitness function; vectorized functions receive population rows.",
    ),
    parameter("nvars", "Positive exact integer-valued variable count."),
    parameter("A", "Full or sparse real linear inequality matrix, or []."),
    parameter("b", "Real right side for A*x<=b, or []."),
    parameter("Aeq", "Full or sparse real equality matrix, or []."),
    parameter("beq", "Real right side for Aeq*x=beq, or []."),
    parameter("lb", "Lower-bound prefix, or [] for unbounded coordinates."),
    parameter("ub", "Upper-bound prefix, or [] for unbounded coordinates."),
    parameter(
        "nonlcon",
        "Selected two-output c<=0, ceq=0 function, or [].",
    ),
    parameter("intcon", "Exact one-based integer-variable indices, or []."),
    parameter("options", "Canonical ga options or []."),
];
const OUTPUTS: &[BuiltinParamDescriptor] = &[
    parameter("x", "Best retained feasible point with finite real declared or measured fitness; [] before any such observation."),
    parameter("fval", "Raw fitness belonging to x, never a penalty or augmented selection score; [] when x is empty."),
    parameter("exitflag", "1 fixed-merit/feasibility tolerance; 3 unchanged; 5 fitness limit; 0 generations; -1 observer; -2 no feasible fitness; -4 stall time; -5 time; -9 unverified proposal."),
    parameter("output", "Generation, point/call counts, timing, constraint violation, initial RNG state, hybrid details, message and fitness provenance."),
    parameter("population", "Last completed population, excluding any separate hybrid point; [] before a completed initial population."),
    parameter("scores", "Last population raw fitness values, or penalty selection scores for mixed-integer problems; not the separate hybrid score."),
];
const fn input(index: usize) -> BuiltinParamDescriptor {
    parameter(INPUTS[index].name, INPUTS[index].description)
}
pub const GA_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GA.INVALID_ARGUMENT", identifier: Some("RunMat:ga:InvalidArgument"),
    when: "A public argument, active option, operator result or callback state violates its representation, geometry or numerical contract.",
    message: "ga: invalid argument",
};
pub const GA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor { label: "[x,fval,exitflag,output,population,scores] = ga(fun,nvars)", inputs: INPUTS.split_at(2).0, outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = ga(fun,nvars,A,b)", inputs: INPUTS.split_at(4).0, outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = ga(fun,nvars,A,b,Aeq,beq)", inputs: INPUTS.split_at(6).0, outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = ga(fun,nvars,A,b,Aeq,beq,lb,ub)", inputs: INPUTS.split_at(8).0, outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = ga(fun,nvars,A,b,Aeq,beq,lb,ub,nonlcon)", inputs: INPUTS.split_at(9).0, outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = ga(fun,nvars,A,b,Aeq,beq,lb,ub,nonlcon,options)", inputs: &[
            input(0),input(1),input(2),input(3),input(4),input(5),input(6),input(7),input(8),input(10),
        ], outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = ga(fun,nvars,A,b,Aeq,beq,lb,ub,nonlcon,intcon)", inputs: INPUTS.split_at(10).0, outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = ga(fun,nvars,A,b,Aeq,beq,lb,ub,nonlcon,intcon,options)", inputs: INPUTS, outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = ga(problem)", inputs: &[parameter("problem", "Scalar structure with fitnessfcn,nvars,options and optional Aineq,Bineq,Aeq,Beq,lb,ub,nonlcon,intcon,solver='ga',rngstate.")], outputs: OUTPUTS },
    ],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[GA_ERROR_INVALID_ARGUMENT],
};
pub const GA_NUMERIC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ga-native-numeric", mode: BuiltinExtensionMode::RunMatOnly,
    description: "Role-admitted exact native-integer and logical problem data, population data, supplied scores and nonlinear results convert to host double. Structural indices remain exact; wide native values that cannot cross exactly are rejected.",
    error_identifier: Some("RunMat:compatibility:GaNumericExtension"),
};
pub const GA_SINGLE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ga-single-to-double", mode: BuiltinExtensionMode::RunMatOnly,
    description: "Role-admitted real single problem/population data, supplied scores and nonlinear results promote to host double. Objective measurements themselves require double.",
    error_identifier: Some("RunMat:compatibility:GaSingleExtension"),
};
pub const GA_INTEGER_OPTIONS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ga-integer-option", mode: BuiltinExtensionMode::RunMatOnly,
    description: "Active native-integer scalar/array options are admitted before provider transfer; structural controls fit usize and continuous controls cross only an exact double boundary.",
    error_identifier: Some("RunMat:compatibility:GaIntegerOptionExtension"),
};
pub(super) const EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    GA_NUMERIC_EXTENSION,
    GA_SINGLE_EXTENSION,
    GA_INTEGER_OPTIONS_EXTENSION,
];
pub(super) const INTEGERS: &[BuiltinIntegerCapabilityDescriptor] = &[
    BuiltinIntegerCapabilityDescriptor {
        form: "selected real numeric GA roles",
        inputs: &[BuiltinIntegerInputCapability { name: "role-admitted problem arrays, initial/returned populations, supplied fitness and nonlinear arrays",
            classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes: "Original compatibility admission precedes transfer; each value must convert exactly to binary64. Objective results retain their separate double-only contract." }],
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "This is not integer arithmetic or a GPU genetic kernel. Authenticated automatic residency is transparent; explicit/unproven residency must be gathered by the caller.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "selected exact structural and numeric GA controls",
        inputs: &[BuiltinIntegerInputCapability { name: "nvars,intcon and active numeric option counts/arrays",
            classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes: "Structural integers decode exactly to usize and are range checked; other integer controls require exact double conversion." }],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "An option or structural extension never grants callback, memory, device or timing permission.",
    },
];
