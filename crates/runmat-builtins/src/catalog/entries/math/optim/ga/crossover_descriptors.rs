use crate::*;
const fn p(name: &'static str, description: &'static str) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        description,
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
    }
}
const INPUTS: &[BuiltinParamDescriptor] = &[
    p("parents","One-based row indices, paired consecutively for each child."),
    p("options","Scalar genetic options context or []; these kernels read no option fields."),
    p("nvars","Positive exact number of columns in the population."),
    p("problem","Scalar R2026a genetic problem context; no callbacks are invoked."),
    p("scores","One finite real score per member for heuristic crossover; unused by other crossover kinds."),
    p("population","Full finite real population with one row per member and nvars columns."),
];
const OUTPUTS: &[BuiltinParamDescriptor] = &[p(
    "children",
    "Host double matrix with one child per consecutive pair of parents.",
)];
pub const GA_CROSSOVER_HELPER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "children=crossoverFcn(parents,options,nvars,problem,scores,population)",
        inputs: INPUTS,
        outputs: OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[super::GA_ERROR_INVALID_ARGUMENT],
};
pub const GA_PARAMETRIC_CROSSOVER_HELPER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "children=crossoverFcn(parents,options,nvars,problem,scores,population)",
            inputs: INPUTS,
            outputs: OUTPUTS,
        },
        BuiltinSignatureDescriptor {
            label: "children=crossoverFcn(parents,options,nvars,problem,scores,population,ratio)",
            inputs: &[
                p("parents", "Consecutive pairs of one-based member indices."),
                p("options", "Scalar context or []."),
                p("nvars", "Positive exact genome length."),
                p("problem", "Scalar genetic problem context."),
                p(
                    "scores",
                    "Finite member scores for heuristic crossover; otherwise unused.",
                ),
                p("population", "Finite real member-by-variable matrix."),
                p(
                    "ratio",
                    "Nonnegative finite scalar, or one per variable for intermediate crossover.",
                ),
            ],
            outputs: OUTPUTS,
        },
    ],
    ..GA_CROSSOVER_HELPER_DESCRIPTOR
};
pub(super) const INTEGERS:&[BuiltinIntegerCapabilityDescriptor]=&[
    BuiltinIntegerCapabilityDescriptor{form:"population, active scores and ratio",inputs:&[BuiltinIntegerInputCapability{name:"population, heuristic scores, optional ratio",classes:&ALL_INTEGER_CLASSES,availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,notes:"Data must convert exactly to binary64 under the GA numeric extension."}],computation_domain:BuiltinIntegerComputationDomain::FloatingPoint,output_class:BuiltinIntegerOutputClassRule::Double,overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,overload:BuiltinIntegerOverloadKind::Multiple,notes:"Arithmetic and result storage use host double."},
    BuiltinIntegerCapabilityDescriptor{form:"parent indices and genome length",inputs:&[BuiltinIntegerInputCapability{name:"parents and nvars",classes:&ALL_INTEGER_CLASSES,availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,notes:"Parents use exact GA numeric conversion; nvars uses the independent exact integer-option gate."}],computation_domain:BuiltinIntegerComputationDomain::Structural,output_class:BuiltinIntegerOutputClassRule::Double,overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,overload:BuiltinIntegerOverloadKind::StructuralParameter,notes:"Indices are range-checked against the actual population; no device kernel is provided."},
];
