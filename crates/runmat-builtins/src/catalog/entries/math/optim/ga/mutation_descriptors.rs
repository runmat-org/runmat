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
const INPUTS:&[BuiltinParamDescriptor]=&[
    p("parents","One-based actual population row indices, one per requested child."),
    p("options","Scalar genetic options or []; only population representation/range, active schedule and feasibility controls are consumed."),
    p("nvars","Positive exact genome length."),
    p("problem","Scalar R2026a operator context with FitnessFcn and active domain inputs; no fitness callback is executed."),
    p("state","Scalar generation state; Gaussian reads Generation, feasible kernels read Best history."),
    p("scores","Inactive for these predefined mutation kernels."),
    p("population","Full real member-by-nvars population."),
];
const OUTPUTS: &[BuiltinParamDescriptor] = &[p(
    "children",
    "Host double child matrix in the requested parent order.",
)];
pub const GA_MUTATION_HELPER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "children=mutationFcn(parents,options,nvars,problem,state,scores,population)",
        inputs: INPUTS,
        outputs: OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[super::GA_ERROR_INVALID_ARGUMENT],
};
pub const GA_PARAMETRIC_MUTATION_HELPER_DESCRIPTOR:BuiltinDescriptor=BuiltinDescriptor{
    signatures:&[
        BuiltinSignatureDescriptor{label:"children=mutationFcn(parents,options,nvars,problem,state,scores,population)",inputs:INPUTS,outputs:OUTPUTS},
        BuiltinSignatureDescriptor{label:"children=mutationFcn(parents,options,nvars,problem,state,scores,population,parameters...)",inputs:&[
            p("parents","One-based member indices."),p("options","Active genetic mutation controls."),p("nvars","Positive exact genome length."),p("problem","Original operator problem context."),p("state","Mutation schedule state."),p("scores","Inactive score context."),p("population","Real member-by-variable matrix."),
            BuiltinParamDescriptor{name:"parameters",description:"Gaussian accepts scale and optional shrink; uniform accepts rate. Other mutation kinds accept no parameters.",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Variadic,default:None}],outputs:OUTPUTS}],
    ..GA_MUTATION_HELPER_DESCRIPTOR
};
pub(super) const INTEGERS:&[BuiltinIntegerCapabilityDescriptor]=&[
    BuiltinIntegerCapabilityDescriptor{form:"active population and original numeric domain",inputs:&[BuiltinIntegerInputCapability{name:"parents,population,domain,active history",classes:&ALL_INTEGER_CLASSES,availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,notes:"Active data must convert exactly through the declared GA numeric extension."}],computation_domain:BuiltinIntegerComputationDomain::FloatingPoint,output_class:BuiltinIntegerOutputClassRule::Double,overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,overload:BuiltinIntegerOverloadKind::Multiple,notes:"No integer result arithmetic or explicit-device kernel is provided."},
    BuiltinIntegerCapabilityDescriptor{form:"selected structural and mutation controls",inputs:&[BuiltinIntegerInputCapability{name:"nvars,Generation and selected genetic option controls",classes:&ALL_INTEGER_CLASSES,availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,notes:"Structural counts retain exact conversion; continuous controls require exact binary64 conversion under their independent option gate."}],computation_domain:BuiltinIntegerComputationDomain::Structural,output_class:BuiltinIntegerOutputClassRule::Double,overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,overload:BuiltinIntegerOverloadKind::StructuralParameter,notes:"Only consumed controls are admitted; inactive scores and callbacks are not traversed."},
];
