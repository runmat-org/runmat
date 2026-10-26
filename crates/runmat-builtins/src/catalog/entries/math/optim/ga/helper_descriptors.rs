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
const SELECTION: &[BuiltinParamDescriptor] = &[
    p("expectation","Full real nonnegative finite scaled fitness weights."),
    p("nParents","Nonnegative exact integer-valued requested parent count."),
    p("options","Scalar genetic options context, or []; these predefined selectors consume no option fields."),
];
const SCALING: &[BuiltinParamDescriptor] = &[
    p(
        "scores",
        "Full real score vector, one value per population member.",
    ),
    p(
        "nParents",
        "Nonnegative exact integer-valued total offspring expectation.",
    ),
];
const SELECTED: &[BuiltinParamDescriptor] = &[p(
    "parents",
    "One-based host double row of exactly nParents selected indices.",
)];
const EXPECTATION: &[BuiltinParamDescriptor] = &[p(
    "expectation",
    "Host double column, one entry per score, with nonnegative entries summing to nParents.",
)];
const fn d(signatures: &'static [BuiltinSignatureDescriptor]) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &[super::GA_ERROR_INVALID_ARGUMENT],
    }
}
pub const GA_SELECTION_HELPER_DESCRIPTOR: BuiltinDescriptor = d(&[BuiltinSignatureDescriptor {
    label: "parents = selection(expectation,nParents,options)",
    inputs: SELECTION,
    outputs: SELECTED,
}]);
pub const GA_TOURNAMENT_HELPER_DESCRIPTOR: BuiltinDescriptor = d(&[
    BuiltinSignatureDescriptor {
        label: "parents = selectiontournament(expectation,nParents,options)",
        inputs: SELECTION,
        outputs: SELECTED,
    },
    BuiltinSignatureDescriptor {
        label: "parents = selectiontournament(expectation,nParents,options,size)",
        inputs: &[
            p("expectation", "Nonnegative finite scaled fitness weights."),
            p("nParents", "Nonnegative exact parent count."),
            p("options", "Scalar options context or []."),
            p(
                "size",
                "Exact tournament size >=2; default four for this standalone helper.",
            ),
        ],
        outputs: SELECTED,
    },
]);
pub const GA_SCALING_HELPER_DESCRIPTOR: BuiltinDescriptor = d(&[BuiltinSignatureDescriptor {
    label: "expectation = fitscaling(scores,nParents)",
    inputs: SCALING,
    outputs: EXPECTATION,
}]);
pub const GA_PARAMETRIC_SCALING_HELPER_DESCRIPTOR: BuiltinDescriptor = d(&[
    BuiltinSignatureDescriptor {label:"expectation = fitscaling(scores,nParents)",inputs:SCALING,outputs:EXPECTATION},
    BuiltinSignatureDescriptor {label:"expectation = fitscaling(scores,nParents,parameter)",inputs:&[
        p("scores","Full real population score vector."),p("nParents","Nonnegative exact offspring count."),
        p("parameter","Top quantity: fraction [0,1] or integer count, default0.4; shift-linear rate: finite >=1, default2."),
    ],outputs:EXPECTATION},
]);
const CREATION_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "Population = creationFcn(nvars,problem,options)",
    inputs: &[
        p("nvars", "Positive exact genome length."),
        p(
            "problem",
            "R2026a operator problem context with FitnessFcn and selected constraint roles.",
        ),
        p(
            "options",
            "Genetic options; only creation controls are active.",
        ),
    ],
    outputs: &[p(
        "Population",
        "PopulationSize-by-nvars host double initial population.",
    )],
}];
pub const GA_CREATION_HELPER_DESCRIPTOR: BuiltinDescriptor = d(CREATION_SIGNATURES);
const NONLINEAR_CREATION_SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "Population = gacreationnonlinearfeasible(nvars,problem,options)",
        inputs: CREATION_SIGNATURES[0].inputs,
        outputs: CREATION_SIGNATURES[0].outputs,
    },
    BuiltinSignatureDescriptor {
        label: "Population = gacreationnonlinearfeasible(nvars,problem,options,Name,Value,...)",
        inputs: &[
            p("nvars", "Positive exact genome length."),
            p("problem", "Original constraint context including nonlcon."),
            p("options", "Active genetic creation controls."),
            BuiltinParamDescriptor {
                name: "creationParameters",
                description: "Unique SolverOpts,UseParallel,NumStartPts name/value pairs.",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Variadic,
                default: None,
            },
        ],
        outputs: CREATION_SIGNATURES[0].outputs,
    },
];
pub const GA_NONLINEAR_CREATION_HELPER_DESCRIPTOR: BuiltinDescriptor =
    d(NONLINEAR_CREATION_SIGNATURES);
pub(super) const INTEGERS:&[BuiltinIntegerCapabilityDescriptor]=&[
    BuiltinIntegerCapabilityDescriptor {
        form:"real score or expectation vector",
        inputs:&[BuiltinIntegerInputCapability{name:"scores or expectation",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes:"The selected data vector must cross exactly to binary64 under the GA numeric extension."}],
        computation_domain:BuiltinIntegerComputationDomain::FloatingPoint,output_class:BuiltinIntegerOutputClassRule::Double,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::Multiple,notes:"Outputs are double indices or double expectations, not integer arithmetic."},
    BuiltinIntegerCapabilityDescriptor {
        form:"exact parent count and selected operator control",
        inputs:&[BuiltinIntegerInputCapability{name:"nParents and optional size/quantity/rate",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes:"The GA integer-option extension admits structural counts exactly; continuous parameters require exact binary64 conversion."}],
        computation_domain:BuiltinIntegerComputationDomain::Structural,output_class:BuiltinIntegerOutputClassRule::Double,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::StructuralParameter,notes:"Authenticated automatic materialization only; no explicit-GPU helper extension."},
];
