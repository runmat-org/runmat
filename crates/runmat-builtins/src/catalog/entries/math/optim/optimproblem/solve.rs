//! Typed problem models use the selected solver's actual effect/control owner.
use super::*;
use crate::*;
mod descriptor;
mod parameter;
pub use parameter::OptimizationSolveParameter;
mod documentation;
mod inference;
pub const OPTIMPROBLEM_SOLVE_CATALOG_ENTRY: BuiltinCatalogEntry =
    entry("solve", documentation::DOCUMENTATION);
pub const OPTIMPROBLEM_SOLVE_METHOD_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "optim.problemdef.OptimizationProblem.solve",
    BuiltinDocumentation {
        title: Some("OptimizationProblem.solve"),
        slug: Some("optimization-problem-solve"),
        examples: &[],
        example_exemption: Some("Covered by the public solve examples."),
        ..documentation::DOCUMENTATION
    },
);
pub use descriptor::*;
const INTEGER_OPTIONS:&[BuiltinIntegerCapabilityDescriptor]=&[
    BuiltinIntegerCapabilityDescriptor{
        form:"solve(problem,...,Options=options)",
        inputs:&[BuiltinIntegerInputCapability{name:"MaxIterations / MaxIter",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes:"The selected solver decodes the exact nonnegative structural count and requires it to fit the selected solver's structural count (u32 for the conic adapters, usize for fmincon and fminunc); zero is honored."}],
        computation_domain:BuiltinIntegerComputationDomain::Structural,output_class:BuiltinIntegerOutputClassRule::Double,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::StructuralParameter,notes:"The selected linprog, lsqlin, quadprog, fmincon or fminunc option extension is checked independently of model data.",
    },
    BuiltinIntegerCapabilityDescriptor{
        form:"solve(problem,...,Options=lsqlin_or_quadprog_options)",
        inputs:&[BuiltinIntegerInputCapability{name:"OptimalityTolerance / TolFun and ConstraintTolerance / TolCon",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::Allowed,
            notes:"The selected lsqlin or quadprog profile requires positive finite tolerances exactly representable as binary64."}],
        computation_domain:BuiltinIntegerComputationDomain::FloatingPoint,output_class:BuiltinIntegerOutputClassRule::Double,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::StructuralParameter,notes:"No new integer model-data or solver-form support is implied.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form:"solve(problem,...,Options=intlinprog_options)",
        inputs:&[BuiltinIntegerInputCapability{name:"MaxNodes and MaxFeasiblePoints",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::Allowed,
            notes:"Structural limits retain exact addressable values and must fit usize; zero/Inf admission follows the selected intlinprog profile."}],
        computation_domain:BuiltinIntegerComputationDomain::Structural,output_class:BuiltinIntegerOutputClassRule::Double,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::StructuralParameter,notes:"Counts remain actual node relaxations and certified feasible point discoveries.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form:"solve(problem,...,Options=intlinprog_options)",
        inputs:&[BuiltinIntegerInputCapability{name:"AbsoluteGapTolerance, RelativeGapTolerance, ConstraintTolerance, MaxTime and ObjectiveCutOff",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::Allowed,
            notes:"Non-count integer controls cross the existing exact binary64 scalar boundary and retain their individual range checks."}],
        computation_domain:BuiltinIntegerComputationDomain::FloatingPoint,output_class:BuiltinIntegerOutputClassRule::Double,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::StructuralParameter,notes:"This does not change model data-class or variable-domain semantics.",
    },
];
const fn entry(name: &'static str, documentation: BuiltinDocumentation) -> BuiltinCatalogEntry {
    let mut value = super::entry::entry(
        name,
        &OPTIMPROBLEM_SOLVE_DESCRIPTOR,
        documentation,
        inference::infer,
    );
    value.contract.effects = FMINCON_CATALOG_ENTRY.contract.effects;
    value.extensions = &[
        LINPROG_GPU_INPUT_EXTENSION,
        LINPROG_INTEGER_OPTION_EXTENSION,
        LSQLIN_INTEGER_OPTION_EXTENSION,
        QUADPROG_INTEGER_OPTION_EXTENSION,
        QUADPROG_GPU_INPUT_EXTENSION,
        FMINCON_INTEGER_OPTION_EXTENSION,
        FMINCON_RESIDENT_EXTENSION,
        FMINUNC_OPTION_EXTENSION,
        FMINUNC_RESIDENT_EXTENSION,
        INTLINPROG_INTEGER_EXTENSION,
    ];
    value.integer_capabilities = INTEGER_OPTIONS;
    value
}
