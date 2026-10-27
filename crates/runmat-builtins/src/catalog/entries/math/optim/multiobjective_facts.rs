//! Shared public-form facts for minimax and goal attainment. Opaque callback
//! values/shapes and unavailable stopped outputs are not invented statically.
use super::{nonlinear_facts as facts, program_facts};
use crate::{
    catalog::inference::argument_error, BuiltinCatalogEntry, OptimOptionField, OptimSolver,
};
use runmat_types::*;
pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    solver: OptimSolver,
) -> CallInference {
    let goal = solver == OptimSolver::Fgoalattain;
    let offset = if goal { 2 } else { 0 };
    let mut diagnostics = Vec::new();
    let problem = request.arguments.len() == 1;
    let names: &[&str] = if goal {
        &[
            "objective",
            "x0",
            "goal",
            "weight",
            "Aineq",
            "bineq",
            "Aeq",
            "beq",
            "lb",
            "ub",
            "nonlcon",
            "options",
        ]
    } else {
        &[
            "objective",
            "x0",
            "Aineq",
            "bineq",
            "Aeq",
            "beq",
            "lb",
            "ub",
            "nonlcon",
            "options",
        ]
    };
    let args = if problem {
        match &request.arguments[0].kind {
            ValueKindFact::Struct(fields) => {
                let required: &[&str] = if goal {
                    &["objective", "x0", "goal", "weight", "solver", "options"]
                } else {
                    &["objective", "x0", "solver", "options"]
                };
                for &name in required {
                    if fields.fields_complete && !fields.fields.contains_key(name) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-MULTIOBJECTIVE-PROBLEM",
                            "missing required problem field",
                            0,
                        ));
                    }
                }
                names
                    .iter()
                    .map(|name| {
                        fields.fields.get(*name).cloned().unwrap_or_else(|| {
                            if fields.fields_complete && !required.contains(name) {
                                facts::double(
                                    ShapeFact::from(vec![Some(0), Some(0)]),
                                    StorageFact::Dense,
                                )
                            } else {
                                ValueFact::unknown(DynamicReason::RuntimeValue)
                            }
                        })
                    })
                    .collect::<Vec<_>>()
            }
            ValueKindFact::Unknown => {
                vec![ValueFact::unknown(DynamicReason::RuntimeValue); names.len()]
            }
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-MULTIOBJECTIVE-PROBLEM",
                    "single input must be a scalar problem structure",
                    0,
                ));
                Vec::new()
            }
        }
    } else {
        request.arguments.clone()
    };
    if !problem
        && !(if goal {
            matches!(args.len(), 4 | 6 | 8 | 10 | 11 | 12)
        } else {
            matches!(args.len(), 2 | 4 | 6 | 8 | 9 | 10)
        })
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-MULTIOBJECTIVE-ARITY",
            "expected a complete public input form",
            0,
        ));
    }
    for index in [0, 8 + offset] {
        if let Some(value) = args.get(index) {
            let empty = index != 0 && value.shape.element_count() == Some(0);
            if !empty
                && !matches!(
                    value.kind,
                    ValueKindFact::Unknown
                        | ValueKindFact::Callable(_)
                        | ValueKindFact::String
                        | ValueKindFact::Character
                )
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-MULTIOBJECTIVE-CALLABLE",
                    "objective and nonlcon must identify a single callable",
                    index,
                ));
            }
        }
    }
    for index in (1..args.len().min(8 + offset)).filter(|index| *index != 0) {
        if let Some(value) = args.get(index) {
            if !matches!(value.kind, ValueKindFact::Unknown | ValueKindFact::Logical)
                && !matches!(value.kind,ValueKindFact::Numeric(n) if n.domain==NumericDomain::Real)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-MULTIOBJECTIVE-NUMERIC",
                    "numeric inputs must be real",
                    index,
                ));
            }
        }
    }
    if let Some(ValueKindFact::Struct(options)) = args.get(9 + offset).map(|value| &value.kind) {
        for name in options
            .fields
            .keys()
            .filter(|name| !name.eq_ignore_ascii_case("Solver"))
        {
            if OptimOptionField::parse_for_solver(name, solver).is_none() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-MULTIOBJECTIVE-OPTION",
                    "option is not implemented by this multiobjective solver",
                    9 + offset,
                ));
            }
        }
    }
    if problem {
        for diagnostic in &mut diagnostics {
            diagnostic.argument = Some(0);
        }
    }
    let numeric = || facts::double(ShapeFact::Unknown, StorageFact::Dense);
    let mut information = std::collections::BTreeMap::new();
    for name in [
        "iterations",
        "funcCount",
        "objectiveCallCount",
        "constraintCallCount",
        "stepsize",
    ] {
        information.insert(name.into(), facts::scalar());
    }
    for name in ["lssteplength", "constrviolation", "firstorderopt"] {
        information.insert(name.into(), numeric());
    }
    for name in ["algorithm", "message"] {
        information.insert(name.into(), ValueFact::scalar(ValueKindFact::String));
    }
    let mut result = program_facts::finish(
        entry,
        request,
        vec![
            facts::like(args.get(1)),
            numeric(),
            numeric(),
            facts::scalar(),
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(information, true))),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        diagnostics,
    );
    for index in [0, 8 + offset] {
        if let Some(ValueKindFact::Callable(callable)) = args.get(index).map(|value| &value.kind) {
            result
                .capabilities
                .0
                .extend(callable.capabilities.0.iter().copied());
        }
    }
    result
}
