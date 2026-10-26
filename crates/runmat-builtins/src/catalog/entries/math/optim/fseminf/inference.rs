use super::super::{nonlinear_facts as facts, program_facts};
use crate::{
    catalog::inference::argument_error, BuiltinCatalogEntry, OptimOptionField, OptimSolver,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let problem = request.arguments.len() == 1;
    let args = if problem {
        match &request.arguments[0].kind {
            ValueKindFact::Struct(fields) => {
                for name in [
                    "objective",
                    "x0",
                    "ntheta",
                    "seminfcon",
                    "solver",
                    "options",
                ] {
                    if fields.fields_complete && !fields.fields.contains_key(name) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-FSEMINF-PROBLEM",
                            "problem requires objective, x0, ntheta, seminfcon, solver and options",
                            0,
                        ));
                    }
                }
                [
                    "objective",
                    "x0",
                    "ntheta",
                    "seminfcon",
                    "Aineq",
                    "bineq",
                    "Aeq",
                    "beq",
                    "lb",
                    "ub",
                    "options",
                ]
                .into_iter()
                .map(|name| {
                    fields.fields.get(name).cloned().unwrap_or_else(|| {
                        if fields.fields_complete {
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
            ValueKindFact::Unknown => vec![ValueFact::unknown(DynamicReason::RuntimeValue); 11],
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FSEMINF-PROBLEM",
                    "single argument must be a scalar problem structure",
                    0,
                ));
                Vec::new()
            }
        }
    } else {
        request.arguments.clone()
    };
    if !problem && !matches!(args.len(), 4 | 6 | 8 | 10 | 11) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FSEMINF-ARITY",
            "expected objective, x0 and complete public constraint/bound forms",
            0,
        ));
    }
    for index in [0, 3] {
        if let Some(value) = args.get(index) {
            if !matches!(
                value.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Callable(_)
                    | ValueKindFact::String
                    | ValueKindFact::Character
            ) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FSEMINF-CALLABLE",
                    "objective/seminfcon must identify a single callable",
                    index,
                ));
            }
        }
    }
    let n = args.get(1).and_then(|value| value.shape.element_count());
    for index in [1, 2, 4, 5, 6, 7, 8, 9] {
        if let Some(value) = args.get(index) {
            if value.storage == StorageFact::Sparse
                || (!matches!(value.kind, ValueKindFact::Unknown)
                    && !matches!(value.kind,ValueKindFact::Numeric(n)if n.domain==NumericDomain::Real && n.class==NumericClass::Double))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FSEMINF-NUMERIC",
                    "numeric roles must be real double",
                    index,
                ));
            }
        }
    }
    if let Some(value) = args.get(2) {
        if value.shape.element_count().is_some_and(|n| n != 1) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FSEMINF-NTHETA",
                "ntheta must be scalar",
                2,
            ));
        }
    }
    if !problem
        && request
            .literals
            .numeric_at(2)
            .is_some_and(|n| !n.is_finite() || n <= 0.0 || n.fract() != 0.0)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-FSEMINF-NTHETA",
            "ntheta must be a positive integer",
            2,
        ));
    }
    if n == Some(0) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FSEMINF-X0",
            "x0 must be nonempty",
            1,
        ));
    }
    let normalized = CallRequest {
        arguments: args.clone(),
        literals: LiteralContext::default(),
        outputs: request.outputs.clone(),
    };
    program_facts::constraint_geometry(&normalized, 1, &[(4, 5), (6, 7)], &[], &mut diagnostics);
    for index in [8, 9] {
        if args
            .get(index)
            .and_then(|value| value.shape.element_count())
            .zip(n)
            .is_some_and(|(count, n)| count > n)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FSEMINF-BOUND",
                "bounds cannot contain more elements than x0",
                index,
            ));
        }
    }
    facts::options(args.get(10), 10, true, &mut diagnostics);
    if let Some(ValueKindFact::Struct(options)) = args.get(10).map(|value| &value.kind) {
        for name in options.fields.keys() {
            if !name.eq_ignore_ascii_case("Solver")
                && OptimOptionField::parse_for_solver(name, OptimSolver::Fseminf).is_none()
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FSEMINF-OPTION",
                    "option is not consumed by the implemented fseminf algorithms",
                    10,
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
        "wallclocktime",
    ] {
        information.insert(name.into(), facts::scalar());
    }
    for name in ["constrviolation", "firstorderopt", "lssteplength"] {
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
            facts::scalar(),
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(information, true))),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        diagnostics,
    );
    for index in [0, 3] {
        if let Some(ValueKindFact::Callable(callable)) = args.get(index).map(|value| &value.kind) {
            result
                .capabilities
                .0
                .extend(callable.capabilities.0.iter().copied());
        }
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count < 4)
    {
        result.effects.0.remove(&EffectKind::Clock);
    }
    result
}
