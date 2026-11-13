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
                for name in ["objective", "x0", "solver", "options"] {
                    if fields.fields_complete && !fields.fields.contains_key(name) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-FMINCON-PROBLEM",
                            "problem requires objective, x0, solver and options",
                            0,
                        ));
                    }
                }
                [
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
            ValueKindFact::Unknown => vec![ValueFact::unknown(DynamicReason::RuntimeValue); 10],
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FMINCON-PROBLEM",
                    "single argument must be a scalar problem structure",
                    0,
                ));
                Vec::new()
            }
        }
    } else {
        request.arguments.clone()
    };
    if !problem && !matches!(args.len(), 4 | 6 | 8 | 9 | 10) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FMINCON-ARITY",
            "expected objective, x0 and complete public constraint/bound forms",
            0,
        ));
    }
    for index in [0, 8] {
        if let Some(value) = args.get(index) {
            let empty = index == 8
                && value.shape.element_count() == Some(0)
                && matches!(value.kind, ValueKindFact::Numeric(_));
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
                    "RM-CATALOG-FMINCON-CALLABLE",
                    "objective/nonlcon must identify a single callable",
                    index,
                ));
            }
        }
    }
    let n = args.get(1).and_then(|value| value.shape.element_count());
    for index in [1, 2, 3, 4, 5, 6, 7] {
        if let Some(value) = args.get(index) {
            if !matches!(value.kind, ValueKindFact::Unknown | ValueKindFact::Logical)
                && !matches!(value.kind,ValueKindFact::Numeric(n)if n.domain==NumericDomain::Real)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FMINCON-NUMERIC",
                    "numeric roles must be real",
                    index,
                ));
            }
        }
    }
    if n == Some(0) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FMINCON-X0",
            "x0 must be nonempty",
            1,
        ));
    }
    let normalized = CallRequest {
        arguments: args.clone(),
        literals: LiteralContext::default(),
        outputs: request.outputs.clone(),
    };
    program_facts::constraint_geometry(&normalized, 1, &[(2, 3), (4, 5)], &[], &mut diagnostics);
    for index in [6, 7] {
        if args
            .get(index)
            .and_then(|value| value.shape.element_count())
            .zip(n)
            .is_some_and(|(count, n)| count > n)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FMINCON-BOUND",
                "bounds cannot contain more elements than x0",
                index,
            ));
        }
    }
    facts::options(args.get(9), 9, true, &mut diagnostics);
    if let Some(ValueKindFact::Struct(options)) = args.get(9).map(|value| &value.kind) {
        for name in options.fields.keys() {
            if !name.eq_ignore_ascii_case("Solver")
                && OptimOptionField::parse_for_solver(name, OptimSolver::Fmincon).is_none()
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FMINCON-OPTION",
                    "option is not consumed by the implemented fmincon algorithms",
                    9,
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
    // An Algorithm field's presence is not a proof of its string value.
    let options = args.get(9).and_then(|value| match &value.kind {
        ValueKindFact::Struct(fields) => Some(fields),
        _ => None,
    });
    let may_reflective = args.get(9).is_some_and(|value| {
        value.shape.element_count() != Some(0)
            && options.is_none_or(|fields| {
                !fields.fields_complete
                    || fields.fields.keys().any(|name| {
                        OptimOptionField::parse_for_solver(name, OptimSolver::Fmincon)
                            == Some(OptimOptionField::Algorithm)
                    })
            })
    });
    let multiply = options.and_then(|fields| {
        fields.fields.iter().find_map(|(name, value)| {
            (OptimOptionField::parse_for_solver(name, OptimSolver::Fmincon)
                == Some(OptimOptionField::HessianMultiplyFcn))
            .then_some(value)
        })
    });
    let hessian = if may_reflective {
        if multiply.is_some() || options.is_none_or(|fields| !fields.fields_complete) {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        } else {
            facts::double(ShapeFact::Unknown, StorageFact::Unknown)
        }
    } else {
        numeric()
    };
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
    for name in ["constrviolation", "firstorderopt"] {
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
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
                information,
                !may_reflective,
            ))),
            ValueFact::unknown(DynamicReason::RuntimeValue),
            numeric(),
            hessian,
        ],
        diagnostics,
    );
    for index in [0, 8] {
        if let Some(ValueKindFact::Callable(callable)) = args.get(index).map(|value| &value.kind) {
            result
                .capabilities
                .0
                .extend(callable.capabilities.0.iter().copied());
        }
    }
    if let Some(ValueKindFact::Callable(callable)) = multiply.map(|value| &value.kind) {
        result
            .capabilities
            .0
            .extend(callable.capabilities.0.iter().copied());
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
