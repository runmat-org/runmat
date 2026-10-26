use super::super::super::program_facts as facts;
use super::*;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    if matches!(request.arguments.first().map(|value| &value.kind), Some(ValueKindFact::Object(object))
        if object.runtime_class.as_ref().is_some_and(|class| class.is(crate::EQUATION_PROBLEM_CLASS)))
    {
        return super::super::super::eqnproblem::infer_solve(request, entry);
    }
    let mut diagnostics = Vec::new();
    let start = match request.arguments.get(1) {
        Some(ValueFact {
            kind: ValueKindFact::Struct(_),
            ..
        }) => Some(2),
        Some(ValueFact {
            kind: ValueKindFact::Unknown,
            ..
        }) if request.literals.literal_string_at(1).is_none() => None,
        _ => Some(1),
    };
    if request.arguments.is_empty()
        || start.is_some_and(|start| {
            !request
                .arguments
                .len()
                .saturating_sub(start)
                .is_multiple_of(2)
        })
    {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-CATALOG-SOLVE-ARITY","solve expects a problem, optional starting-point struct, then Solver and Options pairs",0));
    }
    let mut seen = [false; 2];
    let mut default_display = true;
    let mut selected = None;
    let mut callbacks_empty = true;
    for index in (start.unwrap_or(request.arguments.len())..request.arguments.len()).step_by(2) {
        let Some(name) = request.literals.literal_string_at(index) else {
            default_display = false;
            callbacks_empty = false;
            continue;
        };
        let Some(field) = OptimizationSolveParameter::parse(&name) else {
            diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-SOLVE-CONTROL",
                "unsupported solve parameter",
                index,
            ));
            continue;
        };
        let slot = OptimizationSolveParameter::ALL
            .iter()
            .position(|&known| known == field)
            .expect("canonical solve parameter");
        if seen[slot] {
            diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-SOLVE-CONTROL",
                "duplicate solve parameter",
                index,
            ));
        }
        seen[slot] = true;
        match field {
            OptimizationSolveParameter::Solver => {
                if let Some(solver) = request.literals.literal_string_at(index + 1) {
                    selected = OptimSolver::parse(&solver);
                    if !matches!(
                        selected,
                        Some(
                            OptimSolver::Linprog
                                | OptimSolver::Lsqlin
                                | OptimSolver::Quadprog
                                | OptimSolver::Fmincon
                                | OptimSolver::Fminunc
                                | OptimSolver::Lsqnonlin
                                | OptimSolver::Intlinprog
                        )
                    ) {
                        diagnostics.push(crate::catalog::inference::argument_error(
                            "RM-CATALOG-SOLVE-SOLVER",
                            "this model objective family supports linprog, lsqlin, quadprog, fmincon, fminunc, lsqnonlin and intlinprog",
                            index + 1,
                        ));
                    }
                }
            }
            OptimizationSolveParameter::Options => {
                callbacks_empty = request.arguments.get(index+1).is_some_and(|value| {
                    value.shape.element_count()==Some(0) || matches!(&value.kind,ValueKindFact::Struct(fields)
                        if fields.fields_complete && fields.fields.iter().all(|(name,value)|
                            !matches!(OptimOptionField::parse_for_solver(name,OptimSolver::Intlinprog),
                                Some(OptimOptionField::OutputFcn | OptimOptionField::PlotFcn))
                            || value.shape.element_count()==Some(0)))
                });
                default_display = match request.arguments.get(index + 1) {
                    Some(value) if value.shape.element_count() == Some(0) => true,
                    Some(ValueFact {
                        kind: ValueKindFact::Struct(fields),
                        ..
                    }) if fields.fields_complete => !fields.fields.keys().any(|name| {
                        OptimOptionField::parse(name) == Some(OptimOptionField::Display)
                    }),
                    _ => false,
                };
            }
        }
    }
    let exitflag = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::OPTIMIZATION_EXITFLAG.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let mut fields = std::collections::BTreeMap::new();
    if selected.is_some_and(|solver| solver != OptimSolver::Intlinprog) {
        fields.insert(
            "iterations".into(),
            facts::double(ShapeFact::Scalar, StorageFact::Scalar),
        );
        for name in ["constrviolation", "firstorderopt"] {
            if name == "constrviolation"
                && matches!(
                    selected,
                    None | Some(
                        OptimSolver::Lsqlin | OptimSolver::Fminunc | OptimSolver::Lsqnonlin
                    )
                )
            {
                continue;
            }
            fields.insert(
                name.into(),
                if selected.is_none() || selected == Some(OptimSolver::Fmincon) {
                    facts::double(ShapeFact::Unknown, StorageFact::Unknown)
                } else {
                    facts::double(ShapeFact::Scalar, StorageFact::Scalar)
                },
            );
        }
        if matches!(selected, Some(OptimSolver::Linprog | OptimSolver::Quadprog)) {
            fields.insert(
                "dualitygap".into(),
                facts::double(ShapeFact::Scalar, StorageFact::Scalar),
            );
        }
        if selected == Some(OptimSolver::Fminunc) {
            for name in ["funcCount", "stepsize"] {
                fields.insert(
                    name.into(),
                    facts::double(ShapeFact::Scalar, StorageFact::Scalar),
                );
            }
        }
        if selected == Some(OptimSolver::Fmincon) {
            for name in [
                "funcCount",
                "objectiveCallCount",
                "constraintCallCount",
                "stepsize",
            ] {
                fields.insert(
                    name.into(),
                    facts::double(ShapeFact::Scalar, StorageFact::Scalar),
                );
            }
        }
        if selected == Some(OptimSolver::Linprog) {
            for name in ["primalfeasibility", "dualfeasibility"] {
                fields.insert(
                    name.into(),
                    facts::double(ShapeFact::Scalar, StorageFact::Scalar),
                );
            }
        }
        if matches!(
            selected,
            Some(
                OptimSolver::Lsqlin
                    | OptimSolver::Quadprog
                    | OptimSolver::Fmincon
                    | OptimSolver::Lsqnonlin
            )
        ) {
            fields.insert(
                "wallclocktime".into(),
                facts::double(ShapeFact::Scalar, StorageFact::Scalar),
            );
        }
    }
    if selected == Some(OptimSolver::Lsqnonlin) {
        for name in ["funcCount", "stepsize", "cgiterations", "lssteplength"] {
            fields.insert(
                name.into(),
                facts::double(ShapeFact::Scalar, StorageFact::Scalar),
            );
        }
    }
    if selected == Some(OptimSolver::Intlinprog) {
        for name in ["relativegap", "absolutegap", "numfeaspoints", "numnodes"] {
            fields.insert(
                name.into(),
                facts::double(ShapeFact::Unknown, StorageFact::Unknown),
            );
        }
        for name in ["constrviolation", "lowerbound", "certifiedobjective"] {
            fields.insert(
                name.into(),
                facts::double(ShapeFact::Scalar, StorageFact::Scalar),
            );
        }
    }
    if matches!(
        selected,
        Some(OptimSolver::Fminunc | OptimSolver::Fmincon | OptimSolver::Lsqnonlin)
    ) {
        fields.insert(
            "objectivederivative".into(),
            ValueFact::scalar(ValueKindFact::String),
        );
    }
    for name in ["Solver", "message", "algorithm"] {
        fields.insert(name.into(), ValueFact::scalar(ValueKindFact::String));
    }
    let outputs = vec![
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
            Default::default(),
            false,
        ))),
        facts::double(ShapeFact::Unknown, StorageFact::Unknown),
        exitflag,
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
            fields,
            selected.is_some_and(|solver| solver != OptimSolver::Lsqlin),
        ))),
        if matches!(
            selected,
            Some(OptimSolver::Intlinprog | OptimSolver::Fminunc)
        ) {
            facts::double(ShapeFact::from(vec![Some(0), Some(0)]), StorageFact::Dense)
        } else if selected.is_some() {
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
                Default::default(),
                false,
            )))
        } else {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        },
    ];
    let mut result = facts::finish(entry, request, outputs, diagnostics);
    if callbacks_empty
        || matches!(
            selected,
            Some(
                OptimSolver::Linprog
                    | OptimSolver::Lsqlin
                    | OptimSolver::Quadprog
                    | OptimSolver::Lsqnonlin
            )
        )
    {
        result.effects.0.remove(&EffectKind::Unknown);
    }
    if default_display && selected == Some(OptimSolver::Linprog) {
        for effect in [
            EffectKind::FilesystemWrite,
            EffectKind::UserInterface,
            EffectKind::HostCallback,
        ] {
            result.effects.0.remove(&effect);
        }
    }
    if matches!(selected, Some(OptimSolver::Linprog | OptimSolver::Fminunc))
        || matches!(
            selected,
            Some(
                OptimSolver::Lsqlin
                    | OptimSolver::Quadprog
                    | OptimSolver::Fmincon
                    | OptimSolver::Lsqnonlin
            )
        ) && request
            .outputs
            .requested
            .known_count()
            .is_some_and(|count| count < 4)
    {
        result.effects.0.remove(&EffectKind::Clock);
    }
    result
}
