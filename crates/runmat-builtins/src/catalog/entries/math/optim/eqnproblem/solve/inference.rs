use super::super::super::program_facts as facts;
use crate::*;
use runmat_types::*;
pub(in crate::catalog::entries::math::optim) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 4)
    {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-CATALOG-EQN-SOLVE-OUTPUTS",
            "equation solve returns at most four outputs and does not return lambda",
            0,
        ));
    }
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
            "RM-CATALOG-EQN-SOLVE-ARITY",
            "solve expects an equation problem, optional x0 structure, and Solver/Options pairs",
            0,
        ));
    }
    let mut seen = [false; 2];
    for index in (start.unwrap_or(request.arguments.len())..request.arguments.len()).step_by(2) {
        let Some(name) = request.literals.literal_string_at(index) else {
            continue;
        };
        let Some(parameter) = OptimizationSolveParameter::parse(&name) else {
            diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-EQN-SOLVE-PARAMETER",
                "unsupported solve parameter",
                index,
            ));
            continue;
        };
        let slot = OptimizationSolveParameter::ALL
            .iter()
            .position(|&known| known == parameter)
            .expect("canonical solve parameter");
        if seen[slot] {
            diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-EQN-SOLVE-PARAMETER",
                "duplicate solve parameter",
                index,
            ));
        }
        seen[slot] = true;
        if parameter == OptimizationSolveParameter::Solver {
            if let Some(name) = request.literals.literal_string_at(index + 1) {
                if !matches!(
                    OptimSolver::parse(&name),
                    Some(
                        OptimSolver::Lsqlin
                            | OptimSolver::Fzero
                            | OptimSolver::Fsolve
                            | OptimSolver::Lsqnonlin
                    )
                ) {
                    diagnostics.push(crate::catalog::inference::argument_error(
                        "RM-CATALOG-EQN-SOLVE-SOLVER",
                        "equation solve supports lsqlin, fzero, fsolve and lsqnonlin",
                        index + 1,
                    ));
                }
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
    for name in ["Solver", "algorithm", "message"] {
        fields.insert(name.into(), ValueFact::scalar(ValueKindFact::String));
    }
    fields.insert(
        "wallclocktime".into(),
        facts::double(ShapeFact::Scalar, StorageFact::Scalar),
    );
    let outputs = vec![
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
            Default::default(),
            false,
        ))),
        ValueFact::unknown(DynamicReason::RuntimeValue),
        exitflag,
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, false))),
    ];
    let mut result = facts::finish(entry, request, outputs, diagnostics);
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
