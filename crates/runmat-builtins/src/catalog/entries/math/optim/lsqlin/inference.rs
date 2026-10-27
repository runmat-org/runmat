use super::super::program_facts as facts;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut normalized = request.clone();
    let mut problem = false;
    let uncertain_warm = request.arguments.len() == 9 && request.arguments.get(8).is_some_and(|value| {
        matches!(&value.kind, ValueKindFact::Unknown)
            || matches!(&value.kind, ValueKindFact::Object(object) if object.runtime_class.is_none())
    });
    let warm = request
        .arguments
        .get(8)
        .filter(|_| request.arguments.len() == 9)
        .and_then(|value| match &value.kind {
            ValueKindFact::Object(object)
                if object.runtime_class.as_ref().is_some_and(|class| {
                    class.is(super::super::optimwarmstart::LSQLIN_WARM_START_CLASS)
                }) =>
            {
                Some(value.clone())
            }
            _ => None,
        });
    if let Some(value) = &warm {
        let ValueKindFact::Object(object) = &value.kind else {
            unreachable!()
        };
        normalized.arguments[8] = object
            .properties
            .get("X")
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
        normalized.arguments.push(
            object
                .properties
                .get("Options")
                .cloned()
                .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
        );
    } else if uncertain_warm {
        normalized.arguments[8] = ValueFact::unknown(DynamicReason::RuntimeValue);
        normalized
            .arguments
            .push(ValueFact::unknown(DynamicReason::RuntimeValue));
    }
    if request.arguments.len() == 1 {
        problem = true;
        match &request.arguments[0].kind {
            ValueKindFact::Struct(value) => {
                if request.arguments[0]
                    .shape
                    .element_count()
                    .is_some_and(|count| count != 1)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-LSQLIN-PROBLEM",
                        "problem must be a scalar struct",
                        0,
                    ));
                }
                let fields = [
                    "C", "d", "Aineq", "bineq", "Aeq", "beq", "lb", "ub", "x0", "options",
                ];
                for name in ["C", "d", "solver", "options"] {
                    if value.fields_complete && !value.fields.contains_key(name) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-LSQLIN-PROBLEM",
                            "problem requires C, d, solver and options",
                            0,
                        ));
                    }
                }
                normalized.arguments = fields
                    .into_iter()
                    .map(|name| {
                        value.fields.get(name).cloned().unwrap_or_else(|| {
                            if value.fields_complete && !matches!(name, "C" | "d") {
                                facts::double(
                                    ShapeFact::from(vec![Some(0), Some(0)]),
                                    StorageFact::Dense,
                                )
                            } else {
                                ValueFact::unknown(DynamicReason::RuntimeValue)
                            }
                        })
                    })
                    .collect();
            }
            ValueKindFact::Unknown => normalized.arguments.clear(),
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-LSQLIN-PROBLEM",
                    "single argument must be a scalar problem struct",
                    0,
                ));
                normalized.arguments.clear();
            }
        }
    } else if !matches!(request.arguments.len(), 4 | 6 | 8 | 9 | 10) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LSQLIN-ARITY",
            "expected C/d, complete constraint and bound pairs, optional x0 and options",
            0,
        ));
    }
    let args = &normalized.arguments;
    for (index, arg) in args.iter().take(9).enumerate() {
        let role = match index {
            0 => facts::OptimNumericInput::NonemptySparseMatrix,
            6..=8 => facts::OptimNumericInput::OptionalSparseArray,
            _ => facts::OptimNumericInput::OptionalSparseMatrix,
        };
        facts::numeric(&normalized, index, role, &mut diagnostics);
        if matches!(index, 1 | 3 | 5)
            && arg.shape.element_count() != Some(0)
            && arg.shape.known_dims().is_some_and(|dims| {
                dims.len() == 2
                    && dims[0].is_some_and(|n| n != 1)
                    && dims[1].is_some_and(|n| n != 1)
            })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-LSQLIN-VECTOR",
                "right-hand sides and x0 must be vectors",
                index,
            ));
        }
        if matches!(arg.kind, ValueKindFact::Logical) {
            diagnostics.push(argument_error(
                "RM-CATALOG-LSQLIN-REAL",
                "least-squares operands must be real numeric, not logical",
                index,
            ));
        }
    }
    let geometry = args.first().and_then(|c| c.shape.known_dims()).map(|dims| {
        (
            dims.first().copied().flatten(),
            dims.get(1)
                .copied()
                .flatten()
                .or(Some(1).filter(|_| dims.len() < 2)),
        )
    });
    let n = geometry.and_then(|(_, n)| n);
    if let Some((Some(m), _)) = geometry {
        if args
            .get(1)
            .and_then(|v| v.shape.element_count())
            .is_some_and(|d| d != m)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-LSQLIN-RHS",
                "d must have one element per C row",
                1,
            ));
        }
    }
    if let Some(n) = n {
        for index in [6, 7, 8] {
            if let Some(count) = args.get(index).and_then(|v| v.shape.element_count()) {
                if (index == 8 && count != 0 && count != n) || (matches!(index, 6 | 7) && count > n)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-LSQLIN-LENGTH",
                        "numeric dimensions must agree with C",
                        index,
                    ));
                }
            }
        }
        let mut geometry = normalized.clone();
        if geometry.arguments.len() > 1 {
            geometry.arguments[1].shape = ShapeFact::from(vec![Some(n), Some(1)]);
            facts::constraint_geometry(&geometry, 1, &[(2, 3), (4, 5)], &[], &mut diagnostics);
        }
    }
    super::super::nonlinear_facts::options(args.get(9), 9, true, &mut diagnostics);
    if problem {
        for diagnostic in &mut diagnostics {
            diagnostic.argument = Some(0);
        }
    }
    let numeric = || facts::double(ShapeFact::Unknown, StorageFact::Dense);
    let scalar = || facts::double(ShapeFact::Scalar, StorageFact::Scalar);
    let variable_method = algorithm_may_vary(args.get(9));
    let first = match warm {
        Some(mut value) => {
            if let ValueKindFact::Object(object) = &mut value.kind {
                object.properties.insert(
                    "X".into(),
                    facts::double(ShapeFact::from(vec![None, Some(1)]), StorageFact::Dense),
                );
            }
            value
        }
        None if uncertain_warm => ValueFact::unknown(DynamicReason::RuntimeValue),
        None => numeric(),
    };
    let mut fields = std::collections::BTreeMap::new();
    for name in [
        "iterations",
        "constrviolation",
        "firstorderopt",
        "dualitygap",
        "wallclocktime",
    ] {
        if variable_method && matches!(name, "constrviolation" | "dualitygap") {
            continue;
        }
        fields.insert(name.into(), scalar());
    }
    for name in ["algorithm", "linearsolver", "message"] {
        if variable_method && name == "linearsolver" {
            continue;
        }
        fields.insert(name.into(), ValueFact::scalar(ValueKindFact::String));
    }
    let mut result = facts::finish(
        entry,
        request,
        vec![
            first,
            numeric(),
            numeric(),
            scalar(),
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
                fields,
                !variable_method,
            ))),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        diagnostics,
    );
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count < 5)
    {
        result.effects.0.remove(&EffectKind::Clock);
    }
    result
}
fn algorithm_may_vary(options: Option<&ValueFact>) -> bool {
    match options.map(|value| &value.kind) {
        Some(ValueKindFact::Unknown) => true,
        Some(ValueKindFact::Struct(value)) => {
            !value.fields_complete
                || value.fields.iter().any(|(name, value)| {
                    crate::OptimOptionField::parse_for_solver(name, crate::OptimSolver::Lsqlin)
                        == Some(crate::OptimOptionField::Algorithm)
                        && value.shape.element_count() != Some(0)
                })
        }
        _ => false,
    }
}
