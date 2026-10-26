use super::super::program_facts as facts;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut normalized = request.clone();
    let mut problem = false;
    let warm = request.arguments.get(8).filter(|_| request.arguments.len() == 9)
        .filter(|value| matches!(&value.kind, ValueKindFact::Object(object)
            if object.runtime_class.as_ref().is_some_and(|class| class.is(super::super::optimwarmstart::QUADPROG_WARM_START_CLASS))))
        .cloned();
    let uncertain_warm = request.arguments.len() == 9 && request.arguments.get(8).is_some_and(|value| {
        matches!(&value.kind, ValueKindFact::Unknown)
            || matches!(&value.kind, ValueKindFact::Object(object) if object.runtime_class.is_none())
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
                        "RM-CATALOG-QP-PROBLEM",
                        "problem must be a scalar struct",
                        0,
                    ));
                }
                let fields = [
                    "H", "f", "Aineq", "bineq", "Aeq", "beq", "lb", "ub", "x0", "options",
                ];
                for name in ["H", "f", "solver", "options"] {
                    if value.fields_complete && !value.fields.contains_key(name) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-QP-PROBLEM",
                            "problem requires H, f, solver and options",
                            0,
                        ));
                    }
                }
                normalized.arguments = fields
                    .into_iter()
                    .map(|name| {
                        value.fields.get(name).cloned().unwrap_or_else(|| {
                            if value.fields_complete && !matches!(name, "H" | "f") {
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
                    "RM-CATALOG-QP-PROBLEM",
                    "single argument must be a scalar problem struct",
                    0,
                ));
                normalized.arguments.clear();
            }
        }
    } else if !matches!(request.arguments.len(), 2 | 4 | 6 | 8 | 9 | 10) {
        diagnostics.push(argument_error(
            "RM-CATALOG-QP-ARITY",
            "expected H/f, complete constraint and bound pairs, optional x0 and options",
            0,
        ));
    }
    let args = &normalized.arguments;
    for (index, arg) in args.iter().take(9).enumerate() {
        let role = match index {
            0 => facts::OptimNumericInput::NonemptySparseMatrix,
            1 | 2 | 4 | 6 | 7 => facts::OptimNumericInput::OptionalSparseMatrix,
            _ => facts::OptimNumericInput::OptionalVector,
        };
        facts::numeric(&normalized, index, role, &mut diagnostics);
        if matches!(arg.kind, ValueKindFact::Logical) {
            diagnostics.push(argument_error(
                "RM-CATALOG-QP-REAL",
                "QP operands must be real numeric, not logical",
                index,
            ));
        }
    }
    let n = args
        .first()
        .and_then(|h| h.shape.known_dims())
        .and_then(|dims| {
            let rows = dims.first().copied().flatten();
            let cols = dims
                .get(1)
                .copied()
                .flatten()
                .or(Some(1).filter(|_| dims.len() < 2));
            if rows.zip(cols).is_some_and(|(r, c)| r != c) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-QP-SQUARE",
                    "H must be square",
                    0,
                ));
            }
            rows.filter(|r| Some(*r) == cols)
        });
    if let Some(n) = n {
        for index in [1, 6, 7, 8] {
            if let Some(count) = args.get(index).and_then(|v| v.shape.element_count()) {
                if (matches!(index, 1 | 8) && count != 0 && count != n)
                    || (matches!(index, 6 | 7) && count > n)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-QP-LENGTH",
                        "numeric dimensions must agree with H",
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
    let first = match warm {
        Some(mut fact) => {
            if let ValueKindFact::Object(object) = &mut fact.kind {
                object.properties.insert(
                    "X".into(),
                    facts::double(ShapeFact::from(vec![None, Some(1)]), StorageFact::Dense),
                );
            }
            fact
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
        fields.insert(name.into(), scalar());
    }
    for name in ["algorithm", "linearsolver", "message"] {
        fields.insert(name.into(), ValueFact::scalar(ValueKindFact::String));
    }
    let mut result = facts::finish(
        entry,
        request,
        vec![
            first,
            numeric(),
            scalar(),
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true))),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        diagnostics,
    );
    let timed_option = args.get(9).is_some_and(|value| match &value.kind {
        ValueKindFact::Struct(fields) => {
            !fields.fields_complete
                || fields
                    .fields
                    .keys()
                    .any(|name| name.eq_ignore_ascii_case("MaxTime"))
        }
        ValueKindFact::Numeric(_) if value.shape.element_count() == Some(0) => false,
        _ => true,
    });
    if !timed_option
        && request
            .outputs
            .requested
            .known_count()
            .is_some_and(|count| count < 4)
    {
        result.effects.0.remove(&EffectKind::Clock);
    }
    result
}
