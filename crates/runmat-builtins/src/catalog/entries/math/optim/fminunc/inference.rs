use super::super::{nonlinear_facts as facts, scalar_facts};
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FMINUNC-ARITY",
            "expected a callback, initial guess and optional options",
            0,
        ));
    }
    scalar_facts::callback(request, &mut diagnostics);
    let initial = request.arguments.get(1);
    facts::numeric(initial, 1, false, &mut diagnostics);
    facts::options(request.arguments.get(2), 2, false, &mut diagnostics);
    if let Some(ValueKindFact::Struct(options)) = request.arguments.get(2).map(|value| &value.kind)
    {
        for name in options.fields.keys() {
            if crate::OptimOptionField::parse(name).is_some()
                && crate::OptimOptionField::parse_for_solver(name, crate::OptimSolver::Fminunc)
                    .is_none()
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FMINUNC-OPTION",
                    "option is not recognized for fminunc",
                    2,
                ));
            }
        }
    }
    let x = facts::like(initial);
    let n = initial.and_then(|v| v.shape.element_count());
    let callback = scalar_facts::callback_invocation_with_arguments(request, vec![x.clone()]);
    let value = callback.as_ref().and_then(|c| c.outputs.first());
    facts::numeric(value, 0, false, &mut diagnostics);
    if value
        .and_then(|v| v.shape.element_count())
        .is_some_and(|n| n != 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-FMINUNC-OBJECTIVE",
            "objective must return a scalar",
            0,
        ));
    }
    // Struct field facts do not retain the Algorithm string's literal value.
    // A selected/unknown TR mode can return [], a sparse/dense last model, or
    // arbitrary original Hinfo. Do not forge an n-by-n numeric sixth output.
    let known_qn = match request.arguments.get(2).map(|value| &value.kind) {
        None => true,
        Some(ValueKindFact::Struct(options)) => {
            options.fields_complete
                && !options.fields.keys().any(|name| {
                    crate::OptimOptionField::parse(name) == Some(crate::OptimOptionField::Algorithm)
                })
        }
        _ => false,
    };
    let update_may_be_limited_memory = match request.arguments.get(2).map(|value| &value.kind) {
        Some(ValueKindFact::Struct(options)) => options.fields.keys().any(|name| {
            matches!(
                crate::OptimOptionField::parse(name),
                Some(
                    crate::OptimOptionField::HessianApproximation
                        | crate::OptimOptionField::HessUpdate
                )
            )
        }),
        _ => false,
    };
    let hessian = if !known_qn {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    } else if update_may_be_limited_memory {
        // The update field has no literal value here. L-BFGS returns [],
        // while the other QN updates return a dense Hessian estimate.
        facts::double(ShapeFact::Unknown, StorageFact::Dense)
    } else {
        facts::matrix(n, n)
    };
    let outputs = vec![
        x.clone(),
        facts::scalar(),
        facts::scalar(),
        facts::diagnostics(&["iterations", "funcCount", "firstorderopt", "stepsize"]),
        x,
        hessian,
    ];
    facts::finish(request, entry, outputs, diagnostics, callback)
}
