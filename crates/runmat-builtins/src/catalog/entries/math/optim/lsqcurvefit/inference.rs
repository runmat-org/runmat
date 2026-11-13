use super::super::{nonlinear_facts as facts, scalar_facts};
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if ![4, 5, 6, 7].contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LSQCURVEFIT-ARITY",
            "expected fun,x0,xdata,ydata and optional paired bounds and options",
            0,
        ));
    }
    scalar_facts::callback(request, &mut diagnostics);
    let initial = request.arguments.get(1);
    facts::numeric(initial, 1, false, &mut diagnostics);
    // xdata is the original callback payload, not a solver-domain conversion.
    let response = request.arguments.get(3);
    facts::numeric(response, 3, false, &mut diagnostics);
    let n = initial.and_then(|v| v.shape.element_count());
    if request.arguments.len() >= 6 {
        facts::bound(request.arguments.get(4), 4, n, false, &mut diagnostics);
        facts::bound(request.arguments.get(5), 5, n, false, &mut diagnostics);
    }
    let options_index = if request.arguments.len() == 5 { 4 } else { 6 };
    facts::options(
        request.arguments.get(options_index),
        options_index,
        false,
        &mut diagnostics,
    );
    let x = facts::like(initial);
    let xdata = request
        .arguments
        .get(2)
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let callback =
        scalar_facts::callback_invocation_with_arguments(request, vec![x.clone(), xdata]);
    let value = callback.as_ref().and_then(|c| c.outputs.first());
    facts::numeric(value, 0, false, &mut diagnostics);
    if value.zip(response).is_some_and(|(v, y)| {
        match (v.shape.known_dims(), y.shape.known_dims()) {
            (Some(a), Some(b)) => {
                a.len() != b.len()
                    || a.iter()
                        .zip(b.iter())
                        .any(|(a, b)| a.zip(*b).is_some_and(|(a, b)| a != b))
            }
            _ => false,
        }
    }) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LSQCURVEFIT-MODEL-SHAPE",
            "model result must match ydata shape",
            0,
        ));
    }
    let residual = facts::double(
        response
            .map(|v| v.shape.clone())
            .unwrap_or(ShapeFact::Unknown),
        StorageFact::Dense,
    );
    let m = response.and_then(|v| v.shape.element_count());
    let outputs = vec![
        x.clone(),
        facts::scalar(),
        residual,
        facts::scalar(),
        facts::diagnostics(&["iterations", "funcCount", "firstorderopt"]),
        facts::lambda(&x),
        facts::matrix(m, n),
    ];
    facts::finish(request, entry, outputs, diagnostics, callback)
}
