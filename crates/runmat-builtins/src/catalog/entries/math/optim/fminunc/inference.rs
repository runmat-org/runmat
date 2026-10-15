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
    let hessian = if x.storage == StorageFact::Scalar {
        facts::scalar()
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
