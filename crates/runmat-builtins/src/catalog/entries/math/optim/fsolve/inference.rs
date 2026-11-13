use super::super::{nonlinear_facts as facts, scalar_facts};
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FSOLVE-ARITY",
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
    let residual = facts::like(value);
    let m = value.and_then(|v| v.shape.element_count());
    let outputs = vec![
        x,
        residual,
        facts::scalar(),
        facts::diagnostics(&["iterations", "funcCount", "firstorderopt", "stepsize"]),
        facts::matrix(m, n),
    ];
    facts::finish(request, entry, outputs, diagnostics, callback)
}
