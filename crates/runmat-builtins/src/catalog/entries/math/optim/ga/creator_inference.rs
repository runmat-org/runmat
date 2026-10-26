use super::super::nonlinear_facts as facts;
use crate::{
    catalog::inference::argument_error, BuiltinCatalogEntry, OptimGeneticCreation as Kind,
};
use runmat_types::*;
fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry, kind: Kind) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 3
        && !(kind == Kind::NonlinearFeasible
            && request.arguments.len() >= 3
            && request.arguments.len() <= 9
            && request.arguments.len() % 2 == 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-GA-CREATION-ARITY",
            "expected nvars,problem,options and only nonlinear creation name/value pairs",
            0,
        ));
    }
    if let Some(value) = request.arguments.first() {
        if !matches!(
            value.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
        ) || value.storage == StorageFact::Sparse
            || value.shape.element_count().is_some_and(|n| n != 1)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GA-CREATION-NVARS",
                "genome length must be a positive numeric structural scalar",
                0,
            ));
        }
    }
    let columns = request.literals.numeric_at(0).and_then(|n| {
        if n.is_finite() && n > 0. && n.fract() == 0. && (n as u128) <= usize::MAX as u128 {
            Some(n as usize)
        } else {
            diagnostics.push(argument_error(
                "RM-CATALOG-GA-CREATION-NVARS",
                "genome length must be positive, integral and structurally representable",
                0,
            ));
            None
        }
    });
    if let Some(value) = request.arguments.get(1) {
        if !matches!(
            value.kind,
            ValueKindFact::Unknown | ValueKindFact::Struct(_)
        ) || value.shape.element_count().is_some_and(|n| n != 1)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GA-CREATION-PROBLEM",
                "creation requires a scalar operator problem context",
                1,
            ));
        }
        if let ValueKindFact::Struct(problem) = &value.kind {
            if problem.fields_complete
                && !problem.fields.contains_key("FitnessFcn")
                && !problem.fields.contains_key("fitnessfcn")
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-GA-CREATION-FITNESS",
                    "operator problem requires FitnessFcn",
                    1,
                ));
            }
        }
    }
    facts::options(request.arguments.get(2), 2, true, &mut diagnostics);
    facts::finish(
        request,
        entry,
        vec![facts::matrix(None, columns)],
        diagnostics,
        None,
    )
}
pub(super) fn uniform(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    infer(r, e, Kind::Uniform)
}
pub(super) fn linear(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    infer(r, e, Kind::LinearFeasible)
}
pub(super) fn nonlinear(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    infer(r, e, Kind::NonlinearFeasible)
}
pub(super) fn integer(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    infer(r, e, Kind::UniformInteger)
}
pub(super) fn sobol(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    infer(r, e, Kind::Sobol)
}
