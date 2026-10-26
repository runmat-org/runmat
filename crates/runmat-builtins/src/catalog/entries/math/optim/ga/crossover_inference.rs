use super::super::nonlinear_facts as facts;
use crate::{
    catalog::inference::argument_error, BuiltinCatalogEntry, OptimGeneticCrossover as Kind,
};
use runmat_types::*;
fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry, kind: Kind) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 6
        && !(matches!(kind, Kind::Intermediate | Kind::Heuristic) && request.arguments.len() == 7)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-GA-CROSSOVER-ARITY",
            "expected six crossover inputs and only a supported optional ratio",
            0,
        ));
    }
    facts::options(request.arguments.get(1), 1, true, &mut diagnostics);
    for index in [0, 2, 4, 5, 6] {
        if index == 4 && kind != Kind::Heuristic {
            continue;
        }
        let Some(value) = request.arguments.get(index) else {
            continue;
        };
        if !matches!(
            value.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Logical
                | ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
        ) || value.storage == StorageFact::Sparse
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GA-CROSSOVER-NUMERIC",
                "active crossover data must be full and real",
                index,
            ));
        }
    }
    if let Some(value) = request.arguments.get(3) {
        if !matches!(
            value.kind,
            ValueKindFact::Unknown | ValueKindFact::Struct(_)
        ) || value.shape.element_count().is_some_and(|n| n != 1)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GA-CROSSOVER-PROBLEM",
                "problem must be a scalar structure",
                3,
            ));
        }
    }
    let columns = request.literals.numeric_at(2).and_then(|n| {
        if n.is_finite() && n > 0. && n.fract() == 0. && n < usize::MAX as f64 {
            Some(n as usize)
        } else {
            diagnostics.push(argument_error(
                "RM-CATALOG-GA-CROSSOVER-NVARS",
                "nvars must be positive, integral and structurally representable",
                2,
            ));
            None
        }
    });
    let rows = request
        .arguments
        .first()
        .and_then(|v| v.shape.element_count())
        .and_then(|n| {
            if n % 2 == 0 {
                Some(n / 2)
            } else {
                diagnostics.push(argument_error(
                    "RM-CATALOG-GA-CROSSOVER-PARENTS",
                    "parents require complete consecutive pairs",
                    0,
                ));
                None
            }
        });
    if let Some(value) = request.arguments.get(2) {
        if value.shape.element_count().is_some_and(|n| n != 1)
            || matches!(value.kind, ValueKindFact::Logical)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GA-CROSSOVER-NVARS",
                "nvars requires a numeric structural scalar",
                2,
            ));
        }
    }
    facts::finish(
        request,
        entry,
        vec![facts::matrix(rows, columns)],
        diagnostics,
        None,
    )
}
pub(super) fn scattered(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    infer(r, e, Kind::Scattered)
}
pub(super) fn single(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    infer(r, e, Kind::SinglePoint)
}
pub(super) fn two(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    infer(r, e, Kind::TwoPoint)
}
pub(super) fn intermediate(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    infer(r, e, Kind::Intermediate)
}
pub(super) fn arithmetic(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    infer(r, e, Kind::Arithmetic)
}
pub(super) fn heuristic(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    infer(r, e, Kind::Heuristic)
}
pub(super) fn laplace(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    infer(r, e, Kind::Laplace)
}
