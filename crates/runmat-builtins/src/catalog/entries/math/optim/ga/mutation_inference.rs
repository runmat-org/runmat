use super::super::nonlinear_facts as facts;
use crate::{
    catalog::inference::argument_error, BuiltinCatalogEntry, OptimGeneticMutation as Kind,
};
use runmat_types::*;
fn infer(r: &CallRequest, e: &BuiltinCatalogEntry, kind: Kind) -> CallInference {
    let mut diagnostics = Vec::new();
    let maximum = match kind {
        Kind::Gaussian => 9,
        Kind::Uniform => 8,
        _ => 7,
    };
    if !(7..=maximum).contains(&r.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-GA-MUTATION-ARITY",
            "expected seven mutation inputs and only supported numerical parameters",
            0,
        ));
    }
    facts::options(r.arguments.get(1), 1, true, &mut diagnostics);
    for index in [0, 2, 6, 7, 8] {
        if let Some(value) = r.arguments.get(index) {
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
                    "RM-CATALOG-GA-MUTATION-NUMERIC",
                    "active mutation data must be full real numeric data",
                    index,
                ));
            }
        }
    }
    for index in [3, 4] {
        if let Some(value) = r.arguments.get(index) {
            if !matches!(
                value.kind,
                ValueKindFact::Unknown | ValueKindFact::Struct(_)
            ) || value.shape.element_count().is_some_and(|n| n != 1)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-GA-MUTATION-CONTEXT",
                    "problem and state require scalar structures",
                    index,
                ));
            }
        }
    }
    let columns = r.literals.numeric_at(2).and_then(|n| {
        if n.is_finite() && n > 0. && n.fract() == 0. && n < usize::MAX as f64 {
            Some(n as usize)
        } else {
            diagnostics.push(argument_error(
                "RM-CATALOG-GA-MUTATION-NVARS",
                "nvars must be a positive representable integer",
                2,
            ));
            None
        }
    });
    if r.arguments.get(2).is_some_and(|v| {
        v.shape.element_count().is_some_and(|n| n != 1) || matches!(v.kind, ValueKindFact::Logical)
    }) {
        diagnostics.push(argument_error(
            "RM-CATALOG-GA-MUTATION-NVARS",
            "nvars must be a numeric structural scalar",
            2,
        ));
    }
    let rows = r.arguments.first().and_then(|v| v.shape.element_count());
    facts::finish(r, e, vec![facts::matrix(rows, columns)], diagnostics, None)
}
pub(super) fn gaussian(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    infer(r, e, Kind::Gaussian)
}
pub(super) fn uniform(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    infer(r, e, Kind::Uniform)
}
pub(super) fn power(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    infer(r, e, Kind::Power)
}
pub(super) fn adaptive(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    infer(r, e, Kind::AdaptiveFeasible)
}
pub(super) fn positive(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    infer(r, e, Kind::PositiveBasis)
}
