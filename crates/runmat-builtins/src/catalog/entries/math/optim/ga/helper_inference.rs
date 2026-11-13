use super::super::nonlinear_facts as facts;
use crate::{
    catalog::inference::argument_error, BuiltinCatalogEntry, OptimGeneticScaling as Scaling,
    OptimGeneticSelection as Selection,
};
use runmat_types::*;
fn count(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) -> Option<usize> {
    if let Some(value) = request.arguments.get(1) {
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
                "RM-CATALOG-GA-HELPER-COUNT",
                "nParents requires a real numeric structural scalar",
                1,
            ));
        }
    }
    request.literals.numeric_at(1).and_then(|n| {
        if n.is_finite() && n >= 0. && n.fract() == 0. && (n as u128) <= usize::MAX as u128 {
            Some(n as usize)
        } else {
            diagnostics.push(argument_error(
                "RM-CATALOG-GA-HELPER-COUNT",
                "nParents must be nonnegative, integral and structurally representable",
                1,
            ));
            None
        }
    })
}
fn vector(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if let Some(value) = request.arguments.first() {
        if !matches!(
            value.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Logical
                | ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
        ) || value.storage == StorageFact::Sparse
            || value.shape.known_dims().is_some_and(|dims| {
                value.shape.element_count() != Some(0)
                    && (dims.len() > 2
                        || dims
                            .iter()
                            .filter(|axis| axis.is_some_and(|n| n != 1))
                            .count()
                            > 1)
            })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GA-HELPER-VECTOR",
                "the helper requires a full real vector",
                0,
            ));
        }
    }
}
fn selection(request: &CallRequest, entry: &BuiltinCatalogEntry, kind: Selection) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 3
        && !(kind == Selection::Tournament && request.arguments.len() == 4)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-GA-SELECTION-ARITY",
            "expected expectation,nParents,options and optional tournament size",
            0,
        ));
    }
    vector(request, &mut diagnostics);
    let parents = count(request, &mut diagnostics);
    facts::options(request.arguments.get(2), 2, true, &mut diagnostics);
    if kind == Selection::Tournament
        && request.literals.numeric_at(3).is_some_and(|n| {
            !n.is_finite() || n < 2. || n.fract() != 0. || (n as u128) > usize::MAX as u128
        })
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-GA-TOURNAMENT-SIZE",
            "size must be an exact integer >=2",
            3,
        ));
    }
    facts::finish(
        request,
        entry,
        vec![facts::matrix(Some(1), parents)],
        diagnostics,
        None,
    )
}
fn scaling(request: &CallRequest, entry: &BuiltinCatalogEntry, kind: Scaling) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2
        && !(matches!(kind, Scaling::Top | Scaling::ShiftLinear) && request.arguments.len() == 3)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-GA-SCALING-ARITY",
            "expected scores,nParents and the selected optional parameter",
            0,
        ));
    }
    vector(request, &mut diagnostics);
    count(request, &mut diagnostics);
    facts::finish(
        request,
        entry,
        vec![facts::matrix(
            request
                .arguments
                .first()
                .and_then(|v| v.shape.element_count()),
            Some(1),
        )],
        diagnostics,
        None,
    )
}
pub(super) fn stochastic(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    selection(r, e, Selection::StochasticUniform)
}
pub(super) fn remainder(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    selection(r, e, Selection::Remainder)
}
pub(super) fn uniform(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    selection(r, e, Selection::Uniform)
}
pub(super) fn roulette(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    selection(r, e, Selection::Roulette)
}
pub(super) fn tournament(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    selection(r, e, Selection::Tournament)
}
pub(super) fn rank(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    scaling(r, e, Scaling::Rank)
}
pub(super) fn proportional(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    scaling(r, e, Scaling::Proportional)
}
pub(super) fn top(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    scaling(r, e, Scaling::Top)
}
pub(super) fn shift_linear(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    scaling(r, e, Scaling::ShiftLinear)
}
