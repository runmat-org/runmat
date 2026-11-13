use super::super::{nonlinear_facts as facts, scalar_facts};
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
use std::collections::BTreeMap;
pub(super) fn solver(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !matches!(request.arguments.len(), 1 | 2 | 4 | 5) {
        diagnostics.push(argument_error(
            "RM-CATALOG-PS-ARITY",
            "expected fun,nvars; optional complete bounds/options; or a problem structure",
            0,
        ));
    }
    let problem = if request.arguments.len() == 1 {
        match &request.arguments[0].kind {
            ValueKindFact::Struct(fields) => Some(fields),
            ValueKindFact::Unknown => None,
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PS-PROBLEM",
                    "single input requires a scalar problem structure",
                    0,
                ));
                None
            }
        }
    } else {
        None
    };
    if let Some(problem) = problem {
        for name in ["objective", "nvars", "lb", "ub", "solver", "options"] {
            if problem.fields_complete && !problem.fields.contains_key(name) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PS-FIELD",
                    "problem is missing a required public field",
                    0,
                ));
            }
        }
    }
    let nvars = problem
        .and_then(|p| p.fields.get("nvars"))
        .or_else(|| request.arguments.get(1));
    if let Some(value) = nvars {
        if !double(value) || value.shape.element_count().is_some_and(|n| n != 1) {
            diagnostics.push(argument_error(
                "RM-CATALOG-PS-NVARS",
                "nvars must be a positive integer-valued double scalar",
                1,
            ));
        }
    }
    let columns = if problem.is_none() {
        request.literals.numeric_at(1).and_then(|n| {
            (n.is_finite() && n > 0.0 && n.fract() == 0.0 && n < usize::MAX as f64)
                .then_some(n as usize)
        })
    } else {
        None
    };
    if problem.is_none() && request.literals.numeric_at(1).is_some() && columns.is_none() {
        diagnostics.push(argument_error(
            "RM-CATALOG-PS-NVARS",
            "nvars must be positive, integral and structurally representable",
            1,
        ));
    }
    for (index, name) in [(2, "lb"), (3, "ub")] {
        if let Some(value) = problem
            .and_then(|p| p.fields.get(name))
            .or_else(|| request.arguments.get(index))
        {
            if !double(value) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PS-BOUND",
                    "bounds must be full real double arrays",
                    index,
                ));
            }
        }
    }
    let controls = problem
        .and_then(|p| p.fields.get("options"))
        .or_else(|| request.arguments.get(4));
    facts::options(controls, 4, true, &mut diagnostics);
    if let Some(ValueFact {
        kind: ValueKindFact::Struct(options),
        ..
    }) = controls
    {
        for (name, value) in &options.fields {
            if name.eq_ignore_ascii_case("Solver") {
                continue;
            }
            if crate::OptimOptionField::parse_for_solver(name, crate::OptimSolver::Particleswarm)
                .is_none_or(|field| !crate::OptimSolver::Particleswarm.accepts(field))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PS-OPTION",
                    "unsupported particle-swarm control",
                    4,
                ));
            }
            let _ = value;
        }
    }
    let unbounded = if let Some(problem) = problem {
        ["lb", "ub"].iter().all(|name| {
            problem
                .fields
                .get(*name)
                .is_some_and(|v| v.shape.element_count() == Some(0))
        })
    } else {
        request.arguments.len() == 2
            || [2, 3].iter().all(|&index| {
                request
                    .arguments
                    .get(index)
                    .is_some_and(|v| v.shape.element_count() == Some(0))
            })
    };
    let row = if unbounded {
        facts::matrix(Some(1), columns)
    } else {
        facts::double(ShapeFact::Unknown, StorageFact::Dense)
    };
    let measured_scalar = if unbounded {
        facts::scalar()
    } else {
        facts::double(ShapeFact::Unknown, StorageFact::Unknown)
    };
    let callback = if request.arguments.len() > 1 {
        scalar_facts::callback(request, &mut diagnostics);
        scalar_facts::callback_invocation_with_arguments(
            request,
            vec![facts::matrix(None, columns)],
        )
    } else {
        None
    };
    let mut summary = BTreeMap::new();
    for name in ["iterations", "funccount", "swarmObjectiveCallCount"] {
        summary.insert(name.into(), facts::scalar());
    }
    summary.insert(
        "hybridflag".into(),
        facts::double(ShapeFact::Unknown, StorageFact::Unknown),
    );
    summary.insert("message".into(), ValueFact::scalar(ValueKindFact::String));
    summary.insert(
        "rngstate".into(),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    );
    let output = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(summary, true)));
    let points = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        [
            (
                "X".into(),
                if unbounded {
                    facts::matrix(None, columns)
                } else {
                    facts::double(ShapeFact::Unknown, StorageFact::Dense)
                },
            ),
            (
                "Fval".into(),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ),
        ]
        .into_iter()
        .collect(),
        true,
    )));
    facts::finish(
        request,
        entry,
        vec![row, measured_scalar, facts::scalar(), output, points],
        diagnostics,
        callback,
    )
}
pub(super) fn creation(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1
        || request
            .arguments
            .first()
            .is_some_and(|v| !matches!(v.kind, ValueKindFact::Struct(_) | ValueKindFact::Unknown))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-PS-CREATION",
            "expected one scalar particleswarm problem structure",
            0,
        ));
    }
    facts::finish(
        request,
        entry,
        vec![facts::matrix(None, None)],
        diagnostics,
        None,
    )
}
pub(super) fn plot(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PS-PLOT",
            "expected optimValues,state",
            0,
        ));
    }
    if let Some(value) = request.arguments.first() {
        if !matches!(
            value.kind,
            ValueKindFact::Struct(_) | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-PS-PLOT",
                "optimValues must be a scalar structure",
                0,
            ));
        }
        if let ValueKindFact::Struct(values) = &value.kind {
            for name in ["iteration", "bestfval"] {
                if let Some(value) = values.fields.get(name) {
                    if !double(value) || value.shape.element_count().is_some_and(|n| n != 1) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-PS-PLOT-METRIC",
                            "selected plot metrics must be real double scalars",
                            0,
                        ));
                    }
                } else if values.fields_complete {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-PS-PLOT-METRIC",
                        "optimValues requires iteration and bestfval",
                        0,
                    ));
                }
            }
        }
    }
    if request
        .literals
        .literal_string_at(1)
        .is_some_and(|v| crate::OptimObserverState::parse(&v).is_none())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-PS-PLOT-STATE",
            "state must be init, iter or done",
            1,
        ));
    }
    facts::finish(
        request,
        entry,
        vec![ValueFact::scalar(ValueKindFact::Logical)],
        diagnostics,
        None,
    )
}
fn double(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Unknown)
        || matches!(value.kind,ValueKindFact::Numeric(n)if n.class==NumericClass::Double&&n.domain==NumericDomain::Real)
            && value.storage != StorageFact::Sparse
}
