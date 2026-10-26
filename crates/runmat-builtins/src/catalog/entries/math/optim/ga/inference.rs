//! Facts describe representations, not feasibility, stochastic convergence,
//! callback purity, compatibility admission or a finite allocation grant.
use super::super::{nonlinear_facts as facts, scalar_facts};
use crate::{
    catalog::inference::argument_error, BuiltinCatalogEntry, OptimOptionField, OptimSolver,
};
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn solver(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !matches!(request.arguments.len(), 1 | 2 | 4 | 6 | 8 | 9 | 10 | 11) {
        diagnostics.push(argument_error(
            "RM-CATALOG-GA-ARITY",
            "expected a ga problem structure or complete positional argument pairs",
            0,
        ));
    }
    let problem = if request.arguments.len() == 1 {
        match &request.arguments[0].kind {
            ValueKindFact::Struct(problem) => Some(problem),
            ValueKindFact::Unknown => None,
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-GA-PROBLEM",
                    "single input requires a scalar ga problem structure",
                    0,
                ));
                None
            }
        }
    } else {
        None
    };
    if let Some(problem) = problem {
        for required in ["fitnessfcn", "nvars", "options"] {
            if problem.fields_complete && !problem.fields.contains_key(required) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-GA-FIELD",
                    "problem is missing a required public field",
                    0,
                ));
            }
        }
    }
    let role = |name: &str, index: usize| {
        problem
            .and_then(|problem| problem.fields.get(name))
            .or_else(|| {
                (request.arguments.len() != 1)
                    .then(|| request.arguments.get(index))
                    .flatten()
            })
    };
    let nvars = role("nvars", 1);
    if let Some(nvars) = nvars {
        real(nvars, 1, &mut diagnostics);
        if matches!(nvars.kind, ValueKindFact::Logical)
            || nvars.storage == StorageFact::Sparse
            || nvars.shape.element_count().is_some_and(|count| count != 1)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GA-NVARS",
                "nvars requires one positive exact structural integer",
                1,
            ));
        }
    }
    if request.arguments.len() != 1
        && request.literals.numeric_at(1).is_some_and(|n| {
            !n.is_finite() || n <= 0. || n.fract() != 0. || (n as u128) > usize::MAX as u128
        })
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-GA-NVARS",
            "nvars must be positive, integral and structurally representable",
            1,
        ));
    }
    let controls = problem
        .and_then(|problem| problem.fields.get("options"))
        .or_else(|| match request.arguments.len() {
            11 => request.arguments.get(10),
            10 => request
                .arguments
                .get(9)
                .filter(|value| matches!(value.kind, ValueKindFact::Struct(_))),
            _ => None,
        });
    facts::options(
        controls,
        if request.arguments.len() == 11 { 10 } else { 9 },
        true,
        &mut diagnostics,
    );
    let custom_possible = match controls {
        Some(ValueFact {
            kind: ValueKindFact::Struct(options),
            ..
        }) => {
            for name in options.fields.keys() {
                if name.eq_ignore_ascii_case("Solver") {
                    continue;
                }
                if OptimOptionField::parse_for_solver(name, OptimSolver::Ga)
                    .is_none_or(|field| !OptimSolver::Ga.accepts(field))
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-GA-OPTION",
                        "unsupported genetic option",
                        9,
                    ));
                }
            }
            !options.fields_complete
                || options
                    .fields
                    .keys()
                    .any(|name| name.eq_ignore_ascii_case("PopulationType"))
        }
        None => request.arguments.len() == 1 || request.arguments.len() == 10,
        _ => true,
    };
    // The selector can deactivate constraints for custom/bitstring genomes.
    // Unknown selector values do not justify a static numeric-role rejection.
    if !custom_possible {
        for (name, index) in [
            ("Aineq", 2),
            ("Bineq", 3),
            ("Aeq", 4),
            ("Beq", 5),
            ("lb", 6),
            ("ub", 7),
        ] {
            if let Some(value) = role(name, index) {
                real(value, index, &mut diagnostics);
            }
        }
    }
    let callback = if request.arguments.len() != 1 {
        scalar_facts::callback(request, &mut diagnostics);
        scalar_facts::callback_invocation_with_arguments(
            request,
            vec![if custom_possible {
                ValueFact::unknown(DynamicReason::RuntimeValue)
            } else {
                facts::matrix(None, None)
            }],
        )
    } else {
        None
    };
    let mut fields = BTreeMap::new();
    for name in [
        "generations",
        "funccount",
        "objectiveCallCount",
        "nonlinearPointCount",
        "nonlinearCallCount",
        "innergenerations",
        "linearnodes",
        "hybridfunccount",
    ] {
        fields.insert(name.into(), facts::scalar());
    }
    fields.insert("totaltime".into(), facts::scalar());
    for name in ["maxconstraint", "hybridflag"] {
        fields.insert(
            name.into(),
            facts::double(ShapeFact::Unknown, StorageFact::Unknown),
        );
    }
    fields.insert(
        "hybridaccepted".into(),
        ValueFact::scalar(ValueKindFact::Logical),
    );
    for name in ["message", "problemtype"] {
        fields.insert(name.into(), ValueFact::scalar(ValueKindFact::String));
    }
    for name in ["rngstate", "fitnessprovenance"] {
        fields.insert(name.into(), ValueFact::unknown(DynamicReason::RuntimeValue));
    }
    let point = if custom_possible {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    } else {
        facts::double(ShapeFact::Unknown, StorageFact::Dense)
    };
    let population = point.clone();
    let outputs = vec![
        point,
        facts::double(ShapeFact::Unknown, StorageFact::Unknown),
        facts::scalar(),
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true))),
        population,
        ValueFact::unknown(DynamicReason::RuntimeValue),
    ];
    facts::finish(request, entry, outputs, diagnostics, callback)
}
fn real(value: &ValueFact, index: usize, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if !matches!(value.kind, ValueKindFact::Unknown | ValueKindFact::Logical)
        && !matches!(value.kind, ValueKindFact::Numeric(n) if n.domain == NumericDomain::Real)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-GA-REAL",
            "active solver roles require real numeric data",
            index,
        ));
    }
}
