use super::super::{nonlinear_facts as facts, scalar_facts};
use crate::{
    catalog::inference::argument_error, BuiltinCatalogEntry, OptimOptionField, OptimSolver,
};
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn plot(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PATTERN-PLOT",
            "expected optimvalues,flag",
            0,
        ));
    }
    if let Some(metadata) = request.arguments.first() {
        if !matches!(
            metadata.kind,
            ValueKindFact::Struct(_) | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-PATTERN-PLOT-METADATA",
                "optimvalues must be a scalar structure",
                0,
            ));
        }
    }
    if request
        .literals
        .literal_string_at(1)
        .is_some_and(|value| crate::OptimObserverState::parse(&value).is_none())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-PATTERN-PLOT-PHASE",
            "flag must be init, iter or done",
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

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !matches!(request.arguments.len(), 1 | 2 | 4 | 6 | 8 | 9 | 10) {
        diagnostics.push(argument_error("RM-CATALOG-PATTERN-ARITY", "expected fun,x0 and complete constraint pairs, optional nonlcon/options, or one problem structure", 0));
    }
    let problem = if request.arguments.len() == 1 {
        match &request.arguments[0].kind {
            ValueKindFact::Struct(fields) => Some(fields),
            ValueKindFact::Unknown => None,
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PATTERN-PROBLEM",
                    "one input must be a scalar problem structure",
                    0,
                ));
                None
            }
        }
    } else {
        None
    };
    if let Some(problem) = problem {
        for name in [
            "objective",
            "x0",
            "Aineq",
            "bineq",
            "Aeq",
            "beq",
            "lb",
            "ub",
            "nonlcon",
            "options",
            "solver",
        ] {
            if problem.fields_complete && !problem.fields.contains_key(name) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PATTERN-FIELD",
                    "problem is missing a required public field",
                    0,
                ));
            }
        }
    }
    for (index, name, vector, empty) in [
        (1, "x0", true, false),
        (2, "Aineq", false, true),
        (3, "bineq", true, true),
        (4, "Aeq", false, true),
        (5, "beq", true, true),
        (6, "lb", true, true),
        (7, "ub", true, true),
    ] {
        let value = if request.arguments.len() == 1 {
            problem.and_then(|p| p.fields.get(name))
        } else {
            request.arguments.get(index)
        };
        if let Some(value) = value {
            let real = matches!(value.kind, ValueKindFact::Unknown | ValueKindFact::Logical)
                || matches!(value.kind, ValueKindFact::Numeric(n) if n.domain == NumericDomain::Real);
            let geometry = value.shape.known_dims().is_none_or(|dims| {
                (empty && value.shape.element_count() == Some(0))
                    || dims.len() <= 2
                        && (!vector
                            || dims.len() <= 1
                            || dims.iter().any(|dim| dim.is_none_or(|n| n == 1)))
            });
            if !real || !geometry || !empty && value.shape.element_count() == Some(0) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PATTERN-DATA",
                    "data must have real numeric matrix/vector geometry and x0 must be nonempty",
                    if request.arguments.len() == 1 {
                        0
                    } else {
                        index
                    },
                ));
            }
        }
    }
    let options = if request.arguments.len() == 1 {
        problem.and_then(|p| p.fields.get("options"))
    } else {
        request.arguments.get(9)
    };
    facts::options(options, 9, true, &mut diagnostics);
    if let Some(ValueFact {
        kind: ValueKindFact::Struct(options),
        ..
    }) = options
    {
        for name in options.fields.keys() {
            if name.eq_ignore_ascii_case("Solver") {
                continue;
            }
            if OptimOptionField::parse_for_solver(name, OptimSolver::Patternsearch).is_none() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PATTERN-OPTION",
                    "option is not a patternsearch control",
                    9,
                ));
            }
        }
    }
    let callback = if request.arguments.len() >= 2 {
        scalar_facts::callback(request, &mut diagnostics);
        let x0 = &request.arguments[1];
        scalar_facts::callback_invocation(
            request,
            facts::double(x0.shape.clone(), StorageFact::Dense),
        )
    } else {
        None
    };
    let mut fields = BTreeMap::new();
    for name in ["iterations", "funccount", "constraintcallcount", "meshsize"] {
        fields.insert(name.into(), facts::scalar());
    }
    for name in ["message", "problemtype", "pollmethod"] {
        fields.insert(name.into(), ValueFact::scalar(ValueKindFact::String));
    }
    fields.insert(
        "searchmethod".into(),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    );
    fields.insert(
        "maxconstraint".into(),
        facts::double(ShapeFact::Unknown, StorageFact::Dense),
    );
    fields.insert(
        "rngstate".into(),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    );
    let function = if request.arguments.len() == 1 {
        problem.and_then(|p| p.fields.get("objective"))
    } else {
        request.arguments.first()
    };
    fields.insert(
        "function".into(),
        function
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
    );
    facts::finish(
        request,
        entry,
        vec![
            facts::double(ShapeFact::Unknown, StorageFact::Dense),
            facts::double(ShapeFact::Unknown, StorageFact::Dense),
            facts::scalar(),
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true))),
        ],
        diagnostics,
        callback,
    )
}
