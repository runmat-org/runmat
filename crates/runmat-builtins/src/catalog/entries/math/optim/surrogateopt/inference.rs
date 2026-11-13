use super::super::{nonlinear_facts as facts, scalar_facts};
use crate::{
    catalog::inference::argument_error, BuiltinCatalogEntry, OptimOptionField as Field, OptimSolver,
};
use runmat_types::*;
use std::collections::BTreeMap;
fn double(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
    )
}
pub(super) fn solver(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let arity = request.arguments.len();
    if !matches!(arity, 1 | 2 | 3 | 4 | 5 | 8 | 9) {
        diagnostics.push(argument_error("RM-CATALOG-SURROGATE-ARITY", "expected bounds with optional integer/linear/options inputs, a problem, or a checkpoint path/options", 0));
    }
    let problem = if arity == 1 {
        match &request.arguments[0].kind {
            ValueKindFact::Struct(fields) => Some(fields),
            _ => None,
        }
    } else {
        None
    };
    let checkpoint = arity <= 2 && problem.is_none();
    if let Some(problem) = problem {
        for name in ["objective", "lb", "ub", "solver", "options"] {
            if problem.fields_complete && !problem.fields.contains_key(name) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SURROGATE-PROBLEM",
                    "problem is missing a required field",
                    0,
                ));
            }
        }
        for name in problem.fields.keys() {
            if !matches!(
                name.as_str(),
                "objective"
                    | "lb"
                    | "ub"
                    | "intcon"
                    | "Aineq"
                    | "bineq"
                    | "Aeq"
                    | "beq"
                    | "solver"
                    | "options"
                    | "rngstate"
            ) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SURROGATE-PROBLEM",
                    "unsupported problem field",
                    0,
                ));
            }
        }
    }
    let fourth_options =
        arity == 4 && matches!(request.arguments[3].kind, ValueKindFact::Struct(_));
    let controls = if let Some(problem) = problem {
        problem.fields.get("options")
    } else if checkpoint {
        request.arguments.get(1)
    } else {
        match arity {
            4 if fourth_options => request.arguments.get(3),
            5 => request.arguments.get(4),
            9 => request.arguments.get(8),
            _ => None,
        }
    };
    facts::options(controls, arity.saturating_sub(1), true, &mut diagnostics);
    if let Some(ValueFact {
        kind: ValueKindFact::Struct(options),
        ..
    }) = controls
    {
        for name in options.fields.keys() {
            if !name.eq_ignore_ascii_case("Solver")
                && Field::parse_for_solver(name, OptimSolver::Surrogateopt)
                    .is_none_or(|field| !OptimSolver::Surrogateopt.accepts(field))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SURROGATE-OPTION",
                    "unsupported surrogate option",
                    arity.saturating_sub(1),
                ));
            }
        }
    }
    let mut columns = None;
    if !checkpoint {
        let lower = problem
            .and_then(|fields| fields.fields.get("lb"))
            .or_else(|| request.arguments.get(1));
        let upper = problem
            .and_then(|fields| fields.fields.get("ub"))
            .or_else(|| request.arguments.get(2));
        columns = lower.and_then(|value| value.shape.element_count());
        for (index, value) in [(1, lower), (2, upper)] {
            if let Some(value) = value {
                if !double(value) || value.shape.element_count() == Some(0) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-SURROGATE-BOUND",
                        "bounds must be nonempty real double vectors",
                        index,
                    ));
                }
            }
        }
        if columns
            .zip(upper.and_then(|value| value.shape.element_count()))
            .is_some_and(|(left, right)| left != right)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SURROGATE-BOUND",
                "bound extents must agree",
                2,
            ));
        }
        if let Some(problem) = problem {
            for name in ["intcon", "Aineq", "bineq", "Aeq", "beq"] {
                if problem.fields.get(name).is_some_and(|value| !double(value)) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-SURROGATE-DATA",
                        "integer indices and linear data must be real double",
                        0,
                    ));
                }
            }
        } else {
            for index in 3..arity.min(8) {
                if index == 3 && fourth_options || index == 4 && arity == 5 {
                    continue;
                }
                if !double(&request.arguments[index]) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-SURROGATE-DATA",
                        "integer indices and linear data must be real double",
                        index,
                    ));
                }
            }
        }
    }
    let callback = if !checkpoint && problem.is_none() {
        scalar_facts::callback(request, &mut diagnostics);
        scalar_facts::callback_invocation_with_arguments(
            request,
            vec![facts::matrix(None, columns)],
        )
    } else {
        None
    };
    let mut summary = BTreeMap::new();
    for name in ["funccount", "elapsedtime"] {
        summary.insert(name.into(), facts::scalar());
    }
    summary.insert("message".into(), ValueFact::scalar(ValueKindFact::String));
    summary.insert(
        "constrviolation".into(),
        facts::double(ShapeFact::Unknown, StorageFact::Unknown),
    );
    summary.insert(
        "ineq".into(),
        facts::double(ShapeFact::Unknown, StorageFact::Dense),
    );
    summary.insert(
        "rngstate".into(),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    );
    let output = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(summary, true)));
    let trials = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        [
            ("X".into(), facts::matrix(None, columns)),
            (
                "Fval".into(),
                facts::double(ShapeFact::Unknown, StorageFact::Unknown),
            ),
            ("Ineq".into(), facts::matrix(None, None)),
        ]
        .into_iter()
        .collect(),
        true,
    )));
    facts::finish(
        request,
        entry,
        vec![
            facts::double(ShapeFact::Unknown, StorageFact::Dense),
            facts::double(ShapeFact::Unknown, StorageFact::Unknown),
            facts::scalar(),
            output,
            trials,
        ],
        diagnostics,
        callback,
    )
}
