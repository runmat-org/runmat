use super::super::{nonlinear_facts as facts, scalar_facts};
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let (function, initial, lower, upper, options) = if request.arguments.len() == 1 {
        match &request.arguments[0].kind {
            ValueKindFact::Struct(problem) => {
                let function =
                    facts::field(problem, "objective").or_else(|| facts::field(problem, "fun"));
                let initial = facts::field(problem, "x0");
                if problem.fields_complete && (function.is_none() || initial.is_none()) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-LSQNONLIN-PROBLEM",
                        "problem requires objective or fun, and x0",
                        0,
                    ));
                }
                for field in ["A", "b", "Aineq", "bineq", "Aeq", "beq", "nonlcon"] {
                    if facts::field(problem, field)
                        .and_then(|v| v.shape.element_count())
                        .is_some_and(|n| n != 0)
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-LSQNONLIN-CONSTRAINT",
                            "nonempty constraints are not supported",
                            0,
                        ));
                    }
                }
                (
                    function.unwrap_or(&unknown),
                    initial,
                    facts::field(problem, "lb"),
                    facts::field(problem, "ub"),
                    facts::field(problem, "options"),
                )
            }
            ValueKindFact::Unknown => (&unknown, None, None, None, None),
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-LSQNONLIN-PROBLEM",
                    "single argument must be a problem struct",
                    0,
                ));
                (&unknown, None, None, None, None)
            }
        }
    } else {
        if ![2, 4, 5, 8, 9, 10].contains(&request.arguments.len()) {
            diagnostics.push(argument_error(
                "RM-CATALOG-LSQNONLIN-ARITY",
                "expected fun,x0, paired bounds/options, or empty constraint placeholders",
                0,
            ));
        }
        let options_index = match request.arguments.len() {
            5 => Some(4),
            10 => Some(9),
            _ => None,
        };
        if request.arguments.len() >= 8 {
            for (index, value) in request.arguments.iter().enumerate().take(9).skip(4) {
                if value.shape.element_count().is_some_and(|n| n != 0) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-LSQNONLIN-CONSTRAINT",
                        "constraint placeholders must be empty",
                        index,
                    ));
                }
            }
        }
        (
            request.arguments.first().unwrap_or(&unknown),
            request.arguments.get(1),
            request.arguments.get(2),
            request.arguments.get(3),
            options_index.and_then(|index| request.arguments.get(index)),
        )
    };
    let callback_request = CallRequest {
        arguments: vec![function.clone()],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    scalar_facts::callback(&callback_request, &mut diagnostics);
    facts::numeric(initial, 1, false, &mut diagnostics);
    let n = initial.and_then(|v| v.shape.element_count());
    facts::bound(lower, 2, n, true, &mut diagnostics);
    facts::bound(upper, 3, n, true, &mut diagnostics);
    facts::options(options, 4, true, &mut diagnostics);
    let x = facts::like(initial);
    let callback =
        scalar_facts::callback_invocation_with_arguments(&callback_request, vec![x.clone()]);
    let value = callback.as_ref().and_then(|c| c.outputs.first());
    facts::numeric(value, 0, false, &mut diagnostics);
    let residual = facts::double(
        value.map(|v| v.shape.clone()).unwrap_or(ShapeFact::Unknown),
        StorageFact::Dense,
    );
    let m = value.and_then(|v| v.shape.element_count());
    let outputs = vec![
        x.clone(),
        facts::scalar(),
        residual,
        facts::scalar(),
        facts::diagnostics(&[
            "iterations",
            "funcCount",
            "firstorderopt",
            "stepsize",
            "cgiterations",
            "lssteplength",
        ]),
        facts::lambda(&x),
        facts::matrix(m, n),
    ];
    facts::finish(request, entry, outputs, diagnostics, callback)
}
