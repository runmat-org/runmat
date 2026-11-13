//! Static future facts for the two scheduling identities.

use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, ExecutionFact, FutureStateFact, LiteralValue,
    ValueFact, ValueKindFact, ValueSequenceFact,
};

pub(super) fn infer_parfeval(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, false)
}

pub(super) fn infer_parfeval_on_all(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer(request, entry, true)
}

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry, on_all: bool) -> CallInference {
    let first = request.arguments.first().map(|value| &value.kind);
    let callable_index = usize::from(matches!(
        first,
        Some(ValueKindFact::Execution(ExecutionFact::Pool))
    ));
    let count_index = callable_index + 1;
    let mut diagnostics = Vec::new();
    if request.arguments.len() <= count_index {
        diagnostics.push(argument_error(
            "RM-CATALOG-PARFEVAL-ARITY",
            format!(
                "{} requires a function handle and output count",
                entry.identity.name
            ),
            count_index,
        ));
    }
    let count = request
        .literals
        .literal_args
        .get(count_index)
        .and_then(output_count);
    if request
        .literals
        .literal_args
        .get(count_index)
        .is_some_and(known_numeric_literal)
        && count.is_none()
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-PARFEVAL-OUTPUT-COUNT",
            "scheduled output count must be a nonnegative integer scalar",
            count_index,
        ));
    }
    let output = match (
        first,
        request
            .arguments
            .get(callable_index)
            .map(|value| &value.kind),
        count,
    ) {
        // Fetching all-worker results assembles them by vertical concatenation
        // (or as cells). That result cannot inherit one worker's exact shape.
        (_, _, Some(count)) if on_all => ValueSequenceFact::fixed(
            (0..count)
                .map(|_| ValueFact::unknown(DynamicReason::RuntimeValue))
                .collect(),
        ),
        // An unresolved first argument could be either a pool or a callable.
        // The downstream executor resolves that overload; do not invent outputs.
        (Some(ValueKindFact::Unknown), _, _) => ValueSequenceFact::dynamic(),
        (_, Some(ValueKindFact::Callable(callable)), Some(count)) => ValueSequenceFact {
            outputs: (0..count)
                .map(|index| {
                    callable
                        .outputs
                        .get(index)
                        .cloned()
                        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue))
                })
                .collect(),
            variadic: !callable.outputs_complete || callable.variadic_outputs,
        },
        _ => ValueSequenceFact::dynamic(),
    };
    let future = ValueFact::scalar(ValueKindFact::Execution(ExecutionFact::Future {
        output,
        state: FutureStateFact::Unknown,
    }));
    finish_fixed(entry, request, future, diagnostics)
}

fn known_numeric_literal(value: &LiteralValue) -> bool {
    matches!(
        value,
        LiteralValue::Number(_) | LiteralValue::Real { .. } | LiteralValue::Integer { .. }
    )
}

fn output_count(value: &LiteralValue) -> Option<usize> {
    match value {
        LiteralValue::Integer { text, .. } => text.parse().ok(),
        LiteralValue::Number(value) => finite_nonnegative_integer(*value),
        LiteralValue::Real { text, .. } => text.parse().ok().and_then(finite_nonnegative_integer),
        _ => None,
    }
}

fn finite_nonnegative_integer(value: f64) -> Option<usize> {
    (value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= usize::MAX as f64)
        .then_some(value as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::entries::parallel::futures::{
        PARFEVAL_CATALOG_ENTRY, PARFEVAL_ON_ALL_CATALOG_ENTRY,
    };
    use runmat_types::{
        CallableFact, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount,
    };

    fn request(arguments: Vec<ValueFact>, count: LiteralValue) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::new(vec![LiteralValue::Unknown, count]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        }
    }

    #[test]
    fn typed_integer_output_counts_are_accepted_by_both_schedulers() {
        for (entry, inference) in [
            (
                &PARFEVAL_CATALOG_ENTRY,
                infer_parfeval as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
            ),
            (&PARFEVAL_ON_ALL_CATALOG_ENTRY, infer_parfeval_on_all),
        ] {
            let inference = inference(
                &request(
                    vec![
                        ValueFact::unknown(DynamicReason::RuntimeValue),
                        ValueFact::unknown(DynamicReason::RuntimeValue),
                    ],
                    LiteralValue::Integer {
                        text: "2".into(),
                        class: NumericClass::UInt32,
                    },
                ),
                entry,
            );
            assert!(inference.diagnostics.is_empty(), "{}", entry.identity.name);
            assert!(matches!(
                inference.outputs[0].kind,
                ValueKindFact::Execution(ExecutionFact::Future { .. })
            ));
        }
    }

    #[test]
    fn known_callable_outputs_remain_typed_in_future_fact() {
        let single = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }));
        let callable = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
            identity: None,
            capabilities: Default::default(),
            parameters: Vec::new(),
            parameters_complete: true,
            outputs: vec![single.clone()],
            outputs_complete: true,
            variadic_inputs: false,
            variadic_outputs: false,
            captures: Vec::new(),
            captures_complete: true,
        }));
        let inference = infer_parfeval(
            &request(
                vec![callable, ValueFact::unknown(DynamicReason::RuntimeValue)],
                LiteralValue::Integer {
                    text: "1".into(),
                    class: NumericClass::UInt32,
                },
            ),
            &PARFEVAL_CATALOG_ENTRY,
        );
        let ValueKindFact::Execution(ExecutionFact::Future { output, .. }) =
            &inference.outputs[0].kind
        else {
            panic!("parfeval must return a future fact");
        };
        assert_eq!(output.outputs, vec![single]);
        assert!(!output.variadic);
    }

    #[test]
    fn all_worker_assembly_keeps_count_without_claiming_one_workers_shape() {
        let callable = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
            identity: None,
            capabilities: Default::default(),
            parameters: Vec::new(),
            parameters_complete: true,
            outputs: vec![ValueFact::scalar(ValueKindFact::Logical)],
            outputs_complete: true,
            variadic_inputs: false,
            variadic_outputs: false,
            captures: Vec::new(),
            captures_complete: true,
        }));
        let inference = infer_parfeval_on_all(
            &request(
                vec![callable, ValueFact::unknown(DynamicReason::RuntimeValue)],
                LiteralValue::Number(1.0),
            ),
            &PARFEVAL_ON_ALL_CATALOG_ENTRY,
        );
        let ValueKindFact::Execution(ExecutionFact::Future { output, .. }) =
            &inference.outputs[0].kind
        else {
            panic!("parfevalOnAll must return a future fact");
        };
        assert_eq!(output.outputs.len(), 1);
        assert_eq!(output.outputs[0].kind, ValueKindFact::Unknown);
    }

    #[test]
    fn negative_known_output_count_and_missing_arguments_are_diagnosed() {
        let negative = infer_parfeval(
            &request(
                vec![
                    ValueFact::unknown(DynamicReason::RuntimeValue),
                    ValueFact::unknown(DynamicReason::RuntimeValue),
                ],
                LiteralValue::Number(-1.0),
            ),
            &PARFEVAL_CATALOG_ENTRY,
        );
        assert!(negative
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-PARFEVAL-OUTPUT-COUNT"));
        let missing = infer_parfeval_on_all(
            &request(Vec::new(), LiteralValue::Unknown),
            &PARFEVAL_ON_ALL_CATALOG_ENTRY,
        );
        assert!(missing
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-PARFEVAL-ARITY"));
    }
}
