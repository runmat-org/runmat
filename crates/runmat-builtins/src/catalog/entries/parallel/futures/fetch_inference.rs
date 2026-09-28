//! Fetch facts follow the future's requested outputs without owning execution.

use crate::catalog::inference::{argument_error, default_double_scalar};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, ExecutionFact,
    InferenceDiagnostic, LiteralValue, ValueFact, ValueKindFact, ValueSequenceFact,
};

pub(super) fn infer_outputs(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len().is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FETCHOUTPUTS-ARITY",
            "fetchOutputs requires a future followed by complete name-value pairs",
            request.arguments.len(),
        ));
    }
    for index in (1..request.arguments.len()).step_by(2) {
        if let Some(option) = request.literals.literal_string_at(index) {
            if !option.eq_ignore_ascii_case("UniformOutput") {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FETCHOUTPUTS-OPTION",
                    "fetchOutputs recognizes only the UniformOutput option",
                    index,
                ));
            }
        }
    }
    let output = future_output(request);
    finish(request, entry, output, false, diagnostics)
}

pub(super) fn infer_next(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FETCHNEXT-ARITY",
            "fetchNext requires a future array and optional timeout",
            request.arguments.len(),
        ));
    }
    if let Some(timeout) = request.literals.literal_args.get(1) {
        if known_numeric_literal(timeout) && !valid_timeout(timeout) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FETCHNEXT-TIMEOUT",
                "fetchNext timeout must be finite and nonnegative",
                1,
            ));
        }
    }
    let payload = future_output(request);
    if let (Some(requested), false) = (request.outputs.requested.known_count(), payload.variadic) {
        if requested > 0 && requested != payload.outputs.len() + 1 {
            diagnostics.push(InferenceDiagnostic::error(
                "RM-CATALOG-FETCHNEXT-OUTPUT-COUNT",
                "fetchNext output count must match the selected future",
            ));
        }
    }
    // A finite timeout can return [] in every output slot. A known scalar
    // index or worker result would be a false claim on that path.
    finish(request, entry, payload, true, diagnostics)
}

fn future_output(request: &CallRequest) -> ValueSequenceFact {
    match request.arguments.first().map(|fact| &fact.kind) {
        Some(ValueKindFact::Execution(ExecutionFact::Future { output, .. })) => output.clone(),
        _ => ValueSequenceFact::dynamic(),
    }
}

fn finish(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    payload: ValueSequenceFact,
    include_index: bool,
    diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let may_timeout = include_index && request.arguments.len() == 2;
    let mut outputs = Vec::with_capacity(payload.outputs.len() + usize::from(include_index));
    if include_index {
        outputs.push(if may_timeout {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        } else {
            default_double_scalar()
        });
    }
    outputs.extend(payload.outputs.into_iter().map(|fact| {
        if may_timeout {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        } else {
            fact
        }
    }));
    let mut contract = if payload.variadic {
        let mut contract = CallContract::dynamic(DynamicReason::RuntimeValue);
        contract.outputs = outputs;
        contract
    } else {
        CallContract::fixed(outputs)
    };
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    inference.diagnostics.splice(0..0, diagnostics);
    inference
}

fn known_numeric_literal(value: &LiteralValue) -> bool {
    matches!(
        value,
        LiteralValue::Number(_) | LiteralValue::Real { .. } | LiteralValue::Integer { .. }
    )
}

fn valid_timeout(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::Integer { text, .. } => text.parse::<u64>().is_ok(),
        LiteralValue::Number(value) => value.is_finite() && *value >= 0.0,
        LiteralValue::Real { text, .. } => {
            text.parse::<f64>().is_ok_and(|v| v.is_finite() && v >= 0.0)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::entries::parallel::futures::{
        FETCH_NEXT_CATALOG_ENTRY, FETCH_OUTPUTS_CATALOG_ENTRY,
    };
    use runmat_types::{
        FutureStateFact, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount,
    };

    fn future(output: ValueSequenceFact) -> ValueFact {
        ValueFact::scalar(ValueKindFact::Execution(ExecutionFact::Future {
            output,
            state: FutureStateFact::Unknown,
        }))
    }

    fn request(future: ValueFact, requested: usize, timeout: bool) -> CallRequest {
        CallRequest {
            arguments: if timeout {
                vec![future, ValueFact::unknown(DynamicReason::RuntimeValue)]
            } else {
                vec![future]
            },
            literals: if timeout {
                LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(0.1)])
            } else {
                LiteralContext::default()
            },
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(requested)),
        }
    }

    #[test]
    fn one_future_preserves_its_known_result() {
        let single = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }));
        let known = future(ValueSequenceFact::single(single.clone()));
        let fetched = infer_outputs(
            &request(known.clone(), 1, false),
            &FETCH_OUTPUTS_CATALOG_ENTRY,
        );
        assert_eq!(fetched.outputs, vec![single.clone()]);
        let next = infer_next(&request(known, 2, false), &FETCH_NEXT_CATALOG_ENTRY);
        assert_eq!(next.outputs[0], default_double_scalar());
        assert_eq!(next.outputs[1], single);
    }

    #[test]
    fn unknown_payload_remains_fetchable_and_timeout_does_not_claim_a_scalar() {
        let unknown = future(ValueSequenceFact::dynamic());
        let fetched = infer_outputs(
            &request(unknown.clone(), 1, false),
            &FETCH_OUTPUTS_CATALOG_ENTRY,
        );
        assert_eq!(fetched.outputs[0].kind, ValueKindFact::Unknown);
        assert!(fetched.diagnostics.is_empty());
        let next = infer_next(&request(unknown, 1, true), &FETCH_NEXT_CATALOG_ENTRY);
        assert_eq!(next.outputs[0].kind, ValueKindFact::Unknown);
        assert!(next.diagnostics.is_empty());

        let known = future(ValueSequenceFact::single(default_double_scalar()));
        let timed = infer_next(&request(known, 2, true), &FETCH_NEXT_CATALOG_ENTRY);
        assert_eq!(timed.outputs.len(), 2);
        assert!(timed
            .outputs
            .iter()
            .all(|fact| fact.kind == ValueKindFact::Unknown));
    }

    #[test]
    fn aggregate_or_cell_assembly_does_not_claim_a_scalar_output() {
        let aggregate = ValueFact::unknown(DynamicReason::RuntimeValue);
        let request = CallRequest {
            arguments: vec![
                aggregate,
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Logical),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("UniformOutput".into()),
                LiteralValue::Bool(false),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        };
        let fetched = infer_outputs(&request, &FETCH_OUTPUTS_CATALOG_ENTRY);
        assert_eq!(fetched.outputs[0].kind, ValueKindFact::Unknown);
        assert!(fetched.diagnostics.is_empty());
    }

    #[test]
    fn known_output_count_mismatch_and_negative_timeout_are_diagnosed() {
        let known = future(ValueSequenceFact::single(default_double_scalar()));
        let mismatch = infer_next(&request(known.clone(), 1, false), &FETCH_NEXT_CATALOG_ENTRY);
        assert!(mismatch
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-FETCHNEXT-OUTPUT-COUNT"));
        let mut timed = request(known, 2, true);
        timed.literals.literal_args[1] = LiteralValue::Number(-1.0);
        let negative = infer_next(&timed, &FETCH_NEXT_CATALOG_ENTRY);
        assert!(negative
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-FETCHNEXT-TIMEOUT"));
    }
}
