use std::collections::BTreeMap;

use runmat_hir::{FunctionId, Span};
use runmat_types::{DynamicReason, LiteralValue, RangeStepFact, ValueFact, ValueKindFact};

use crate::{MirOperand, MirOutputTarget, MirRvalue};

use crate::analysis::engine::FlowState;

use super::{collective_fact, distributed_fact, infer_mir_call, FunctionSummary};

pub(crate) fn infer_rvalue(
    value: &MirRvalue,
    state: &mut FlowState,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    span: Span,
    diagnostics: &mut Vec<crate::MirDiagnostic>,
) -> ValueFact {
    match value {
        MirRvalue::Use(MirOperand::FunctionHandle(identity)) => {
            function_handle_fact(identity, summaries)
        }
        MirRvalue::Future {
            function,
            requested_outputs,
            ..
        } => {
            let output = summaries
                .get(function)
                .map_or_else(dynamic_value, |summary| {
                    if requested_outputs
                        .known_count()
                        .is_none_or(|count| count <= 1)
                    {
                        summary
                            .outputs
                            .first()
                            .cloned()
                            .unwrap_or_else(dynamic_value)
                    } else {
                        ValueFact::scalar(ValueKindFact::OutputList(runmat_types::OutputListFact {
                            outputs: summary.outputs.clone(),
                            variadic: summary.variadic_outputs,
                        }))
                    }
                });
            ValueFact::scalar(ValueKindFact::Execution(
                runmat_types::ExecutionFact::Future {
                    output: Box::new(output),
                    state: runmat_types::FutureStateFact::Lazy,
                },
            ))
        }
        MirRvalue::Spawn(operand) => {
            let output = match operand_fact_with_summaries(operand, state, summaries).kind {
                ValueKindFact::Execution(runmat_types::ExecutionFact::Future {
                    output, ..
                })
                | ValueKindFact::Execution(runmat_types::ExecutionFact::Task { output, .. }) => {
                    *output
                }
                ValueKindFact::Callable(callable) => callable
                    .outputs
                    .first()
                    .cloned()
                    .unwrap_or_else(dynamic_value),
                _ => dynamic_value(),
            };
            ValueFact::scalar(ValueKindFact::Execution(
                runmat_types::ExecutionFact::Task {
                    output: Box::new(output),
                    spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
                },
            ))
        }
        MirRvalue::Call(_) => {
            infer_rvalue_outputs(value, state, summaries, None, span, diagnostics)
                .into_iter()
                .next()
                .unwrap_or_else(dynamic_value)
        }
        MirRvalue::Index { base, .. }
            if matches!(
                operand_fact_with_summaries(base, state, summaries).kind,
                ValueKindFact::Callable(_)
            ) =>
        {
            infer_rvalue_outputs(value, state, summaries, None, span, diagnostics)
                .into_iter()
                .next()
                .unwrap_or_else(dynamic_value)
        }
        MirRvalue::Distributed(operation) => distributed_fact(operation, state),
        MirRvalue::Collective(operation) => collective_fact(operation, state),
        MirRvalue::Range { start, step, end } => {
            let numeric = |operand: &MirOperand| {
                runmat_types::LiteralContext::numeric_from_literal(&operand_literal(operand, state))
            };
            let step = step.as_ref().map_or(RangeStepFact::Implicit, |operand| {
                numeric(operand).map_or(RangeStepFact::Unknown, RangeStepFact::Known)
            });
            let inference = runmat_types::infer_range(numeric(start), step, numeric(end));
            append_inference_diagnostics(
                &inference.diagnostics,
                span,
                "fact-inference",
                diagnostics,
            );
            inference.fact
        }
        _ => {
            let inference =
                crate::analysis::dataflow::simple_rvalue_inference(value, &state.value_facts());
            append_inference_diagnostics(
                &inference.diagnostics,
                span,
                "fact-inference",
                diagnostics,
            );
            inference.fact
        }
    }
}

pub(crate) fn infer_rvalue_outputs(
    value: &MirRvalue,
    state: &FlowState,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    targets: Option<&crate::MirOutputTargetList>,
    span: Span,
    diagnostics: &mut Vec<crate::MirDiagnostic>,
) -> Vec<ValueFact> {
    if let MirRvalue::Index { base, .. } = value {
        if let ValueKindFact::Callable(callable) =
            operand_fact_with_summaries(base, state, summaries).kind
        {
            let requested = targets.map_or(runmat_types::RequestedOutputCount::One, |targets| {
                targets.requested_outputs
            });
            let mut selection = runmat_types::OutputSelection::new(requested);
            if let Some(targets) = targets {
                // A sequence destination has runtime width, so later target
                // offsets are not statically known. Keep discard facts only
                // across the proven fixed-width prefix.
                for (index, target) in targets.targets.iter().enumerate() {
                    match target {
                        MirOutputTarget::Discard => {
                            selection.discarded.insert(index);
                        }
                        MirOutputTarget::Sequence(_) => break,
                        MirOutputTarget::Place(_) => {}
                    }
                }
            }
            let inference = runmat_types::infer_call(
                &callable.call_contract(DynamicReason::RuntimeValue),
                &runmat_types::CallRequest {
                    arguments: Vec::new(),
                    literals: Default::default(),
                    outputs: selection,
                },
            );
            append_inference_diagnostics(
                &inference.diagnostics,
                span,
                "callable-index-contract",
                diagnostics,
            );
            return inference.outputs;
        }
    }
    let MirRvalue::Call(call) = value else {
        let inference =
            crate::analysis::dataflow::simple_rvalue_inference(value, &state.value_facts());
        append_inference_diagnostics(&inference.diagnostics, span, "fact-inference", diagnostics);
        return vec![inference.fact];
    };
    let mut selection = runmat_types::OutputSelection::new(call.requested_outputs);
    if let Some(targets) = targets {
        for (index, target) in targets.targets.iter().enumerate() {
            if matches!(target, MirOutputTarget::Discard) {
                selection.discarded.insert(index);
            }
        }
    }
    let literals = call
        .args
        .iter()
        .filter_map(|argument| argument.operand())
        .map(|operand| operand_literal(operand, state))
        .collect::<Vec<_>>();
    let inference = infer_mir_call(call, &state.value_facts(), &literals, summaries, selection);
    append_inference_diagnostics(&inference.diagnostics, span, "call-contract", diagnostics);
    inference.outputs
}

fn append_inference_diagnostics(
    inferred: &[runmat_types::InferenceDiagnostic],
    span: Span,
    category: &'static str,
    diagnostics: &mut Vec<crate::MirDiagnostic>,
) {
    diagnostics.extend(inferred.iter().map(|diagnostic| {
        let severity = match diagnostic.severity {
            runmat_types::InferenceSeverity::Error => crate::MirDiagnosticSeverity::Error,
            runmat_types::InferenceSeverity::Warning => crate::MirDiagnosticSeverity::Warning,
            runmat_types::InferenceSeverity::Note => crate::MirDiagnosticSeverity::Information,
        };
        crate::MirDiagnostic::new(
            diagnostic.code.clone(),
            severity,
            diagnostic.message.clone(),
            span,
        )
        .with_primary_label("static value contract is not satisfied here")
        .with_category(category)
    }));
}

pub(crate) fn operand_fact(operand: &MirOperand, state: &FlowState) -> ValueFact {
    crate::analysis::dataflow::simple_rvalue_fact(
        &MirRvalue::Use(operand.clone()),
        &state.value_facts(),
    )
}

fn operand_fact_with_summaries(
    operand: &MirOperand,
    state: &FlowState,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) -> ValueFact {
    operand_fact_from_values(operand, &state.value_facts(), summaries)
}

pub(super) fn operand_fact_from_values(
    operand: &MirOperand,
    facts: &[Option<ValueFact>],
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) -> ValueFact {
    match operand {
        MirOperand::Local(local) => facts
            .get(local.0)
            .and_then(Clone::clone)
            .unwrap_or_else(dynamic_value),
        MirOperand::Constant(constant) => crate::analysis::dataflow::constant_fact(constant),
        MirOperand::FunctionHandle(identity) => function_handle_fact(identity, summaries),
    }
}

fn function_handle_fact(
    identity: &runmat_hir::CallableIdentity,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) -> ValueFact {
    let summary = match identity {
        runmat_hir::CallableIdentity::BoundFunction(function)
        | runmat_hir::CallableIdentity::AnonymousFunction(function)
        | runmat_hir::CallableIdentity::ExternalFunction { function, .. } => {
            summaries.get(function)
        }
        _ => None,
    };
    ValueFact::scalar(ValueKindFact::Callable(runmat_types::CallableFact {
        identity: Some(identity.clone()),
        capabilities: summary.map_or_else(Default::default, |summary| summary.capabilities.clone()),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: summary.map_or_else(Vec::new, |summary| summary.outputs.clone()),
        outputs_complete: summary.is_some_and(|summary| summary.outputs_complete),
        variadic_inputs: true,
        variadic_outputs: summary.is_none_or(|summary| summary.variadic_outputs),
        captures: Vec::new(),
        captures_complete: false,
    }))
}

pub(crate) fn rvalue_literal(value: &MirRvalue, state: &FlowState) -> LiteralValue {
    match value {
        MirRvalue::Use(MirOperand::Constant(constant)) => {
            crate::analysis::dataflow::literal_value(constant)
        }
        MirRvalue::Use(MirOperand::Local(local)) => state.locals[local.0].literal.clone(),
        MirRvalue::Unary(runmat_hir::OperatorKind::UnaryMinus, operand) => {
            let literal = operand_literal(operand, state);
            runmat_types::LiteralContext::numeric_from_literal(&literal)
                .map_or(LiteralValue::Unknown, |value| LiteralValue::Number(-value))
        }
        MirRvalue::Unary(runmat_hir::OperatorKind::UnaryPlus, operand) => {
            operand_literal(operand, state)
        }
        MirRvalue::Aggregate {
            kind: crate::MirAggregateKind::Tensor,
            row_lengths,
            elements,
        } => {
            let values = elements
                .iter()
                .map(|element| {
                    element
                        .operand()
                        .map(|operand| operand_literal(operand, state))
                        .unwrap_or(LiteralValue::Unknown)
                })
                .collect::<Vec<_>>();
            if values
                .iter()
                .any(|value| matches!(value, LiteralValue::Unknown))
            {
                LiteralValue::Unknown
            } else if row_lengths.len() <= 1 {
                LiteralValue::Vector(values)
            } else if row_lengths.windows(2).any(|pair| pair[0] != pair[1]) {
                LiteralValue::Unknown
            } else {
                LiteralValue::Matrix(
                    values
                        .chunks(row_lengths.first().copied().unwrap_or(0).max(1))
                        .map(<[LiteralValue]>::to_vec)
                        .collect(),
                )
            }
        }
        _ => LiteralValue::Unknown,
    }
}

fn operand_literal(operand: &MirOperand, state: &FlowState) -> LiteralValue {
    match operand {
        MirOperand::Constant(constant) => crate::analysis::dataflow::literal_value(constant),
        MirOperand::Local(local) => state.locals[local.0].literal.clone(),
        MirOperand::FunctionHandle(_) => LiteralValue::Unknown,
    }
}

fn dynamic_value() -> ValueFact {
    ValueFact::unknown(DynamicReason::Unspecified)
}
