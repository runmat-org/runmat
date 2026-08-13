use std::collections::BTreeMap;

use runmat_hir::{FunctionId, Span};
use runmat_types::{DynamicReason, LiteralValue, ValueFact, ValueKindFact};

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
                    if requested_outputs.fixed_count() <= 1 {
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
        MirRvalue::Distributed(operation) => distributed_fact(operation, state),
        MirRvalue::Collective(operation) => collective_fact(operation, state),
        _ => crate::analysis::dataflow::simple_rvalue_fact(value, &state.value_facts()),
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
    let MirRvalue::Call(call) = value else {
        return vec![crate::analysis::dataflow::simple_rvalue_fact(
            value,
            &state.value_facts(),
        )];
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
        .map(|argument| operand_literal(argument.operand(), state))
        .collect::<Vec<_>>();
    let inference = infer_mir_call(call, &state.value_facts(), &literals, summaries, selection);
    diagnostics.extend(inference.diagnostics.iter().map(|diagnostic| {
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
        .with_primary_label("static call contract is not satisfied here")
        .with_category("call-contract")
    }));
    inference.outputs
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
    match operand {
        MirOperand::FunctionHandle(identity) => function_handle_fact(identity, summaries),
        _ => operand_fact(operand, state),
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
