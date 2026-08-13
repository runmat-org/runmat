use std::collections::BTreeMap;

use runmat_hir::{CallableIdentity, FunctionId};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, LiteralContext,
    LiteralValue, OutputSelection, ValueFact, ValueKindFact,
};

use crate::analysis::dataflow;
use crate::{MirCall, MirCallee, MirOperand};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FunctionSummary {
    pub outputs: Vec<ValueFact>,
    pub outputs_complete: bool,
    pub variadic_outputs: bool,
    pub effects: runmat_types::EffectSet,
    pub capabilities: runmat_types::CapabilitySet,
}

pub(crate) fn infer_mir_call(
    call: &MirCall,
    facts: &[Option<ValueFact>],
    literals: &[LiteralValue],
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    selection: OutputSelection,
) -> CallInference {
    let request = CallRequest {
        arguments: call
            .args
            .iter()
            .map(|argument| operand_fact(argument.operand(), facts, summaries))
            .collect(),
        literals: LiteralContext::new(literals.to_vec()),
        outputs: selection,
    };

    if let Some(function) = bound_function(call) {
        if let Some(summary) = summaries.get(&function) {
            return infer_call(
                &CallContract {
                    outputs: summary.outputs.clone(),
                    variadic_output: (summary.variadic_outputs || !summary.outputs_complete)
                        .then(|| Box::new(ValueFact::unknown(DynamicReason::RuntimeValue))),
                    maximum_outputs: (summary.outputs_complete && !summary.variadic_outputs)
                        .then_some(summary.outputs.len()),
                    effects: summary.effects.clone(),
                    capabilities: summary.capabilities.clone(),
                    dynamic_reason: (!summary.outputs_complete || summary.variadic_outputs)
                        .then_some(DynamicReason::RuntimeValue),
                },
                &request,
            );
        }
    }

    if let Some(name) = static_name(call) {
        if let Some(entry) = runmat_builtins::builtin_catalog_entry_by_name(&name) {
            return runmat_builtins::infer_catalog_call(entry, &request);
        }
        if let Some(inference) = super::infer_legacy_builtin(&name, &request) {
            return inference;
        }
    }

    if let MirCallee::Dynamic(operand) = &call.callee {
        if let ValueKindFact::Callable(callable) = operand_fact(operand, facts, summaries).kind {
            let output_count = callable.outputs.len();
            return infer_call(
                &CallContract {
                    outputs: callable.outputs,
                    variadic_output: (callable.variadic_outputs || !callable.outputs_complete)
                        .then(|| Box::new(ValueFact::unknown(DynamicReason::RuntimeValue))),
                    maximum_outputs: (callable.outputs_complete && !callable.variadic_outputs)
                        .then_some(output_count),
                    effects: Default::default(),
                    capabilities: Default::default(),
                    dynamic_reason: (!callable.outputs_complete || callable.variadic_outputs)
                        .then_some(DynamicReason::RuntimeValue),
                },
                &request,
            );
        }
    }

    infer_call(
        &CallContract::dynamic(DynamicReason::UnresolvedCallable),
        &request,
    )
}

fn bound_function(call: &MirCall) -> Option<FunctionId> {
    match &call.callee {
        MirCallee::Static(
            CallableIdentity::BoundFunction(function)
            | CallableIdentity::AnonymousFunction(function)
            | CallableIdentity::ExternalFunction { function, .. },
        ) => Some(*function),
        _ => None,
    }
}

fn static_name(call: &MirCall) -> Option<String> {
    match &call.callee {
        MirCallee::Static(identity) => identity.display_name(),
        MirCallee::SuperConstructor { super_class, .. } => Some(super_class.clone()),
        MirCallee::SuperMethod { method, .. } => Some(method.clone()),
        MirCallee::Dynamic(_) => None,
    }
}

fn operand_fact(
    operand: &MirOperand,
    facts: &[Option<ValueFact>],
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) -> ValueFact {
    match operand {
        MirOperand::Local(local) => facts
            .get(local.0)
            .and_then(Clone::clone)
            .unwrap_or_else(dynamic_value),
        MirOperand::Constant(constant) => dataflow::constant_fact(constant),
        MirOperand::FunctionHandle(identity) => {
            let summary = match identity {
                CallableIdentity::BoundFunction(function)
                | CallableIdentity::AnonymousFunction(function)
                | CallableIdentity::ExternalFunction { function, .. } => summaries.get(function),
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
    }
}

fn dynamic_value() -> ValueFact {
    ValueFact::unknown(DynamicReason::Unspecified)
}
