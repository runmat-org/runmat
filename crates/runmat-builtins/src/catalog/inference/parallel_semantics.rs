use super::{default_double_scalar, finish_fixed, literal_text};
use crate::{BuiltinCatalogEntry, BuiltinInferenceRule, ParallelInferenceRule};
use runmat_types::{
    codistributor_fact, infer_call, CallContract, CallInference, CallRequest, CodistributorClass,
    DynamicReason, ExecutionFact, FutureStateFact, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ShapeFact, ValueFact, ValueKindFact,
};

pub(super) fn infer_parallel_data(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let BuiltinInferenceRule::Parallel(rule) = entry.contract.inference_rule else {
        unreachable!("parallel inference accepts only typed parallel rules")
    };
    let output = match rule {
        ParallelInferenceRule::Barrier | ParallelInferenceRule::Send => {
            ValueFact::scalar(ValueKindFact::Void)
        }
        ParallelInferenceRule::Probe => ValueFact::scalar(ValueKindFact::Logical),
        ParallelInferenceRule::SpmdIndex | ParallelInferenceRule::SpmdSize => {
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }))
        }
        ParallelInferenceRule::LocalPart => {
            match request.arguments.first().map(|fact| &fact.kind) {
                Some(ValueKindFact::Distributed(distributed)) => distributed.value.as_ref().clone(),
                _ => ValueFact::unknown(DynamicReason::RuntimeValue),
            }
        }
        ParallelInferenceRule::Redistribute => request
            .arguments
            .first()
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
        ParallelInferenceRule::GetCodistributor => {
            let class = request.arguments.first().and_then(|fact| match &fact.kind {
                ValueKindFact::Distributed(distributed) => distributed
                    .scheme
                    .as_ref()
                    .and_then(CodistributorClass::from_scheme),
                _ => None,
            });
            codistributor_fact(class)
        }
        ParallelInferenceRule::GlobalIndices => ValueFact::unknown(DynamicReason::RuntimeValue),
        ParallelInferenceRule::Codistributor1d => {
            codistributor_fact(Some(CodistributorClass::OneDimensional))
        }
        ParallelInferenceRule::Codistributor2dbc => {
            codistributor_fact(Some(CodistributorClass::TwoDimensionalBlockCyclic))
        }
        ParallelInferenceRule::Codistributor => codistributor_fact(None),
        ParallelInferenceRule::CodistributorIsComplete | ParallelInferenceRule::Iscodistributed => {
            ValueFact::scalar(ValueKindFact::Logical)
        }
        ParallelInferenceRule::Broadcast => request
            .arguments
            .get(1)
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
        ParallelInferenceRule::SendReceive => request
            .arguments
            .get(2)
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
        ParallelInferenceRule::Gplus => request
            .arguments
            .first()
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
        ParallelInferenceRule::Cat => request
            .arguments
            .first()
            .cloned()
            .map(|mut fact| {
                fact.shape = ShapeFact::Unknown;
                fact
            })
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
        ParallelInferenceRule::FunctionalReduce => ValueFact::unknown(DynamicReason::RuntimeValue),
        ParallelInferenceRule::Distributed
        | ParallelInferenceRule::Codistributed
        | ParallelInferenceRule::CodistributedBuild
        | ParallelInferenceRule::Receive => ValueFact::unknown(DynamicReason::RuntimeValue),
        ParallelInferenceRule::FetchNext
        | ParallelInferenceRule::FetchOutputs
        | ParallelInferenceRule::Gcp
        | ParallelInferenceRule::GetCurrentJob
        | ParallelInferenceRule::GetCurrentTask
        | ParallelInferenceRule::GetCurrentWorker
        | ParallelInferenceRule::Parfeval
        | ParallelInferenceRule::ParfevalOnAll
        | ParallelInferenceRule::Parpool => {
            unreachable!("parallel rule was routed to its dedicated inference handler")
        }
    };
    finish_fixed(entry, request, output, Vec::new())
}

pub(super) fn infer_parallel_pool(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    current_pool: bool,
) -> CallInference {
    let nocreate = current_pool
        && request
            .literals
            .literal_args
            .first()
            .and_then(literal_text)
            .is_some_and(|value| value.eq_ignore_ascii_case("nocreate"));
    let output = if nocreate {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    } else {
        ValueFact::scalar(ValueKindFact::Execution(ExecutionFact::Pool))
    };
    finish_fixed(entry, request, output, Vec::new())
}

pub(super) fn infer_parallel_future(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let explicit_pool = matches!(
        request.arguments.first().map(|fact| &fact.kind),
        Some(ValueKindFact::Execution(ExecutionFact::Pool))
    );
    let callable_index = usize::from(explicit_pool);
    let output_count_index = callable_index + 1;
    let output = scheduled_callable_output(request, callable_index, output_count_index);
    let future = ValueFact::scalar(ValueKindFact::Execution(ExecutionFact::Future {
        output,
        state: FutureStateFact::Unknown,
    }));
    finish_fixed(entry, request, future, Vec::new())
}

fn scheduled_callable_output(
    request: &CallRequest,
    callable_index: usize,
    output_count_index: usize,
) -> runmat_types::ValueSequenceFact {
    let Some(ValueKindFact::Callable(callable)) =
        request.arguments.get(callable_index).map(|fact| &fact.kind)
    else {
        return runmat_types::ValueSequenceFact::dynamic();
    };
    let Some(output_count) = request
        .literals
        .literal_args
        .get(output_count_index)
        .and_then(literal_output_count)
    else {
        return runmat_types::ValueSequenceFact::dynamic();
    };
    if output_count == 0 {
        return runmat_types::ValueSequenceFact::fixed(Vec::new());
    }
    let outputs = (0..output_count)
        .map(|index| {
            callable
                .outputs
                .get(index)
                .cloned()
                .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue))
        })
        .collect::<Vec<_>>();
    runmat_types::ValueSequenceFact {
        outputs,
        variadic: !callable.outputs_complete || callable.variadic_outputs,
    }
}

fn literal_output_count(literal: &LiteralValue) -> Option<usize> {
    let LiteralValue::Number(value) = literal else {
        return None;
    };
    (value.is_finite() && *value >= 0.0 && value.fract() == 0.0 && *value <= usize::MAX as f64)
        .then_some(*value as usize)
}

pub(super) fn infer_parallel_fetch(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    include_index: bool,
) -> CallInference {
    let payload = match request.arguments.first().map(|fact| &fact.kind) {
        Some(ValueKindFact::Execution(ExecutionFact::Future { output, .. })) => output.clone(),
        _ => runmat_types::ValueSequenceFact::dynamic(),
    };
    let mut outputs = Vec::new();
    if include_index {
        outputs.push(default_double_scalar());
    }
    outputs.extend(payload.outputs);
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    infer_call(&contract, request)
}
