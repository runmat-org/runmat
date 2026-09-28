use super::finish_fixed;
use crate::{BuiltinCatalogEntry, BuiltinInferenceRule, ParallelInferenceRule};
use runmat_types::{
    codistributor_fact, CallInference, CallRequest, CodistributorClass, DynamicReason, ShapeFact,
    ValueFact, ValueKindFact,
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
    };
    finish_fixed(entry, request, output, Vec::new())
}
