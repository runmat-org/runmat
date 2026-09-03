use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, DistributedFact, ValueFact, ValueKindFact};

pub fn infer_partition_local_call(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
) -> CallInference {
    let mut projected = request.clone();
    projected.arguments = request
        .arguments
        .iter()
        .map(|argument| match &argument.kind {
            ValueKindFact::Distributed(distributed) => distributed.value.as_ref().clone(),
            _ => argument.clone(),
        })
        .collect();
    super::routing::infer_local(entry, &projected)
}

pub(super) fn infer_distributed_map(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    source: DistributedFact,
) -> CallInference {
    let mut inference = infer_partition_local_call(entry, request);
    inference.outputs = inference
        .outputs
        .into_iter()
        .map(|value| {
            ValueFact::scalar(ValueKindFact::Distributed(DistributedFact {
                id: source.id,
                owner: source.owner,
                scheme: source.scheme.clone(),
                value: Box::new(value),
                materializable: source.materializable,
            }))
        })
        .collect();
    inference
}

pub(super) fn infer_distributed_scalar_like(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    source: DistributedFact,
) -> CallInference {
    let mut inference = infer_partition_local_call(entry, request);
    inference.outputs = inference
        .outputs
        .into_iter()
        .map(|value| {
            ValueFact::scalar(ValueKindFact::Distributed(DistributedFact {
                id: source.id,
                owner: source.owner,
                scheme: source.scheme.clone(),
                value: Box::new(value),
                materializable: source.materializable,
            }))
        })
        .collect();
    inference
}
