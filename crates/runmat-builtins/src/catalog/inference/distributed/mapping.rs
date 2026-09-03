use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, DistributedFact, ValueFact, ValueKindFact};

pub(in crate::catalog::inference) fn infer_distributed_map(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    source: DistributedFact,
) -> CallInference {
    wrap_outputs(super::infer_partition_local_call(entry, request), source)
}

pub(in crate::catalog::inference) fn infer_distributed_scalar_like(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    source: DistributedFact,
) -> CallInference {
    wrap_outputs(super::infer_partition_local_call(entry, request), source)
}

fn wrap_outputs(mut inference: CallInference, source: DistributedFact) -> CallInference {
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
