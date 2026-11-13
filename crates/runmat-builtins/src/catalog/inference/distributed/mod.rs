use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, ValueKindFact};

mod admission;
mod mapping;

pub(in crate::catalog::inference) use mapping::{
    infer_distributed_map, infer_distributed_scalar_like,
};

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
    let mut inference = super::routing::infer_local(entry, &projected);
    admission::validate(entry, request, &mut inference.diagnostics);
    inference
}
