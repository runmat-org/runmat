use crate::{BuiltinCatalogEntry, BuiltinDistributedPolicy};
use runmat_types::{CallInference, CallRequest, ValueKindFact};

pub(super) fn infer(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    let distributed = request.arguments.iter().find_map(|argument| {
        let ValueKindFact::Distributed(distributed) = &argument.kind else {
            return None;
        };
        Some(distributed.clone())
    });
    let Some(distributed) = distributed else {
        return super::infer_local(entry, request);
    };

    match entry.placement.distributed {
        BuiltinDistributedPolicy::MapUnary | BuiltinDistributedPolicy::MapUnaryConstrained(_) => {
            super::super::distributed::infer_distributed_map(entry, request, distributed)
        }
        BuiltinDistributedPolicy::ScalarLikePrototype => {
            super::super::distributed::infer_distributed_scalar_like(entry, request, distributed)
        }
        BuiltinDistributedPolicy::MaterializeArguments => {
            super::super::distributed::infer_partition_local_call(entry, request)
        }
        BuiltinDistributedPolicy::Unsupported | BuiltinDistributedPolicy::InspectHandles => {
            super::infer_local(entry, request)
        }
    }
}
