use super::{BuiltinCatalogEntry, BuiltinInferenceRule};
use runmat_types::{CallInference, CallRequest, ValueKindFact};

mod acceleration_semantics;
mod aggregate_semantics;
mod array;
mod distributed_semantics;
mod introspection_semantics;
mod math_binary;
mod math_components;
mod math_degree_trigonometric;
mod math_exponential;
mod math_fact_transforms;
mod math_hyperbolic;
mod math_inverse;
mod math_logarithms;
mod math_roots;
mod math_rounding;
mod math_special;
mod math_trigonometric;
mod metadata_predicate;
mod numeric_abs;
mod numeric_classification;
mod numeric_component;
mod numeric_conversion;
mod numeric_limit;
mod parallel_semantics;
mod routing;
mod scalar_logical_reduction;
mod stats_random;
mod support;
mod unary_logical_scalar;

pub use distributed_semantics::infer_partition_local_call;
pub(super) use support::{
    argument_error, default_double_scalar, finish_fixed, literal_text, numeric_kind,
    preserved_binary_residency, unavailable_rule,
};

pub fn infer_catalog_call(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    let distributed = request.arguments.iter().find_map(|argument| {
        let ValueKindFact::Distributed(distributed) = &argument.kind else {
            return None;
        };
        Some(distributed.clone())
    });
    if let Some(distributed) = distributed {
        match entry.placement.distributed {
            crate::BuiltinDistributedPolicy::MapUnary => {
                return distributed_semantics::infer_distributed_map(entry, request, distributed);
            }
            crate::BuiltinDistributedPolicy::ScalarLikePrototype => {
                return distributed_semantics::infer_distributed_scalar_like(
                    entry,
                    request,
                    distributed,
                );
            }
            crate::BuiltinDistributedPolicy::MaterializeArguments => {
                return infer_partition_local_call(entry, request);
            }
            crate::BuiltinDistributedPolicy::Unsupported
            | crate::BuiltinDistributedPolicy::InspectHandles => {}
        }
    }
    infer_catalog_call_local(entry, request)
}

fn infer_catalog_call_local(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    match entry.contract.inference_rule {
        BuiltinInferenceRule::Array(rule) => routing::array::infer(rule, request, entry),
        BuiltinInferenceRule::Math(rule) => routing::math::infer(rule, request, entry),
        BuiltinInferenceRule::Stats(rule) => routing::stats::infer(rule, request, entry),
        BuiltinInferenceRule::Acceleration(rule) => {
            routing::acceleration::infer(rule, request, entry)
        }
        BuiltinInferenceRule::Aggregate(rule) => routing::aggregate::infer(rule, request, entry),
        BuiltinInferenceRule::Introspection(rule) => {
            routing::introspection::infer(rule, request, entry)
        }
        BuiltinInferenceRule::Logical(rule) => routing::logical::infer(rule, request, entry),
        BuiltinInferenceRule::Parallel(rule) => routing::parallel::infer(rule, request, entry),
    }
}
