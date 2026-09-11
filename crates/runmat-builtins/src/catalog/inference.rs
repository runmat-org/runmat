use super::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

mod acceleration_semantics;
mod aggregate_semantics;
mod array;
mod distributed;
pub(in crate::catalog) mod field_path;
mod introspection_semantics;
mod logical;
mod math;
mod math_degree_trigonometric;
mod math_hyperbolic;
mod math_reduction;
mod math_rounding;
mod math_trigonometric;
mod metadata_predicate;
mod numeric_classification;
mod parallel_semantics;
mod routing;
mod scalar_logical_reduction;
mod stats_random;
mod support;
mod unary_logical_scalar;

pub use distributed::infer_partition_local_call;
pub(crate) use support::facts::materialize;
pub(super) use support::{
    argument_error, default_double_scalar, finish_fixed, finish_fixed_outputs, literal_text,
    numeric_kind, preserved_binary_residency, unavailable_rule,
};

pub fn infer_catalog_call(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    routing::infer(entry, request)
}
