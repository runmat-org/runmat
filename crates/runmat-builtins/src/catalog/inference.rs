mod acceleration_semantics;
mod array;
mod distributed;
pub(in crate::catalog) mod deep_learning;
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
pub use routing::finite_fusion_fact_route;
pub use routing::infer as infer_catalog_call;
pub use routing::infer_fusion_numeric_fact;
pub(crate) use support::facts::materialize;
pub use support::materialization::gathered_value_fact;
pub(crate) use support::object_handle::known_invalid_scalar_object_handle;
pub(crate) use support::text::known_invalid_text_scalar;
pub(crate) use support::text::possible_character_row;
pub(crate) use support::text::{invalid_text_comparison_container, text_comparison_result};
pub(crate) use support::{argument_error, finish_fixed};
pub(super) use support::{
    default_double_scalar, finish_fixed_outputs, literal_text, numeric_kind,
    preserved_binary_residency,
};
