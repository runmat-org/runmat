//! Concrete array geometry shared by runtime and provider admission.
mod broadcast;
mod flip;
mod indexing;

pub use broadcast::{
    broadcast_concrete_shape, ConcreteBroadcastDimensions, ConcreteBroadcastError,
};
pub use flip::FlipGeometry;
pub use indexing::{
    is_vector_shape, logical_linear_source_is_row, numeric_linear_shape, NumericLinearShape,
};
