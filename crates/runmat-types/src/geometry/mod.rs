//! Concrete array geometry shared by runtime and provider admission.
mod broadcast;
mod flip;

pub use broadcast::{
    broadcast_concrete_shape, ConcreteBroadcastDimensions, ConcreteBroadcastError,
};
pub use flip::FlipGeometry;
