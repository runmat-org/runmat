//! Concrete array geometry shared by runtime and provider admission.
mod broadcast;

pub use broadcast::{broadcast_concrete_shape, ConcreteBroadcastError};
