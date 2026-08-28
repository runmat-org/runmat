//! Native remote-execution transport mechanisms.
//!
//! This crate owns bounded framing, replay/flow control, reconnect policy,
//! route/session state, and resumable opaque-object transfer. It does not own
//! workload scheduling, Server policy, or MATLAB values.

pub mod control;
pub mod error;
pub mod frame;
pub mod identity;
pub mod overlay;
pub mod transfer;

pub use error::{TransportError, TransportResult};
