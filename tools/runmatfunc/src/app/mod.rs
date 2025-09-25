//! High-level application orchestration: manages state and delegates to submodules.

pub mod actions;
pub mod config;
pub mod state;

pub use state::AppContext;
