//! Control System Toolbox builtins.

use crate::RuntimeError;

pub mod damp;
pub mod db;
pub mod dcgain;
pub mod feedback;
pub mod impulse;
pub mod isstable;
pub mod nyquist;
pub mod pole;
pub mod rlocus;
pub mod ss;
pub mod step;
pub mod stepinfo;
pub mod tf;
pub mod tf_model;
pub(crate) mod type_resolvers;

fn is_nonfatal_plot_setup_error(err: &RuntimeError) -> bool {
    let lower = err.to_string().to_ascii_lowercase();
    lower.contains("plotting is unavailable")
        || lower.contains("non-main thread")
        || lower.contains("interactive plotting failed")
        || lower.contains("eventloop can't be recreated")
}
