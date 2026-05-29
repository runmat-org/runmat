//! Control System Toolbox builtins.

use crate::RuntimeError;

pub mod db;
pub mod impulse;
pub mod nyquist;
pub mod ss;
pub mod step;
pub mod tf;
pub(crate) mod type_resolvers;

fn is_nonfatal_plot_setup_error(err: &RuntimeError) -> bool {
    let lower = err.to_string().to_ascii_lowercase();
    lower.contains("plotting is unavailable")
        || lower.contains("non-main thread")
        || lower.contains("interactive plotting failed")
        || lower.contains("eventloop can't be recreated")
}
