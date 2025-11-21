//! CLI argument parsing for runmatfunc.

pub mod args;
pub mod commands;

pub use args::{parse, CliArgs};
pub use commands::handle_command;
