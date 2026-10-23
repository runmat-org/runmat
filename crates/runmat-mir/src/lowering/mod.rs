mod await_preparation;
mod await_replacements;
pub(crate) use await_replacements::AwaitReplacements;
mod class_defaults;
mod conditional_await;
mod control_flow;
mod ctx;
mod evaluation_order;
mod expr;
mod fixed_target;
mod function;
mod lexical_captures;
mod place;
mod stmt;

pub(crate) use ctx::MirLoweringContext;
pub use function::lower_assembly;
