mod await_preparation;
mod conditional_await;
mod control_flow;
mod ctx;
mod evaluation_order;
mod expr;
mod function;
mod class_defaults;
mod lexical_captures;
mod place;
mod stmt;

pub(crate) use ctx::MirLoweringContext;
pub use function::lower_assembly;
