//! Symbolic mathematics builtins for RunMat
//!
//! Provides MATLAB-compatible symbolic math operations.

mod arithmetic;
mod collect;
mod diff;
mod expand;
mod factor;
mod integrate;
pub mod matlabfunction;
mod simplify;
mod solve;
mod subs;
mod sym;
mod syms;

pub use arithmetic::{is_symbolic_operation, value_to_sym};
pub use matlabfunction::eval_compiled_function;
