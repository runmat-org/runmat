#[cfg(feature = "compiler")]
pub mod analysis;
#[cfg(feature = "compiler")]
pub mod lowering;
pub mod parallel;

mod construct;

mod assembly;
mod async_;
mod block;
mod body;
mod call;
#[cfg(feature = "compiler")]
mod diagnostics;
mod expression_region;
mod function;
mod ids;
mod indexing;
mod lexical_environment;
mod operand;
mod place;
mod remap;
mod rvalue;
mod stmt;
mod terminator;

pub use assembly::*;
pub use async_::*;
pub use block::*;
pub use body::*;
pub use call::*;
pub use construct::*;
#[cfg(feature = "compiler")]
pub use diagnostics::*;
pub use expression_region::*;
pub use function::*;
pub use ids::*;
pub use indexing::*;
pub use lexical_environment::*;
pub use operand::*;
pub use place::*;
pub use remap::{remap_closed_function_ids, remap_function_ids};
pub use rvalue::*;
pub use stmt::*;
pub use terminator::*;

/// Portable schema for [`MirAssembly`] payloads retained in executable units.
/// Version 13 forwards anonymous call-body results through the destination-
/// cardinality sequence and variadic function-output ABI.
pub const MIR_SCHEMA_VERSION: u16 = 13;
