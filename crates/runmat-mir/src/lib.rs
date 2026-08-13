pub mod analysis;
pub mod lowering;
pub mod parallel;

mod construct;

mod assembly;
mod async_;
mod block;
mod body;
mod call;
mod diagnostics;
mod function;
mod ids;
mod indexing;
mod operand;
mod place;
mod rvalue;
mod stmt;
mod terminator;

pub use assembly::*;
pub use async_::*;
pub use block::*;
pub use body::*;
pub use call::*;
pub use construct::*;
pub use diagnostics::*;
pub use function::*;
pub use ids::*;
pub use indexing::*;
pub use operand::*;
pub use place::*;
pub use rvalue::*;
pub use stmt::*;
pub use terminator::*;

/// Portable schema for [`MirAssembly`] payloads retained in executable units.
pub const MIR_SCHEMA_VERSION: u16 = 2;
