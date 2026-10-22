//! Member and array semantics for structure values.

mod member;
pub mod operations;

pub use member::{assign_member_sequence, assign_member_values, gather_member};
