//! Canonical function-reference topology for bytecode dependency discovery
//! and identity-domain remapping.

#[path = "function_remap/grammar.rs"]
mod grammar;
#[path = "function_remap/policy.rs"]
mod policy;

pub(super) use policy::visit_instruction;
pub(crate) use policy::{
    remap_closed_function, remap_eval_local_function, remap_eval_local_instruction,
};
