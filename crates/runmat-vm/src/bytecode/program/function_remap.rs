//! Canonical function-reference topology for bytecode dependency discovery
//! and identity-domain remapping.

#[path = "function_remap/grammar.rs"]
mod grammar;
#[path = "function_remap/policy.rs"]
mod policy;
#[path = "function_remap/product.rs"]
mod product;
#[path = "function_remap/reference_sites.rs"]
mod reference_sites;

pub(super) use policy::visit_instruction;
pub use policy::{
    rebind_session_source, remap_closed_session_function, remap_session_entrypoint,
    remap_session_function, SessionCallableBinding,
};
pub(crate) use policy::{
    remap_closed_function, remap_eval_local_function, remap_eval_local_instruction,
};
pub use product::{remap_decorated_eval_metadata, remap_decorated_session_definitions};
pub use reference_sites::remap_reference_sites;
pub use reference_sites::{instruction_reference_sites, FunctionReferenceKind};
