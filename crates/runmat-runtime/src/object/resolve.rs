//! Executor-neutral member and static-member resolution.
//!
//! This module owns MATLAB struct/object/handle member access, class access
//! control, dependent and dynamic properties, `subsref`/`subsasgn` overloads,
//! graphics-handle compatibility, and canonical GC-barrier stores. Executors
//! provide only their current function identity for access checks and attach
//! their own frame/source context to returned runtime errors.

use crate::builtins::introspection::dynamicprops;
use crate::call::identity::external_qualified_display_name;
use crate::execution::{RootedValue, RootedValueLease, RootedValueSequence};
use crate::object::dispatch::{
    call_object_property_getter_with_outputs, call_object_property_setter_with_outputs,
};
use crate::object::indexing::ObjectIndexOp;
use crate::sequence::ResolveValueSequence;
use crate::RuntimeError;
use runmat_types::ClassIdentity;
use runmat_value::{Closure, StructValue, Tensor, Value};

/// Preserve character-vector shape at both exception property access paths.
pub(crate) fn exception_text_value(text: &str) -> Value {
    let chars = if text.is_empty() {
        runmat_value::CharArray::new(Vec::new(), 0, 0).expect("empty character array")
    } else {
        runmat_value::CharArray::new_row(text)
    };
    Value::CharArray(chars)
}

const IDENT_PROPERTY_PRIVATE_ACCESS: &str = "RunMat:PropertyPrivateAccess";
const IDENT_PROPERTY_READ_ONLY: &str = "RunMat:PropertyReadOnly";

#[path = "resolve/property_store.rs"]
mod property_store;
#[path = "resolve/member_presence.rs"]
mod member_presence;
pub(crate) use member_presence::{has_stored_member, MemberPresence};

#[path = "resolve/access.rs"]
mod access;
use access::*;
#[path = "resolve/graphics.rs"]
mod graphics;
use graphics::*;
#[path = "resolve/read.rs"]
mod read;
pub use read::{
    load_member, load_member_dynamic, load_member_dynamic_with_context, load_member_with_context,
    read_member_sequence_with_context,
};
#[path = "resolve/static.rs"]
mod static_member;
pub use static_member::load_static_member;
#[path = "resolve/store.rs"]
mod store;
use store::store_member_with_context;
pub use store::{store_member, store_member_dynamic, store_member_in_context};
#[path = "resolve/store_sequence.rs"]
mod store_sequence;
pub use store_sequence::{
    member_sequence_cardinality, store_member_dynamic_traced,
    store_member_dynamic_traced_in_context, store_member_sequence_traced,
    store_member_sequence_traced_in_context, store_member_traced, store_member_traced_in_context,
};
#[cfg(test)]
#[path = "resolve/tests.rs"]
mod tests;
