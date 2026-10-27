//! Executor-neutral member and static-member resolution.
//!
//! This module owns MATLAB struct/object/handle member access, class access
//! control, dependent and dynamic properties, `subsref`/`subsasgn` overloads,
//! graphics-handle compatibility, and canonical GC-barrier stores. Executors
//! provide their current function identity and validated method-frame evidence
//! for access checks, and attach
//! their own frame/source context to returned runtime errors.

use crate::builtins::introspection::dynamicprops;
use crate::call::identity::external_qualified_display_name;
use crate::execution::{RootedValue, RootedValueLease, RootedValueSequence};
use crate::object::dispatch::{
    call_object_property_getter_with_outputs, call_object_property_setter_with_outputs,
};
use crate::object::protocol::ProtocolDispatch;
use crate::RuntimeError;
use runmat_types::ClassIdentity;
use runmat_value::{Closure, StructValue, Tensor, Value};

/// Preserve character-vector shape at both exception property access paths.
pub(crate) fn exception_text_value(text: &str) -> Value {
    let chars = if text.is_empty() {
        runmat_value::CharArray::from_code_units(Vec::new(), 0, 0).expect("empty character array")
    } else {
        runmat_value::CharArray::new_row(text)
    };
    Value::CharArray(chars)
}

const IDENT_PROPERTY_PRIVATE_ACCESS: &str = "RunMat:PropertyPrivateAccess";
const IDENT_PROPERTY_READ_ONLY: &str = "RunMat:PropertyReadOnly";

#[path = "resolve/member_presence.rs"]
mod member_presence;
#[path = "resolve/property_store.rs"]
mod property_store;
pub(crate) use member_presence::{has_stored_member, MemberPresence};

#[path = "resolve/access.rs"]
mod access;
use access::*;
#[path = "resolve/graphics.rs"]
mod graphics;
use graphics::*;
#[path = "resolve/read.rs"]
mod read;
#[path = "resolve/read_array.rs"]
mod read_array;
pub use read::{
    load_member, load_member_dynamic, load_member_dynamic_with_context, load_member_with_access,
    load_member_with_context, read_member_sequence_text_with_access,
    read_member_sequence_text_with_context, read_member_sequence_with_access,
    read_member_sequence_with_context,
};
pub(crate) use read::{
    read_default_member_sequence_text_with_optional_access,
    read_member_sequence_text_with_optional_access,
};
#[path = "resolve/static.rs"]
mod static_member;
pub use static_member::load_static_member;
pub(crate) use static_member::{enumeration_member_name, enumeration_value_in};
#[path = "resolve/member_scope.rs"]
mod member_scope;
#[path = "resolve/store.rs"]
mod store;
pub(crate) use member_scope::{ContextualMemberScope, MemberStoreScope, OrdinaryMemberScope};
use store::store_member_with_context;
pub use store::{
    store_member, store_member_dynamic, store_member_in_context, store_member_traced_with_access,
};
pub(crate) use store::{
    store_member_text_with_optional_access, store_member_text_with_optional_access_scope,
};
#[path = "resolve/store_sequence.rs"]
mod store_sequence;
pub use store_sequence::{
    member_sequence_cardinality, store_member_dynamic_traced,
    store_member_dynamic_traced_in_context, store_member_sequence_traced,
    store_member_sequence_traced_in_context, store_member_sequence_traced_with_access,
    store_member_sequence_values, store_member_sequence_values_in_context, store_member_traced,
    store_member_traced_in_context,
};
pub(crate) use store_sequence::{
    store_default_member_sequence_values, store_member_sequence_values_with_optional_access_scope,
};
#[cfg(test)]
#[path = "resolve/tests.rs"]
mod tests;
