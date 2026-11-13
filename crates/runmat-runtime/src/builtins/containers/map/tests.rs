use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_value::{IntegerStorage, NumericDType};

pub(super) fn error_message(err: crate::RuntimeError) -> String {
    err.message.clone()
}

pub(super) fn containers_map_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    block_on(super::containers_map_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}

pub(super) fn containers_map_keys(map: Value) -> BuiltinResult<Value> {
    block_on(super::containers_map_keys(map))
}

pub(super) fn containers_map_values(map: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(super::containers_map_values(
        map,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

pub(super) fn containers_map_is_key(map: Value, key_spec: Value) -> BuiltinResult<Value> {
    block_on(super::containers_map_is_key(map, key_spec))
}

pub(super) fn containers_map_remove(map: Value, key_spec: Value) -> BuiltinResult<Value> {
    block_on(super::containers_map_remove(map, key_spec))
}

pub(super) fn containers_map_subsref(
    map: Value,
    kind: String,
    payload: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
    block_on(super::containers_map_subsref(map, subscript))
}

pub(super) fn containers_map_subsasgn(
    map: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
    block_on(super::containers_map_subsasgn(map, subscript, rhs))
}

#[path = "tests/general.rs"]
mod general;
#[path = "tests/integer.rs"]
mod integer;
#[path = "tests/lifecycle.rs"]
mod lifecycle;
#[path = "tests/resident.rs"]
mod resident;
