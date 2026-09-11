use super::*;
use crate::{build_runtime_error, BuiltinResult};
use runmat_builtins::{
    SETFIELD_ERROR_INDEX_OUT_OF_BOUNDS, SETFIELD_ERROR_INDEX_SHAPE, SETFIELD_ERROR_MISSING_FIELD,
    SETFIELD_ERROR_NON_STRUCT_ASSIGNMENT, SETFIELD_INDEXED_RESIDENT_EXTENSION,
    SETFIELD_OBJECT_FAMILY_EXTENSION,
};
use runmat_gc::gc_allocate;
use runmat_types::MemberAccess;
use runmat_value::{
    CellArray, HandleRef, IntValue, IntegerStorage, LogicalArray, NumericScalar, ObjectInstance,
    StructArray, StructValue, Tensor, Value,
};

fn error_message(err: crate::RuntimeError) -> String {
    err.message().to_string()
}

fn run_setfield(base: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    futures::executor::block_on(setfield_builtin(base, rest))
}

mod aggregate_indexing;
mod object_handles;
mod object_properties;
mod residency;
mod scalar_indexing;
mod structures;
mod validation;
mod vector_nd_indexing;
