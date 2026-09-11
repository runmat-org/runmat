use super::*;
use crate::{build_runtime_error, BuiltinResult};
use runmat_builtins::{
    BuiltinIntegerOutputClassRule, BuiltinIntegerOverloadKind, GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
    GETFIELD_ERROR_INDEX_SHAPE, GETFIELD_ERROR_MISSING_FIELD, GETFIELD_INDEXED_RESIDENT_EXTENSION,
    GETFIELD_INTEGER_CAPABILITIES, GETFIELD_OBJECT_FAMILY_EXTENSION,
    GETFIELD_TEXTUAL_INDEX_EXTENSION,
};
use runmat_types::MemberAccess;
use runmat_value::{
    CellArray, CharArray, ComplexTensor, HandleRef, IntValue, IntegerStorage, Listener,
    LogicalArray, MException, NumericStorage, ObjectInstance, StructArray, StructValue, Tensor,
    Value,
};

#[cfg(feature = "wgpu")]
use runmat_accelerate::backend::wgpu::provider as wgpu_backend;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::HostTensorView;

fn error_message(err: crate::RuntimeError) -> String {
    err.message().to_string()
}

fn run_getfield(base: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    futures::executor::block_on(getfield_builtin(base, rest))
}

mod aggregate_indexing;
mod index_forms;
mod linear_orientation;
mod object_properties;
mod residency;
mod special_objects;
mod structures;
mod typed_indexing;
mod validation;
