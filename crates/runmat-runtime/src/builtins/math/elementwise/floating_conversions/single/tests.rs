use super::residency::{free_rejected_single_handle, valid_single_like_output};
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::test_support;
use crate::BuiltinResult;
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{
    SINGLE_DESCRIPTOR, SINGLE_ERROR_INVALID_INPUT, SINGLE_LIKE_OUTPUT_EXTENSION,
};
use runmat_value::{
    CharArray, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, NumericStorage,
    SparseTensor, SymbolicArray, SymbolicExpr, Tensor, Value,
};

fn single_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(super::single_builtin(value, rest))
}

mod host;
mod provider;
mod provider_contract;
mod values;
#[cfg(feature = "wgpu")]
mod wgpu;
