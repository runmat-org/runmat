use super::provider::valid_double_gpu_output;
use super::residency::{
    free_rejected_double_handle, resolved_actual_double_owner, valid_double_like_output,
};
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::test_support;
use crate::BuiltinResult;
use futures::executor::block_on;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::ProviderPrecision;
use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView, HostTensorView};
use runmat_builtins::{
    DOUBLE_DESCRIPTOR, DOUBLE_ERROR_INVALID_ARGUMENT, DOUBLE_ERROR_INVALID_INPUT,
};
use runmat_value::{
    CharArray, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray,
    NumericDType, NumericStorage, SparseTensor, StringArray, SymbolicArray, SymbolicExpr, Tensor,
    Value,
};

fn double_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(super::double_builtin(value, rest))
}

mod errors;
mod host;
mod provider;
mod provider_contract;
#[cfg(feature = "wgpu")]
mod wgpu;
