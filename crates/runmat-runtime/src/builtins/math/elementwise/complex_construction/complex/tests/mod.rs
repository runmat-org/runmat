pub(super) use futures::executor::block_on;
pub(super) use runmat_builtins::{COMPLEX_ERROR_INTEGER_CLASS, COMPLEX_ERROR_INVALID_INPUT};
pub(super) use runmat_value::{
    CharArray, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray,
    NumericDType, StringArray, Tensor, Value,
};

pub(super) use crate::builtins::common::{gpu_helpers, test_support};
pub(super) use crate::BuiltinResult;

pub(super) fn complex_call(real: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(super::complex_builtin(real, rest))
}

mod host;
mod provider;

#[cfg(feature = "wgpu")]
mod wgpu;
