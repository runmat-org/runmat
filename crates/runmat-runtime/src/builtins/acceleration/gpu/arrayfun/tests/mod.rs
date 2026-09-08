use super::arrayfun_builtin;
use super::callback::Callable;
use super::error_context::make_error_struct;
use super::input::{ArrayData, ArrayInput};
use super::output::{
    classify_for_test as classify_value, ClassifiedValueForTest as ClassifiedValue,
    UniformCollector,
};
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_builtins::{
    ARRAYFUN_ERROR_INVALID_INPUT, ARRAYFUN_ERROR_UNDEFINED_FUNCTION,
    ARRAYFUN_ERROR_UNIFORM_OUTPUT_OPTION, ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE,
    ARRAYFUN_GPU_OPTIONS_EXTENSION, ARRAYFUN_HOST_SCALAR_EXPANSION_EXTENSION,
    ARRAYFUN_TEXT_CALLABLE_EXTENSION,
};
use runmat_value::{
    CharArray, Closure, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, Tensor,
    Value,
};
use std::sync::Arc;

fn call(func: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    block_on(arrayfun_builtin(func, rest))
}

fn values(tensor: &Tensor) -> Vec<f64> {
    tensor.materialize_f64()
}

mod policy;
mod storage;
mod storage_results;

mod callback;
mod callback_resolution;
mod empty;

mod invocation;

mod provider;
mod wgpu;
