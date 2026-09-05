mod host;
mod table;

use super::super::errors;
use super::OPERATION;
use crate::BuiltinResult;
use runmat_value::Value;

pub(super) async fn execute(value: Value) -> BuiltinResult<(Value, Value)> {
    super::super::extensions::ensure(OPERATION, &value).await?;
    match value {
        Value::Object(object) if crate::builtins::table::is_tabular_object(&object) => {
            table::evaluate(object).await
        }
        Value::GpuTensor(_) => Err(errors::with_detail(
            OPERATION,
            &runmat_builtins::LOG2_ERROR_GPU_DISSECTION,
            "GPU-resident input is unsupported",
        )),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(errors::with_detail(
            OPERATION,
            &runmat_builtins::LOG2_ERROR_COMPLEX_DISSECTION,
            "complex input is rejected by the current compatibility release",
        )),
        Value::SparseTensor(_) => Err(errors::invalid(
            OPERATION,
            "sparse input is not currently supported",
        )),
        Value::CharArray(chars) => host::characters(chars),
        Value::String(_) | Value::StringArray(_) => {
            Err(errors::invalid(OPERATION, "expected real numeric input"))
        }
        value => host::numeric(value),
    }
}
