//! Dense real floating construction, after shape/class admission.
use super::{host, provider};
use crate::context::preparation::{execute_current_invocation, BuiltinPreparationAdmission};
use crate::BuiltinResult;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinPreparationFormId, COMPLEX_CATALOG_ENTRY};
use runmat_value::{NumericDType, Value};

fn dense_float(value: &Value) -> bool {
    matches!(value, Value::Tensor(t) if matches!(t.numeric_dtype(), NumericDType::F32 | NumericDType::F64))
}

fn admission(binary: bool) -> BuiltinPreparationAdmission {
    BuiltinPreparationAdmission {
        binding: BuiltinBindingIdentity {
            builtin: COMPLEX_CATALOG_ENTRY.identity,
            variant: "default",
        },
        form: BuiltinPreparationFormId(u16::from(binary)),
        signature: usize::from(binary),
        input_count: if binary { 2 } else { 1 },
        requested_outputs: crate::output_count::current_output_count(),
    }
}

pub(super) async fn unary(real: Value) -> BuiltinResult<Value> {
    if !dense_float(&real) {
        return provider::unary(real).await;
    }
    execute_current_invocation(admission(false), vec![real], |mut values| {
        provider::unary(values.remove(0))
    })
    .await
}

pub(super) async fn binary(real: Value, imaginary: Value) -> BuiltinResult<Value> {
    if !dense_float(&real) || !dense_float(&imaginary) {
        return provider::binary(real, imaginary).await;
    }
    if let (Value::Tensor(real), Value::Tensor(imaginary)) = (&real, &imaginary) {
        host::compatible_shape(real, imaginary)?;
    }
    execute_current_invocation(admission(true), vec![real, imaginary], |mut values| {
        let imaginary = values.remove(1);
        provider::binary(values.remove(0), imaginary)
    })
    .await
}
