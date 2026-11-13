use runmat_builtins::{
    BuiltinBindingIdentity, BuiltinCatalogIdentity, BuiltinPreparationFormId, ImageIndexMode,
};
use runmat_value::{IntValue, NumericDType, Value};
use std::future::Future;

use crate::context::preparation::{execute_invocation, BuiltinPreparationAdmission};
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use crate::BuiltinResult;

pub(crate) fn decode(
    value: &Value,
) -> Result<Option<ImageIndexMode>, runmat_value::Utf16DecodeError> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            ImageIndexMode::from_option_text(text)
        }
        Value::CharArray(chars) => {
            let text: String = runmat_value::decode_utf16_code_units(chars.data())?;
            ImageIndexMode::from_option_text(&text)
        }
        _ => None,
    })
}

/// Preserve domain-owned admission. A sink only materializes the image; it
/// does not reinterpret the indexed control or choose a conversion class.
fn admitted(value: &Value, mode: ImageIndexMode, target: NumericDType) -> bool {
    let supported = |class| match mode {
        ImageIndexMode::Intensity => matches!(
            class,
            NumericDType::F64
                | NumericDType::F32
                | NumericDType::I16
                | NumericDType::U8
                | NumericDType::U16
        ),
        ImageIndexMode::Indexed => matches!(
            class,
            NumericDType::F64 | NumericDType::U8 | NumericDType::U16
        ),
    };
    let needs_value_bounds = |class| {
        mode == ImageIndexMode::Indexed
            && (matches!(target, NumericDType::U8 | NumericDType::U16)
                && class == NumericDType::F64
                || target == NumericDType::U8 && class == NumericDType::U16)
    };
    match value {
        Value::Num(number) if mode == ImageIndexMode::Indexed => {
            let maximum = match target {
                NumericDType::U8 => 256.0,
                NumericDType::U16 => 65536.0,
                _ => f64::INFINITY,
            };
            *number <= maximum
        }
        Value::Num(_) | Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::Int(integer) => match (integer, mode) {
            (IntValue::U8(_), _)
            | (IntValue::U16(_), ImageIndexMode::Intensity)
            | (IntValue::I16(_), ImageIndexMode::Intensity) => true,
            (IntValue::U16(value), ImageIndexMode::Indexed) => {
                target != NumericDType::U8 || *value <= u16::from(u8::MAX)
            }
            _ => false,
        },
        Value::Tensor(tensor) => {
            let class = tensor.numeric_dtype();
            supported(class) && !needs_value_bounds(class)
        }
        Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle) => false,
        Value::GpuTensor(handle) => {
            super::common::resident_numeric_dtype(handle, "image conversion")
                .is_ok_and(|class| supported(class) && !needs_value_bounds(class))
        }
        _ => false,
    }
}

pub(crate) async fn execute<F, Fut>(
    context: Option<&RuntimeContext>,
    name: &'static str,
    value: Value,
    mode: ImageIndexMode,
    target: NumericDType,
    input_count: usize,
    body: F,
) -> BuiltinResult<RootedValue>
where
    F: FnOnce(Value) -> Fut,
    Fut: Future<Output = BuiltinResult<RootedValue>>,
{
    if !admitted(&value, mode, target) {
        return body(value).await;
    }
    let signature = match mode {
        ImageIndexMode::Intensity => 0,
        ImageIndexMode::Indexed => 1,
    };
    execute_invocation(
        context,
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: BuiltinCatalogIdentity { name },
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature,
            input_count,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![value],
        |mut operands| body(operands.pop().expect("one admitted image operand")),
    )
    .await
}

#[cfg(test)]
mod tests;
