mod host;
mod provider;

use runmat_builtins::FloatingLimitKind;
use runmat_value::{NumericDType, Value};

use crate::BuiltinResult;

pub(super) fn execute(
    arguments: Vec<Value>,
    kind: FloatingLimitKind,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match arguments.as_slice() {
        [] => host::value(
            runmat_types::NumericClass::Double,
            false,
            false,
            kind,
            builtin,
        ),
        [class] if super::syntax::text(class).is_some() => {
            let text = super::syntax::text(class).expect("guarded text value");
            let class = parse_class(&text, builtin)?;
            host::value(class, false, false, kind, builtin)
        }
        [keyword, prototype]
            if super::syntax::text(keyword)
                .is_some_and(|text| text.eq_ignore_ascii_case("like")) =>
        {
            like(prototype, kind, builtin)
        }
        _ => Err(super::errors::syntax(
            builtin,
            "expected no arguments, a floating-point class name, or \"like\", prototype",
        )),
    }
}

fn parse_class(text: &str, builtin: &'static str) -> BuiltinResult<runmat_types::NumericClass> {
    runmat_types::NumericClass::from_class_name(text.trim())
        .filter(|class| {
            matches!(
                class,
                runmat_types::NumericClass::Double | runmat_types::NumericClass::Single
            )
        })
        .ok_or_else(|| {
            super::errors::class(
                builtin,
                format!("unsupported floating-point class '{text}'"),
            )
        })
}

fn like(prototype: &Value, kind: FloatingLimitKind, builtin: &'static str) -> BuiltinResult<Value> {
    match prototype {
        Value::Num(_) => host::value(
            runmat_types::NumericClass::Double,
            false,
            false,
            kind,
            builtin,
        ),
        Value::Tensor(tensor) => host::value(
            class(tensor.numeric_dtype(), builtin)?,
            false,
            false,
            kind,
            builtin,
        ),
        Value::Complex(_, _) => host::value(
            runmat_types::NumericClass::Double,
            true,
            false,
            kind,
            builtin,
        ),
        Value::ComplexTensor(tensor) => host::value(
            class(tensor.numeric_dtype(), builtin)?,
            true,
            false,
            kind,
            builtin,
        ),
        Value::SparseTensor(tensor) => {
            let class = class(
                tensor
                    .numeric_dtype()
                    .ok_or_else(|| super::errors::invalid_floating_prototype(builtin))?,
                builtin,
            )?;
            host::value(class, tensor.is_complex(), true, kind, builtin)
        }
        Value::GpuTensor(handle) => provider::like(handle, kind, builtin),
        _ => Err(super::errors::invalid_floating_prototype(builtin)),
    }
}

fn class(dtype: NumericDType, builtin: &'static str) -> BuiltinResult<runmat_types::NumericClass> {
    match dtype {
        NumericDType::F64 => Ok(runmat_types::NumericClass::Double),
        NumericDType::F32 => Ok(runmat_types::NumericClass::Single),
        _ => Err(super::errors::invalid_floating_prototype(builtin)),
    }
}
