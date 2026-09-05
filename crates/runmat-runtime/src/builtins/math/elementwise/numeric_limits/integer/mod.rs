mod provider;
mod value;

use runmat_builtins::IntegerLimitKind;
use runmat_value::{ComplexTensor, IntegerComplexStorage, IntegerStorage, NumericDType, Value};

use crate::BuiltinResult;

pub(super) fn execute(
    arguments: Vec<Value>,
    kind: IntegerLimitKind,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match arguments.as_slice() {
        [] => Ok(Value::Int(value::scalar(NumericDType::I32, kind))),
        [class] if super::syntax::text(class).is_some() => {
            let text = super::syntax::text(class).expect("guarded text value");
            let class = runmat_types::NumericClass::from_class_name(text.trim())
                .and_then(runmat_types::NumericClass::integer_class)
                .ok_or_else(|| {
                    super::errors::class(builtin, format!("unsupported integer class '{text}'"))
                })?;
            Ok(Value::Int(value::scalar(class.into(), kind)))
        }
        [keyword, prototype]
            if super::syntax::text(keyword)
                .is_some_and(|text| text.eq_ignore_ascii_case("like")) =>
        {
            like(prototype, kind, builtin)
        }
        _ => Err(super::errors::syntax(
            builtin,
            "expected no arguments, an integer class name, or \"like\", prototype",
        )),
    }
}

fn like(prototype: &Value, kind: IntegerLimitKind, builtin: &'static str) -> BuiltinResult<Value> {
    match prototype {
        Value::Int(prototype) => Ok(Value::Int(value::scalar(prototype.numeric_dtype(), kind))),
        Value::Tensor(prototype) => {
            let Some(class) = prototype.numeric_dtype().integer_class() else {
                return Err(super::errors::invalid_integer_prototype(builtin));
            };
            Ok(Value::Int(value::scalar(class.into(), kind)))
        }
        Value::ComplexTensor(prototype) => {
            let Some(storage) = prototype.integer_storage() else {
                return Err(super::errors::invalid_integer_prototype(builtin));
            };
            let real = IntegerStorage::from_scalar(value::scalar(
                storage.real.integer_class().into(),
                kind,
            ));
            let imaginary = real.zeros_like(1);
            let storage = IntegerComplexStorage::new(real, imaginary)
                .map_err(|error| super::errors::class(builtin, error))?;
            ComplexTensor::new_integer(storage, vec![1, 1])
                .map(Value::ComplexTensor)
                .map_err(|error| super::errors::class(builtin, error))
        }
        Value::GpuTensor(handle) => provider::like(handle, kind, builtin),
        _ => Err(super::errors::invalid_integer_prototype(builtin)),
    }
}
