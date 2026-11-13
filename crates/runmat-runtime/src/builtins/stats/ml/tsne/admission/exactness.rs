//! Exact typed-integer admission at floating-point t-SNE boundaries.

use super::{AdmittedOption, AdmittedStatsetField};
use crate::BuiltinResult;
use runmat_value::Value;

use super::super::{cooperation::TsneCooperation, internal_with_source, invalid};

enum Frame<'a> {
    Value(&'a Value),
    Elements(&'a [Value], usize),
    Fields(&'a runmat_value::StructValue, usize),
}

pub(in super::super) async fn ensure_exact_tsne_integer_value(
    value: &Value,
    role: &str,
    cooperation: &mut TsneCooperation,
) -> BuiltinResult<()> {
    let mut pending = Vec::new();
    pending.try_reserve(1).map_err(integer_stack_error)?;
    pending.push(Frame::Value(value));
    while let Some(frame) = pending.pop() {
        match frame {
            Frame::Value(Value::Int(integer)) => {
                ensure_exact_integer(integer, role)?;
                cooperation.completed(1).await?;
            }
            Frame::Value(Value::Tensor(tensor)) => {
                if let Some(storage) = tensor.integer_storage() {
                    for index in 0..storage.len() {
                        let integer = storage
                            .value_at(index)
                            .expect("integer storage index is within its length");
                        ensure_exact_integer(&integer, role)?;
                        cooperation.completed(1).await?;
                    }
                }
            }
            Frame::Value(Value::Cell(cell)) => {
                pending.try_reserve(1).map_err(integer_stack_error)?;
                pending.push(Frame::Elements(cell.data(), 0));
            }
            Frame::Value(Value::Struct(value)) => {
                pending.try_reserve(1).map_err(integer_stack_error)?;
                pending.push(Frame::Fields(value, 0));
            }
            Frame::Value(Value::OutputList(values)) => {
                pending.try_reserve(1).map_err(integer_stack_error)?;
                pending.push(Frame::Elements(values, 0));
            }
            Frame::Elements(values, index) if index < values.len() => {
                pending.try_reserve(2).map_err(integer_stack_error)?;
                pending.push(Frame::Elements(values, index + 1));
                pending.push(Frame::Value(&values[index]));
            }
            Frame::Fields(value, index) if index < value.fields.len() => {
                let (_, field) = value
                    .fields
                    .get_index(index)
                    .expect("field index is within the structure");
                pending.try_reserve(2).map_err(integer_stack_error)?;
                pending.push(Frame::Fields(value, index + 1));
                pending.push(Frame::Value(field));
            }
            Frame::Value(_) | Frame::Elements(_, _) | Frame::Fields(_, _) => {}
        }
        cooperation.completed(1).await?;
    }
    Ok(())
}

fn ensure_exact_integer(integer: &runmat_value::IntValue, role: &str) -> BuiltinResult<()> {
    if crate::builtins::common::validation::integer_is_exact_f64(integer) {
        Ok(())
    } else {
        Err(invalid(format!(
            "tsne: integer {role} values must be exactly representable as double"
        )))
    }
}

fn integer_stack_error(source: std::collections::TryReserveError) -> crate::RuntimeError {
    internal_with_source(
        "tsne: integer validation stack exceeds available memory",
        source,
    )
}

pub(in super::super) async fn ensure_exact_tsne_floating_options(
    rest: &[AdmittedOption],
    cooperation: &mut TsneCooperation,
) -> BuiltinResult<()> {
    for option in rest {
        match option {
            AdmittedOption::CacheSizeNumeric(value) => {
                ensure_exact_tsne_integer_value(value, "CacheSize", cooperation).await?
            }
            AdmittedOption::Exaggeration(value) => {
                ensure_exact_tsne_integer_value(value, "Exaggeration", cooperation).await?
            }
            AdmittedOption::InitialY(value) => {
                ensure_exact_tsne_integer_value(value, "InitialY", cooperation).await?
            }
            AdmittedOption::LearnRate(value) => {
                ensure_exact_tsne_integer_value(value, "LearnRate", cooperation).await?
            }
            AdmittedOption::Perplexity(value) => {
                ensure_exact_tsne_integer_value(value, "Perplexity", cooperation).await?
            }
            AdmittedOption::Theta(value) => {
                ensure_exact_tsne_integer_value(value, "Theta", cooperation).await?
            }
            AdmittedOption::Options(fields) => {
                for field in fields {
                    if let AdmittedStatsetField::TolFun(value) = field {
                        ensure_exact_tsne_integer_value(value, "Options.TolFun", cooperation)
                            .await?;
                    }
                    cooperation.completed(1).await?;
                }
            }
            AdmittedOption::OptionsEmptyResident(_) => {}
            AdmittedOption::Algorithm(_)
            | AdmittedOption::CacheSizeMaximal
            | AdmittedOption::Distance(_)
            | AdmittedOption::NumDimensions(_)
            | AdmittedOption::NumPcaComponents(_)
            | AdmittedOption::Standardize(_)
            | AdmittedOption::NumPrint(_)
            | AdmittedOption::Verbose(_) => {}
        }
        cooperation.completed(1).await?;
    }
    Ok(())
}
