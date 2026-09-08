mod character;
mod complex;
mod destination;
mod logical;
mod numeric;

use runmat_value::{ComplexTensor, IntegerComplexStorage, LogicalArray, Tensor, Value};

use super::error::{cell2mat_error_with_message, CELL2MAT_ERROR_INTERNAL};
use super::input::{self, CellEntry, ElementKind};
use super::plan::AssemblyPlan;
use crate::BuiltinResult;

pub(super) fn assemble(
    kind: ElementKind,
    entries: &[CellEntry],
    plan: &AssemblyPlan,
) -> BuiltinResult<Value> {
    match kind {
        ElementKind::Numeric => numeric(entries, plan),
        ElementKind::Complex => complex(entries, plan),
        ElementKind::TypedComplexInteger => typed_complex_integer(entries, plan),
        ElementKind::Logical => logical(entries, plan),
        ElementKind::Character => character::assemble(entries, plan),
    }
}

fn numeric(entries: &[CellEntry], plan: &AssemblyPlan) -> BuiltinResult<Value> {
    let storage = input::numeric_prototype(entries)?.zeros_like(plan.output_len);
    let mut output = Tensor::from_numeric_storage(storage, plan.output_shape.clone())
        .map_err(|error| internal(error.to_string()))?;
    numeric::copy(entries, plan, &mut output)?;
    Ok(Value::Tensor(output))
}

fn complex(entries: &[CellEntry], plan: &AssemblyPlan) -> BuiltinResult<Value> {
    let mut data = vec![(0.0, 0.0); plan.output_len];
    complex::copy_floating(entries, plan, &mut data)?;
    ComplexTensor::new(data, plan.output_shape.clone())
        .map(Value::ComplexTensor)
        .map_err(|error| internal(error.to_string()))
}

fn typed_complex_integer(entries: &[CellEntry], plan: &AssemblyPlan) -> BuiltinResult<Value> {
    let prototype = input::typed_complex_integer(entries)?;
    let mut storage = IntegerComplexStorage::new(
        prototype.real.zeros_like(plan.output_len),
        prototype.imag.zeros_like(plan.output_len),
    )
    .map_err(|error| internal(error.to_string()))?;
    complex::copy_integer(entries, plan, &mut storage)?;
    ComplexTensor::new_integer(storage, plan.output_shape.clone())
        .map(Value::ComplexTensor)
        .map_err(|error| internal(error.to_string()))
}

fn logical(entries: &[CellEntry], plan: &AssemblyPlan) -> BuiltinResult<Value> {
    let mut data = vec![0; plan.output_len];
    logical::copy(entries, plan, &mut data)?;
    LogicalArray::new(data, plan.output_shape.clone())
        .map(Value::LogicalArray)
        .map_err(|error| internal(error.to_string()))
}

fn internal(detail: String) -> crate::RuntimeError {
    cell2mat_error_with_message(format!("cell2mat: {detail}"), &CELL2MAT_ERROR_INTERNAL)
}
