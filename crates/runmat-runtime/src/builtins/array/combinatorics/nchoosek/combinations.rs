use runmat_builtins::{
    NCHOOSEK_ERROR_INTERNAL, NCHOOSEK_ERROR_INVALID_INPUT, NCHOOSEK_ERROR_TOO_LARGE,
};
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, IntegerComplexStorage, LogicalArray, NumericStorage,
    Tensor, Value,
};

use crate::BuiltinResult;

use super::super::enumeration;
use super::{arguments, error};

const MAX_OUTPUT_ELEMENTS: usize = 50_000_000;

pub(super) fn value(value: Value, selection: usize) -> BuiltinResult<Value> {
    match value {
        Value::Complex(real, imaginary) => {
            complex_values(vec![(real, imaginary)], vec![1, 1], selection)
        }
        Value::Bool(flag) => logical(vec![u8::from(flag)], vec![1, 1], selection),
        Value::Tensor(tensor) => numeric(tensor, selection),
        Value::ComplexTensor(tensor) => complex(tensor, selection),
        Value::LogicalArray(array) => logical(array.data.into_vec(), array.shape, selection),
        Value::CharArray(array) => characters(array, selection),
        Value::Num(_) | Value::Int(_) => Err(error::with_message(
            &NCHOOSEK_ERROR_INVALID_INPUT,
            "nchoosek: scalar n must be a nonnegative integer",
        )),
        _ => Err(error::from_descriptor(&NCHOOSEK_ERROR_INVALID_INPUT)),
    }
}

fn numeric(tensor: Tensor, selection: usize) -> BuiltinResult<Value> {
    let population = arguments::vector_len(&tensor.shape)?;
    let rows = output_rows(population, selection)?;
    let storage = tensor.into_numeric_storage().map_err(internal)?;
    let storage = numeric_storage(storage, rows, selection)?;
    Tensor::from_numeric_storage(storage, vec![rows, selection])
        .map(Value::Tensor)
        .map_err(internal)
}

fn numeric_storage(
    storage: NumericStorage,
    rows: usize,
    selection: usize,
) -> BuiltinResult<NumericStorage> {
    macro_rules! combinations {
        ($values:expr, $variant:ident) => {
            NumericStorage::$variant(columns(&$values, rows, selection)?)
        };
    }
    Ok(match storage {
        NumericStorage::F64(values) => combinations!(values, F64),
        NumericStorage::F32(values) => combinations!(values, F32),
        NumericStorage::I8(values) => combinations!(values, I8),
        NumericStorage::I16(values) => combinations!(values, I16),
        NumericStorage::I32(values) => combinations!(values, I32),
        NumericStorage::I64(values) => combinations!(values, I64),
        NumericStorage::U8(values) => combinations!(values, U8),
        NumericStorage::U16(values) => combinations!(values, U16),
        NumericStorage::U32(values) => combinations!(values, U32),
        NumericStorage::U64(values) => combinations!(values, U64),
    })
}

fn complex(tensor: ComplexTensor, selection: usize) -> BuiltinResult<Value> {
    let population = arguments::vector_len(&tensor.shape)?;
    let rows = output_rows(population, selection)?;
    let storage = match tensor.into_complex_storage() {
        ComplexStorage::F64(values) => {
            ComplexStorage::F64(columns(&values, rows, selection)?.into())
        }
        ComplexStorage::F32(values) => {
            ComplexStorage::F32(columns(&values, rows, selection)?.into())
        }
        ComplexStorage::Integer(storage) => {
            let real = storage
                .real
                .from_exact_values_like(columns(&storage.real.exact_values(), rows, selection)?)
                .map_err(internal)?;
            let imaginary = storage
                .imag
                .from_exact_values_like(columns(&storage.imag.exact_values(), rows, selection)?)
                .map_err(internal)?;
            ComplexStorage::Integer(IntegerComplexStorage::new(real, imaginary).map_err(internal)?)
        }
    };
    ComplexTensor::from_complex_storage(storage, vec![rows, selection])
        .map(Value::ComplexTensor)
        .map_err(internal)
}

fn complex_values(
    values: Vec<(f64, f64)>,
    shape: Vec<usize>,
    selection: usize,
) -> BuiltinResult<Value> {
    let population = arguments::vector_len(&shape)?;
    let rows = output_rows(population, selection)?;
    ComplexTensor::new(columns(&values, rows, selection)?, vec![rows, selection])
        .map(Value::ComplexTensor)
        .map_err(internal)
}

fn logical(data: Vec<u8>, shape: Vec<usize>, selection: usize) -> BuiltinResult<Value> {
    let population = arguments::vector_len(&shape)?;
    let rows = output_rows(population, selection)?;
    LogicalArray::new(columns(&data, rows, selection)?, vec![rows, selection])
        .map(Value::LogicalArray)
        .map_err(internal)
}

fn characters(array: CharArray, selection: usize) -> BuiltinResult<Value> {
    let population = arguments::vector_len(&[array.rows, array.cols])?;
    let rows = output_rows(population, selection)?;
    CharArray::new(rows_data(&array.data, rows, selection)?, rows, selection)
        .map(Value::CharArray)
        .map_err(internal)
}

fn output_rows(population: usize, selection: usize) -> BuiltinResult<usize> {
    let rows = enumeration::checked_binomial_usize(population, selection).map_err(too_large)?;
    let elements = rows.checked_mul(selection).ok_or_else(|| {
        error::with_message(&NCHOOSEK_ERROR_TOO_LARGE, "nchoosek: output size overflow")
    })?;
    if elements > MAX_OUTPUT_ELEMENTS {
        return Err(error::from_descriptor(&NCHOOSEK_ERROR_TOO_LARGE));
    }
    Ok(rows)
}

fn columns<T: Clone>(values: &[T], rows: usize, selection: usize) -> BuiltinResult<Vec<T>> {
    if rows == 0 || selection == 0 {
        return Ok(Vec::new());
    }
    let mut output = reserve(rows, selection)?;
    output.resize_with(rows * selection, || values[0].clone());
    let mut row = 0;
    enumeration::for_each_combination(values.len(), selection, |indices| {
        for (column, &source) in indices.iter().enumerate() {
            output[column * rows + row] = values[source].clone();
        }
        row += 1;
    });
    Ok(output)
}

fn rows_data<T: Clone>(values: &[T], rows: usize, selection: usize) -> BuiltinResult<Vec<T>> {
    if rows == 0 || selection == 0 {
        return Ok(Vec::new());
    }
    let mut output = reserve(rows, selection)?;
    enumeration::for_each_combination(values.len(), selection, |indices| {
        output.extend(indices.iter().map(|&source| values[source].clone()));
    });
    Ok(output)
}

fn reserve<T>(rows: usize, columns: usize) -> BuiltinResult<Vec<T>> {
    let count = rows.checked_mul(columns).ok_or_else(|| {
        error::with_message(&NCHOOSEK_ERROR_TOO_LARGE, "nchoosek: output size overflow")
    })?;
    let mut output = Vec::new();
    output.try_reserve_exact(count).map_err(too_large)?;
    Ok(output)
}

fn too_large(detail: impl std::fmt::Debug) -> crate::RuntimeError {
    error::with_message(&NCHOOSEK_ERROR_TOO_LARGE, format!("nchoosek: {detail:?}"))
}

fn internal(detail: impl std::fmt::Display) -> crate::RuntimeError {
    error::with_message(&NCHOOSEK_ERROR_INTERNAL, format!("nchoosek: {detail}"))
}
