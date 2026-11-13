//! MATLAB-compatible `toeplitz` builtin.

use runmat_builtins::{
    BuiltinErrorDescriptor,
    TOEPLITZ_COMPLEX_INTEGER_INPUT_EXTENSION as COMPLEX_INTEGER_INPUT_EXTENSION,
    TOEPLITZ_ERROR_INTERNAL as ERROR_INTERNAL, TOEPLITZ_ERROR_INVALID_INPUT as ERROR_INVALID_INPUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    ComplexTensor, IntegerComplexStorage, IntegerStorage, NumericDType, NumericStorage, Tensor,
    Value,
};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::math::elementwise::complex_components::conjugate_integer_imaginary_storage;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "toeplitz";

#[runtime_builtin(
    name = "toeplitz",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::shape::toeplitz"
)]
async fn toeplitz_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(error_with_detail(
            &runmat_builtins::TOEPLITZ_ERROR_TOO_MANY_OUTPUTS,
            "expected at most one output",
        ));
    }
    if !(1..=2).contains(&args.len()) {
        return Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "expected one or two inputs",
        ));
    }
    if args
        .iter()
        .any(crate::builtins::common::validation::is_typed_complex_integer)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COMPLEX_INTEGER_INPUT_EXTENSION,
            NAME,
        )?;
    }
    let protected: Vec<_> = args
        .iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect();
    let source = gpu_helpers::select_resident_output_source(protected.iter().cloned(), NAME)?;
    let output = match args.len() {
        1 => {
            let r = InputVector::from_value(args.into_iter().next().expect("r")).await?;
            toeplitz_from_vectors(r.conjugated(), r, true)
        }
        2 => {
            let mut iter = args.into_iter();
            let c = InputVector::from_value(iter.next().expect("c")).await?;
            let r = InputVector::from_value(iter.next().expect("r")).await?;
            toeplitz_from_vectors(c, r, false)
        }
        _ => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "expected one or two inputs",
        )),
    }?;
    let Some(source) = source else {
        return Ok(output);
    };
    gpu_helpers::restore_class_preserving_value_protecting(
        &source,
        output,
        NAME,
        &protected.iter().collect::<Vec<_>>(),
    )
    .map_err(|error| error_with_detail(&ERROR_INTERNAL, error))
}

#[path = "toeplitz/input.rs"]
mod input;
use input::InputVector;

fn toeplitz_from_vectors(c: InputVector, r: InputVector, one_input: bool) -> BuiltinResult<Value> {
    let rows = c.len();
    let cols = r.len();
    if matches!(&c, InputVector::TypedComplex(_)) || matches!(&r, InputVector::TypedComplex(_)) {
        return toeplitz_typed_complex(c, r, rows, cols, one_input);
    }
    if matches!(&c, InputVector::TypedInteger(_)) && matches!(&r, InputVector::TypedInteger(_)) {
        return toeplitz_typed_integer(c, r, rows, cols);
    }
    if matches!(&c, InputVector::TypedInteger(_)) || matches!(&r, InputVector::TypedInteger(_)) {
        return Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "integer inputs must use one common native integer class",
        ));
    }
    if c.is_complex() || r.is_complex() {
        return toeplitz_complex(c, r, rows, cols, one_input);
    }
    toeplitz_real(c, r, rows, cols)
}

fn toeplitz_typed_integer(
    c: InputVector,
    r: InputVector,
    rows: usize,
    cols: usize,
) -> BuiltinResult<Value> {
    let (InputVector::TypedInteger(c), InputVector::TypedInteger(r)) = (c, r) else {
        unreachable!("typed integer branch requires typed integer vectors");
    };
    if c.numeric_dtype() != r.numeric_dtype() {
        return Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "integer inputs must use one common native integer class",
        ));
    }
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| error_with_detail(&ERROR_INTERNAL, "output size overflow"))?;
    let mut values = Vec::with_capacity(len);
    for col in 0..cols {
        for row in 0..rows {
            let storage = if row >= col { &c } else { &r };
            values.push(
                storage
                    .value_at(row.abs_diff(col))
                    .expect("toeplitz index is in bounds"),
            );
        }
    }
    let storage = c
        .from_exact_values_like(values)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))?;
    Tensor::new_integer(storage, vec![rows, cols])
        .map(Value::Tensor)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))
}

fn toeplitz_typed_complex(
    c: InputVector,
    r: InputVector,
    rows: usize,
    cols: usize,
    one_input: bool,
) -> BuiltinResult<Value> {
    let (InputVector::TypedComplex(c), InputVector::TypedComplex(r)) = (c, r) else {
        return Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "typed complex integer inputs must use the same integer class",
        ));
    };
    if c.numeric_dtype() != r.numeric_dtype() {
        return Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "typed complex integer inputs must use the same integer class",
        ));
    }
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| error_with_detail(&ERROR_INTERNAL, "output size overflow"))?;
    let mut real = Vec::with_capacity(len);
    let mut imag = Vec::with_capacity(len);
    for col in 0..cols {
        for row in 0..rows {
            let index = row.abs_diff(col);
            let storage = if row >= col { &c } else { &r };
            real.push(
                storage
                    .real()
                    .value_at(index)
                    .expect("toeplitz index is in bounds"),
            );
            imag.push(
                storage
                    .imag()
                    .value_at(index)
                    .expect("toeplitz index is in bounds"),
            );
        }
    }
    if one_input && !real.is_empty() {
        let diagonal_real = r.real().value_at(0).expect("nonempty diagonal");
        let diagonal_imag = r.imag().value_at(0).expect("nonempty diagonal");
        for index in 0..rows.min(cols) {
            real[index + index * rows] = diagonal_real.clone();
            imag[index + index * rows] = diagonal_imag.clone();
        }
    }
    let real = c
        .real()
        .from_exact_values_like(real)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))?;
    let imag = c
        .imag()
        .from_exact_values_like(imag)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))?;
    let storage = IntegerComplexStorage::new(real, imag)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))?;
    ComplexTensor::new_integer(storage, vec![rows, cols])
        .map(Value::ComplexTensor)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))
}

fn toeplitz_real(c: InputVector, r: InputVector, rows: usize, cols: usize) -> BuiltinResult<Value> {
    let dtype = common_real_dtype(&c, &r);
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| error_with_detail(&ERROR_INTERNAL, "output size overflow"))?;
    let mut data = Vec::with_capacity(len);
    for col in 0..cols {
        for row in 0..rows {
            let value = if row >= col {
                c.real_at(row - col)
            } else {
                r.real_at(col - row)
            };
            data.push(value);
        }
    }
    if len == 1 {
        return scalar_or_tensor_real(data, rows, cols, dtype);
    }
    Tensor::new_with_dtype(data, vec![rows, cols], dtype)
        .map(Value::Tensor)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))
}

fn toeplitz_complex(
    c: InputVector,
    r: InputVector,
    rows: usize,
    cols: usize,
    one_input: bool,
) -> BuiltinResult<Value> {
    let dtype = common_complex_dtype(&c, &r);
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| error_with_detail(&ERROR_INTERNAL, "output size overflow"))?;
    let mut data = Vec::with_capacity(len);
    for col in 0..cols {
        for row in 0..rows {
            let value = if row >= col {
                c.complex_at(row - col)
            } else {
                r.complex_at(col - row)
            };
            data.push(value);
        }
    }
    if one_input && !data.is_empty() {
        let diagonal = r.complex_at(0);
        for idx in 0..rows.min(cols) {
            data[idx + idx * rows] = diagonal;
        }
    }
    if len == 1 && dtype == NumericDType::F64 {
        let (re, im) = data[0];
        return Ok(Value::Complex(re, im));
    }
    ComplexTensor::from_f64_values_with_dtype(data, vec![rows, cols], dtype)
        .map(Value::ComplexTensor)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))
}

fn common_complex_dtype(c: &InputVector, r: &InputVector) -> NumericDType {
    if [c, r].iter().any(|value| match value {
        InputVector::Real { dtype, .. } | InputVector::Complex { dtype, .. } => {
            *dtype == NumericDType::F32
        }
        _ => false,
    }) {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}

fn validate_vector_shape(shape: &[usize], len: usize) -> BuiltinResult<()> {
    if len <= 1 {
        return Ok(());
    }
    if is_vector_shape(shape) {
        return Ok(());
    }
    Err(error_with_detail(
        &ERROR_INVALID_INPUT,
        "inputs must be vectors",
    ))
}

fn is_vector_shape(shape: &[usize]) -> bool {
    match shape {
        [] => true,
        [_] => true,
        [rows, cols] => *rows == 1 || *cols == 1,
        dims => dims.iter().filter(|&&dim| dim != 1).take(2).count() <= 1,
    }
}

fn common_real_dtype(c: &InputVector, r: &InputVector) -> NumericDType {
    match (c.dtype(), r.dtype()) {
        (Some(left), Some(right)) if left == right => left,
        _ => NumericDType::F64,
    }
}

fn scalar_or_tensor_real(
    data: Vec<f64>,
    rows: usize,
    cols: usize,
    dtype: NumericDType,
) -> BuiltinResult<Value> {
    if dtype == NumericDType::F64 {
        Ok(Value::Num(data[0]))
    } else {
        Tensor::new_with_dtype(data, vec![rows, cols], dtype)
            .map(Value::Tensor)
            .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))
    }
}

fn error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {detail}", error.message)).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
#[path = "toeplitz/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "toeplitz/provider_tests.rs"]
mod provider_tests;
