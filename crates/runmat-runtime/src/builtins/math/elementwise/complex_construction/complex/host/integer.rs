use runmat_builtins::{
    COMPLEX_ERROR_INTEGER_CLASS, COMPLEX_ERROR_INTERNAL, COMPLEX_ERROR_SIZE_MISMATCH,
};
use runmat_value::{
    ComplexTensor, IntegerComplexStorage, IntegerStorage, NumericStorage, Tensor, Value,
};

use crate::builtins::common::integer_conversion::{IntegerClass, IntegerClassExt};
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::BuiltinResult;

use super::super::error;
use super::{shape, RealInput};

pub(super) fn lift(storage: NumericStorage, shape: Vec<usize>) -> BuiltinResult<Value> {
    let real = storage.into_integer_storage().map_err(|_| {
        error(
            &COMPLEX_ERROR_INTERNAL,
            "floating storage reached integer construction",
        )
    })?;
    let imaginary = real.zeros_like(real.len());
    build(real, imaginary, shape)
}

pub(super) fn compose(real: &RealInput, imaginary: &RealInput) -> BuiltinResult<Value> {
    let real_storage = real.tensor.integer_storage();
    let imaginary_storage = imaginary.tensor.integer_storage();
    let prototype = real_storage.or(imaginary_storage).ok_or_else(|| {
        error(
            &COMPLEX_ERROR_INTERNAL,
            "integer composition had no integer prototype",
        )
    })?;
    validate_classes(real, imaginary, real_storage, imaginary_storage)?;

    let output_shape = shape::compatible(&real.tensor, &imaginary.tensor)?;
    let output_len = output_shape.iter().product();
    let target = IntegerClass::from_storage(prototype);
    let real = component(&real.tensor, real.is_scalar_double, target, output_len)?;
    let imaginary = component(
        &imaginary.tensor,
        imaginary.is_scalar_double,
        target,
        output_len,
    )?;
    build(real, imaginary, output_shape)
}

fn validate_classes(
    real: &RealInput,
    imaginary: &RealInput,
    real_storage: Option<&IntegerStorage>,
    imaginary_storage: Option<&IntegerStorage>,
) -> BuiltinResult<()> {
    match (real_storage, imaginary_storage) {
        (Some(left), Some(right)) if left.numeric_dtype() != right.numeric_dtype() => Err(error(
            &COMPLEX_ERROR_INTEGER_CLASS,
            format!(
                "got {} and {}; integer inputs must have the same class",
                left.class_name(),
                right.class_name()
            ),
        )),
        (Some(_), None) if !imaginary.is_scalar_double => Err(error(
            &COMPLEX_ERROR_INTEGER_CLASS,
            "the noninteger input must be a full scalar double",
        )),
        (None, Some(_)) if !real.is_scalar_double => Err(error(
            &COMPLEX_ERROR_INTEGER_CLASS,
            "the noninteger input must be a full scalar double",
        )),
        _ => Ok(()),
    }
}

fn component(
    tensor: &Tensor,
    scalar_double: bool,
    target: IntegerClass,
    output_len: usize,
) -> BuiltinResult<IntegerStorage> {
    match tensor.integer_storage() {
        Some(storage) if storage.len() == output_len => Ok(storage.clone()),
        Some(storage) if storage.len() == 1 => {
            let value = storage.value_at(0).ok_or_else(|| {
                error(
                    &COMPLEX_ERROR_INTERNAL,
                    "scalar integer storage had no value",
                )
            })?;
            storage
                .from_same_class_values(vec![value; output_len])
                .map_err(|detail| error(&COMPLEX_ERROR_INTERNAL, detail))
        }
        Some(_) => Err(error(
            &COMPLEX_ERROR_SIZE_MISMATCH,
            "real and imaginary parts must have the same size, unless one input is scalar",
        )),
        None if scalar_double => {
            let scalar = tensor
                .as_f64_slice()
                .and_then(|values| values.first())
                .copied()
                .ok_or_else(|| {
                    error(
                        &COMPLEX_ERROR_INTERNAL,
                        "scalar-double component had no double value",
                    )
                })?;
            let value = target.cast_scalar(scalar);
            Ok(target.storage(vec![value; output_len]))
        }
        None => Err(error(
            &COMPLEX_ERROR_INTEGER_CLASS,
            "the noninteger input must be a full scalar double",
        )),
    }
}

fn build(
    real: IntegerStorage,
    imaginary: IntegerStorage,
    shape: Vec<usize>,
) -> BuiltinResult<Value> {
    let storage = IntegerComplexStorage::new(real, imaginary)
        .map_err(|detail| error(&COMPLEX_ERROR_INTERNAL, detail))?;
    ComplexTensor::new_integer(storage, shape)
        .map(complex_tensor_into_value)
        .map_err(|detail| error(&COMPLEX_ERROR_INTERNAL, detail))
}
