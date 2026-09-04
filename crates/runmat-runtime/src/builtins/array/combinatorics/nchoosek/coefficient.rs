use runmat_builtins::{
    NCHOOSEK_ERROR_INTERNAL, NCHOOSEK_ERROR_INVALID_INPUT, NCHOOSEK_ERROR_TOO_LARGE,
};
use runmat_value::{IntValue, Tensor, Value};

use crate::BuiltinResult;

use super::super::enumeration;
use super::error;
use super::numeric::CoefficientClass;

pub(super) fn resolve_class(
    population: CoefficientClass,
    selection: CoefficientClass,
) -> BuiltinResult<CoefficientClass> {
    if population == selection {
        return Ok(population);
    }
    if population == CoefficientClass::Double {
        return Ok(selection);
    }
    if selection == CoefficientClass::Double {
        return Ok(population);
    }
    Err(error::with_message(
        &NCHOOSEK_ERROR_INVALID_INPUT,
        "nchoosek: n and k must have the same type unless one input is double",
    ))
}

pub(super) fn value(
    population: usize,
    selection: usize,
    class: CoefficientClass,
) -> BuiltinResult<Value> {
    match class {
        CoefficientClass::Double => Ok(Value::Num(binomial_f64(population, selection))),
        CoefficientClass::Single => {
            let coefficient = binomial_f64(population, selection) as f32;
            Tensor::from_f32(vec![coefficient], vec![1, 1])
                .map(Value::Tensor)
                .map_err(internal)
        }
        CoefficientClass::I8 => integer(population, selection, i8::MAX as u128, |value| {
            IntValue::I8(value as i8)
        }),
        CoefficientClass::I16 => integer(population, selection, i16::MAX as u128, |value| {
            IntValue::I16(value as i16)
        }),
        CoefficientClass::I32 => integer(population, selection, i32::MAX as u128, |value| {
            IntValue::I32(value as i32)
        }),
        CoefficientClass::I64 => integer(population, selection, i64::MAX as u128, |value| {
            IntValue::I64(value as i64)
        }),
        CoefficientClass::U8 => integer(population, selection, u8::MAX as u128, |value| {
            IntValue::U8(value as u8)
        }),
        CoefficientClass::U16 => integer(population, selection, u16::MAX as u128, |value| {
            IntValue::U16(value as u16)
        }),
        CoefficientClass::U32 => integer(population, selection, u32::MAX as u128, |value| {
            IntValue::U32(value as u32)
        }),
        CoefficientClass::U64 => integer(population, selection, u64::MAX as u128, |value| {
            IntValue::U64(value as u64)
        }),
    }
}

fn integer(
    population: usize,
    selection: usize,
    maximum: u128,
    construct: impl FnOnce(u128) -> IntValue,
) -> BuiltinResult<Value> {
    let coefficient = enumeration::checked_binomial_u128(population, selection).map_err(|_| {
        error::with_message(&NCHOOSEK_ERROR_TOO_LARGE, "nchoosek: coefficient overflow")
    })?;
    if coefficient > maximum {
        return Err(error::with_message(
            &NCHOOSEK_ERROR_TOO_LARGE,
            "nchoosek: coefficient does not fit the requested integer class",
        ));
    }
    Ok(Value::Int(construct(coefficient)))
}

fn binomial_f64(population: usize, selection: usize) -> f64 {
    if selection > population {
        return 0.0;
    }
    let selection = selection.min(population - selection);
    (1..=selection).fold(1.0, |result, index| {
        result * (population - selection + index) as f64 / index as f64
    })
}

fn internal(detail: impl std::fmt::Display) -> crate::RuntimeError {
    error::with_message(&NCHOOSEK_ERROR_INTERNAL, format!("nchoosek: {detail}"))
}
