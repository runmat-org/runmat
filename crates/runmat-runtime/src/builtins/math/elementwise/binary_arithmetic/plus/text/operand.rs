//! Borrowed string-conversion elements; no converted operand mirror.
use runmat_value::{NumericScalar, Utf16CodeUnit, Value};

use super::{invalid_input, TextError};
use crate::BuiltinResult;

mod shape;
use shape::Shape;

pub(super) struct Operand<'a> {
    value: &'a Value,
    shape: Shape<'a>,
}

impl<'a> Operand<'a> {
    pub(super) fn new(value: &'a Value) -> BuiltinResult<Self> {
        let shape = match value {
            Value::String(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Complex(..)
            | Value::Bool(_) => Shape::Matrix([1, 1]),
            Value::StringArray(array) => Shape::Borrowed(array.shape()),
            Value::Tensor(array) => Shape::Borrowed(array.shape()),
            Value::ComplexTensor(array) => Shape::Borrowed(array.shape()),
            Value::LogicalArray(array) => Shape::Borrowed(&array.shape),
            Value::CharArray(array) => {
                if crate::builtins::strings::common::has_unsupported_character_pages(value) {
                    return Err(invalid_input("character inputs must be two-dimensional"));
                }
                // An empty character row converts to one empty string.
                Shape::Matrix([array.rows.max(1), 1])
            }
            Value::Cell(array) => Shape::Borrowed(&array.shape),
            Value::SparseTensor(array) => Shape::Matrix([array.rows, array.cols]),
            _ => {
                return Err(invalid_input(
                    "operand cannot be converted to a string array",
                ))
            }
        };
        Ok(Self { value, shape })
    }

    pub(super) fn shape(&self) -> &[usize] {
        self.shape.dimensions()
    }

    pub(super) fn element(&self, index: usize) -> BuiltinResult<Element<'a>> {
        let missing_element = || invalid_input("operand shape exceeds its payload");
        match self.value {
            Value::String(text) => Ok(Element::text(text.as_str())),
            Value::StringArray(array) => array
                .data()
                .get(index)
                .map(|text| Element::text(text.as_str()))
                .ok_or_else(missing_element),
            Value::Num(value) => Ok(Element::numeric(NumericScalar::F64(*value))),
            Value::Int(value) => Ok(Element::numeric(value.clone().into())),
            Value::Complex(real, imaginary) => Ok(Element::complex(*real, *imaginary)),
            Value::Bool(value) => Ok(Element::Logical(*value)),
            Value::LogicalArray(array) => array
                .data
                .get(index)
                .map(|value| Element::Logical(*value != 0))
                .ok_or_else(missing_element),
            Value::Tensor(array) => array
                .numeric_value_at(index)
                .map(Element::numeric)
                .ok_or_else(missing_element),
            Value::ComplexTensor(array) => {
                let (real, imaginary) =
                    array.numeric_value_at(index).ok_or_else(missing_element)?;
                match (real, imaginary) {
                    (NumericScalar::F64(real), NumericScalar::F64(imaginary)) => {
                        Ok(Element::complex(real, imaginary))
                    }
                    (NumericScalar::F32(real), NumericScalar::F32(imaginary)) => {
                        Ok(Element::complex(real.into(), imaginary.into()))
                    }
                    _ => Err(invalid_input(
                        "complex integer string operands are not supported",
                    )),
                }
            }
            Value::CharArray(array) => {
                if array.rows == 0 {
                    return Ok(Element::Text(""));
                }
                let start = index.checked_mul(array.cols).ok_or_else(missing_element)?;
                let end = start.checked_add(array.cols).ok_or_else(missing_element)?;
                array
                    .data()
                    .get(start..end)
                    .map(Element::Characters)
                    .ok_or_else(missing_element)
            }
            Value::Cell(array) => {
                let value = array.get_column_major(index).ok_or_else(missing_element)?;
                // Cells contain scalar conversions, not recursively expanded arrays.
                if matches!(value, Value::Cell(_)) {
                    return Err(invalid_input(
                        "cell elements must convert to scalar strings",
                    ));
                }
                let operand = Operand::new(value)?;
                if !operand.shape().iter().all(|extent| *extent == 1) {
                    return Err(invalid_input(
                        "cell elements must convert to scalar strings",
                    ));
                }
                operand.element(0)
            }
            Value::SparseTensor(array) => {
                if array.rows == 0 {
                    return Err(missing_element());
                }
                let column = index / array.rows;
                let row = index % array.rows;
                let begin = *array.col_ptrs.get(column).ok_or_else(missing_element)?;
                let end = *array.col_ptrs.get(column + 1).ok_or_else(missing_element)?;
                let rows = array
                    .row_indices
                    .get(begin..end)
                    .ok_or_else(missing_element)?;
                let stored = rows.binary_search(&row).ok().map(|offset| begin + offset);
                if array.is_logical() {
                    return Ok(Element::Logical(stored.is_some()));
                }
                let Some(stored) = stored else {
                    return Ok(Element::numeric(NumericScalar::F64(0.0)));
                };
                if array.is_complex() {
                    let value = array.complex_value_at(stored).ok_or_else(missing_element)?;
                    Ok(Element::complex(value.0, value.1))
                } else {
                    array
                        .numeric_value_at(stored)
                        .map(Element::numeric)
                        .ok_or_else(missing_element)
                }
            }
            _ => Err(invalid_input(
                "operand cannot be converted to a string array",
            )),
        }
    }
}

pub(super) enum Element<'a> {
    Text(&'a str),
    Characters(&'a [Utf16CodeUnit]),
    Numeric(NumericScalar),
    Complex(f64, f64),
    Logical(bool),
    Missing,
}

impl<'a> Element<'a> {
    fn text(text: Option<&'a str>) -> Self {
        match text {
            Some(text) => Self::Text(text),
            None => Self::Missing,
        }
    }

    fn numeric(value: NumericScalar) -> Self {
        if crate::builtins::common::format::scalar_text::numeric_is_missing(value) {
            Self::Missing
        } else {
            Self::Numeric(value)
        }
    }

    fn complex(real: f64, imaginary: f64) -> Self {
        if real.is_nan() || imaginary.is_nan() {
            Self::Missing
        } else {
            Self::Complex(real, imaginary)
        }
    }

    pub(super) fn is_missing(&self) -> bool {
        matches!(self, Self::Missing)
    }

    pub(super) fn write(&self, output: &mut impl std::fmt::Write) -> Result<(), TextError> {
        use crate::builtins::common::format::scalar_text::{
            write_complex_text, write_numeric_text,
        };
        match self {
            Self::Text(text) => output.write_str(text)?,
            Self::Characters(units) => {
                let mut index = 0;
                for character in char::decode_utf16(units.iter().map(|unit| unit.raw())) {
                    let character = character.map_err(|error| {
                        TextError::Utf16(runmat_value::Utf16DecodeError {
                            index,
                            unpaired_surrogate: error.unpaired_surrogate(),
                        })
                    })?;
                    output.write_char(character)?;
                    index += character.len_utf16();
                }
            }
            Self::Numeric(value) => write_numeric_text(output, *value)?,
            Self::Complex(real, imaginary) => write_complex_text(output, *real, *imaginary)?,
            Self::Logical(value) => output.write_str(if *value { "true" } else { "false" })?,
            Self::Missing => return Err(TextError::Format(std::fmt::Error)),
        }
        Ok(())
    }
}
