use super::{format_error, memmap_field_bytes, MemmapDataType, MemmapFormat};
use crate::BuiltinResult;
use runmat_value::{NumericScalar, Value};

impl MemmapFormat {
    /// Encode one complete decoded Data value without lossy f64 conversion.
    /// The caller owns the opened extent and checks the encoded byte count
    /// before starting any write.
    pub(super) fn encode(
        &self,
        value: &Value,
        records: usize,
        repeat: Option<usize>,
    ) -> BuiltinResult<Vec<u8>> {
        let record_bytes = self.validate_layout(Some(records))?;
        let required = record_bytes
            .checked_mul(records)
            .ok_or_else(|| format_error("memmapfile: mapped Data is too large"))?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(required).map_err(|error| {
            format_error(format!("memmapfile: cannot allocate encoded Data: {error}"))
        })?;
        if self.field.is_none() {
            let record_values = self
                .shape
                .iter()
                .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                .ok_or_else(|| format_error("memmapfile: Format shape is too large"))?;
            let total_values = records
                .checked_mul(record_values)
                .ok_or_else(|| format_error("memmapfile: mapped Data is too large"))?;
            let mut expected_shape = self.shape.clone();
            if repeat.is_some() && records > 1 {
                expected_shape.push(records);
            } else if record_values != total_values {
                expected_shape = vec![total_values, 1];
            }
            encode_field(self.dtype, value, &expected_shape, required, &mut bytes)?;
        } else {
            match value {
                Value::Struct(record) if records == 1 => {
                    self.encode_record(|name| record.fields.get(name), &mut bytes)?;
                }
                Value::StructArray(array)
                    if array.len() == records && array.shape() == [records, 1] =>
                {
                    for record in array.elements() {
                        self.encode_record(|name| record.fields.get(name), &mut bytes)?;
                    }
                }
                _ => {
                    return Err(format_error(
                        "memmapfile: assignment cannot change mapped Data structure or shape",
                    ))
                }
            }
        }
        if bytes.len() != required {
            return Err(format_error(
                "memmapfile: assignment cannot change mapped Data size",
            ));
        }
        Ok(bytes)
    }

    fn encode_record<'a>(
        &self,
        get: impl Fn(&str) -> Option<&'a Value>,
        bytes: &mut Vec<u8>,
    ) -> BuiltinResult<()> {
        let first = self
            .field
            .as_deref()
            .expect("structured format has a first field");
        let first_value =
            get(first).ok_or_else(|| format_error("memmapfile: mapped Data field is missing"))?;
        encode_field(
            self.dtype,
            first_value,
            &self.shape,
            memmap_field_bytes(self.dtype, &self.shape)?,
            bytes,
        )?;
        for field in &self.additional_fields {
            let field_value = get(&field.name)
                .ok_or_else(|| format_error("memmapfile: mapped Data field is missing"))?;
            encode_field(
                field.dtype,
                field_value,
                &field.shape,
                memmap_field_bytes(field.dtype, &field.shape)?,
                bytes,
            )?;
        }
        Ok(())
    }
}

fn encode_field(
    dtype: MemmapDataType,
    value: &Value,
    expected_shape: &[usize],
    required: usize,
    target: &mut Vec<u8>,
) -> BuiltinResult<()> {
    let start = target.len();
    match (dtype, value) {
        (MemmapDataType::Character, Value::CharArray(chars)) => {
            if chars.shape() != expected_shape {
                return Err(format_error(
                    "memmapfile: assigned Data shape does not match Format",
                ));
            }
            for character in chars.to_column_major() {
                let byte = u8::try_from(character.raw()).map_err(|_| {
                    format_error("memmapfile: mapped character is outside byte range")
                })?;
                target.push(byte);
            }
        }
        (MemmapDataType::Numeric(expected), Value::Tensor(tensor))
            if tensor.numeric_dtype() == expected =>
        {
            if tensor.shape() != expected_shape {
                return Err(format_error(
                    "memmapfile: assigned Data shape does not match Format",
                ));
            }
            for index in 0..tensor.len() {
                let scalar = tensor
                    .host_buffer()
                    .value_at(index)
                    .expect("validated tensor index");
                match scalar {
                    NumericScalar::F64(value) => target.extend_from_slice(&value.to_le_bytes()),
                    NumericScalar::F32(value) => target.extend_from_slice(&value.to_le_bytes()),
                    NumericScalar::I8(value) => target.push(value as u8),
                    NumericScalar::U8(value) => target.push(value),
                    NumericScalar::I16(value) => target.extend_from_slice(&value.to_le_bytes()),
                    NumericScalar::U16(value) => target.extend_from_slice(&value.to_le_bytes()),
                    NumericScalar::I32(value) => target.extend_from_slice(&value.to_le_bytes()),
                    NumericScalar::U32(value) => target.extend_from_slice(&value.to_le_bytes()),
                    NumericScalar::I64(value) => target.extend_from_slice(&value.to_le_bytes()),
                    NumericScalar::U64(value) => target.extend_from_slice(&value.to_le_bytes()),
                }
            }
        }
        _ => {
            return Err(format_error(
                "memmapfile: assigned Data class does not match Format",
            ))
        }
    }
    if target.len() - start != required {
        return Err(format_error(
            "memmapfile: assigned Data shape does not match Format",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::MemmapFormatPresentation;
    use super::*;
    use runmat_value::{IntegerStorage, NumericDType, Tensor};

    #[test]
    fn exact_uint64_encoder_does_not_round_through_double() {
        let format = MemmapFormat {
            dtype: MemmapDataType::Numeric(NumericDType::U64),
            shape: vec![1, 1],
            field: None,
            additional_fields: Vec::new(),
            presentation: MemmapFormatPresentation::TypeName,
        };
        let value = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap(),
        );
        let encoded = format.encode(&value, 1, None).unwrap();
        assert_eq!(encoded.as_slice(), &u64::MAX.to_le_bytes());
        assert_eq!(format.decode(&encoded, None).unwrap(), value);
    }

    #[test]
    fn encoder_rejects_same_byte_count_with_changed_shape() {
        let format = MemmapFormat {
            dtype: MemmapDataType::Numeric(NumericDType::U16),
            shape: vec![2, 1],
            field: None,
            additional_fields: Vec::new(),
            presentation: MemmapFormatPresentation::TypeName,
        };
        let wrong_shape = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![1, 2]).unwrap(),
        );
        let error = format.encode(&wrong_shape, 1, Some(1)).unwrap_err();
        assert!(error.message().contains("shape"));
    }
}
