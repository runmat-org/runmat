use super::super::compat_error;
use super::{dtype_size, memmap_field_bytes, MemmapField, MemmapFormat};
use crate::BuiltinResult;
use runmat_value::{IntegerStorage, StructValue, Tensor, Value};

impl MemmapFormat {
    pub(in crate::builtins::io::repl_fs::compat) fn decode(
        &self,
        bytes: &[u8],
        repeat: Option<usize>,
    ) -> BuiltinResult<Value> {
        if self.field.is_some() {
            return self.decode_table(bytes, repeat);
        }
        let element_size = dtype_size(&self.dtype)?;
        let record_len = self
            .shape
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| compat_error("memmapfile", "memmapfile: Format shape is too large"))?;
        let record_bytes = self.validate_layout(repeat)?;
        let available_records = bytes.len() / record_bytes;
        let record_count = repeat.unwrap_or(available_records).min(available_records);
        let total_values = record_count * record_len;
        let mut shape = self.shape.clone();
        if repeat.is_some() && record_count > 1 {
            shape.push(record_count);
        } else if record_len != total_values {
            shape = vec![total_values, 1];
        }
        let tensor =
            if let Some(storage) = decode_integer_storage(&self.dtype, bytes, total_values) {
                Tensor::new_integer(storage, shape)
            } else if self.dtype.eq_ignore_ascii_case("single") {
                let values = bytes[..total_values * element_size]
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
                    .collect();
                Tensor::from_f32(values, shape)
            } else {
                let values = bytes[..total_values * element_size]
                    .chunks_exact(8)
                    .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
                    .collect();
                Tensor::new(values, shape)
            }
            .map_err(|error| compat_error("memmapfile", error))?;
        let tensor = Value::Tensor(tensor);
        Ok(tensor)
    }

    fn decode_table(&self, bytes: &[u8], repeat: Option<usize>) -> BuiltinResult<Value> {
        let record_bytes = self.validate_layout(repeat)?;
        let available_records = bytes.len() / record_bytes;
        let records = repeat.unwrap_or(available_records).min(available_records);
        let mut elements = Vec::with_capacity(records);
        for record in bytes.chunks_exact(record_bytes).take(records) {
            let mut structure = StructValue::new();
            let mut cursor = 0usize;
            let mut fields = std::iter::once(MemmapField {
                dtype: self.dtype.clone(),
                shape: self.shape.clone(),
                name: self.field.clone().expect("format table has a first field"),
            })
            .chain(self.additional_fields.iter().cloned());
            for field in &mut fields {
                let field_bytes = memmap_field_bytes(&field.dtype, &field.shape)?;
                let value = MemmapFormat {
                    dtype: field.dtype,
                    shape: field.shape,
                    field: None,
                    additional_fields: Vec::new(),
                }
                .decode(&record[cursor..cursor + field_bytes], Some(1))?;
                structure.insert(field.name, value);
                cursor += field_bytes;
            }
            elements.push(structure);
        }
        if records == 0 {
            let names =
                std::iter::once(self.field.clone().expect("format table has a first field"))
                    .chain(
                        self.additional_fields
                            .iter()
                            .map(|field| field.name.clone()),
                    )
                    .collect();
            return Ok(Value::StructArray(
                runmat_value::StructArray::empty(names, vec![0, 1])
                    .map_err(|error| compat_error("memmapfile", error))?,
            ));
        }
        runmat_value::StructArray::normalize(elements, vec![records, 1])
            .map_err(|error| compat_error("memmapfile", error))
    }
}

fn decode_integer_storage(dtype: &str, bytes: &[u8], value_count: usize) -> Option<IntegerStorage> {
    let bytes = &bytes[..value_count.checked_mul(dtype_size(dtype).ok()?)?];
    Some(match dtype.to_ascii_lowercase().as_str() {
        "int8" => IntegerStorage::I8(bytes.iter().map(|value| *value as i8).collect()),
        "uint8" | "char" => IntegerStorage::U8(bytes.to_vec()),
        "int16" => IntegerStorage::I16(
            bytes
                .chunks_exact(2)
                .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                .collect(),
        ),
        "uint16" => IntegerStorage::U16(
            bytes
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                .collect(),
        ),
        "int32" => IntegerStorage::I32(
            bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
        ),
        "uint32" => IntegerStorage::U32(
            bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
        ),
        "int64" => IntegerStorage::I64(
            bytes
                .chunks_exact(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect(),
        ),
        "uint64" => IntegerStorage::U64(
            bytes
                .chunks_exact(8)
                .map(|chunk| {
                    u64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect(),
        ),
        _ => return None,
    })
}
