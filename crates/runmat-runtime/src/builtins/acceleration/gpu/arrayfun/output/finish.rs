use crate::BuiltinResult;
use runmat_builtins::ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE;
use runmat_value::{CharArray, ComplexTensor, IntegerComplexStorage, LogicalArray, Tensor, Value};

use super::super::error::{arrayfun_error_with_detail, arrayfun_internal};
use super::UniformCollector;

impl UniformCollector {
    pub(in crate::builtins::acceleration::gpu::arrayfun) fn finish(
        self,
        shape: &[usize],
    ) -> BuiltinResult<Value> {
        match self {
            UniformCollector::Pending => {
                let total = shape.iter().product();
                let tensor = Tensor::new(vec![0.0; total], shape.to_vec())
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::Tensor(tensor))
            }
            UniformCollector::F64(data) => {
                let tensor = Tensor::new(data, shape.to_vec())
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::Tensor(tensor))
            }
            UniformCollector::F32(data) => {
                let tensor = Tensor::from_f32(data, shape.to_vec())
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::Tensor(tensor))
            }
            UniformCollector::Integer { prototype, values } => {
                let storage = prototype
                    .from_same_class_values(values)
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                let tensor = Tensor::new_integer(storage, shape.to_vec())
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::Tensor(tensor))
            }
            UniformCollector::Logical(bits) => {
                let logical = LogicalArray::new(bits, shape.to_vec())
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::LogicalArray(logical))
            }
            UniformCollector::ComplexF64(entries) => {
                let tensor = ComplexTensor::new(entries, shape.to_vec())
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::ComplexTensor(tensor))
            }
            UniformCollector::ComplexF32(entries) => {
                let tensor = ComplexTensor::from_f32(entries, shape.to_vec())
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::ComplexTensor(tensor))
            }
            UniformCollector::IntegerComplex {
                real_prototype,
                imag_prototype,
                real_values,
                imag_values,
            } => {
                let real = real_prototype
                    .from_same_class_values(real_values)
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                let imag = imag_prototype
                    .from_same_class_values(imag_values)
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                let storage = IntegerComplexStorage::new(real, imag)
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                let tensor = ComplexTensor::new_integer(storage, shape.to_vec())
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::ComplexTensor(tensor))
            }
            UniformCollector::Char(chars) => {
                let normalized_shape = if shape.is_empty() {
                    vec![1, 1]
                } else {
                    shape.to_vec()
                };

                if normalized_shape.len() > 2 {
                    return Err(arrayfun_error_with_detail(
                        &ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE,
                        "character outputs with UniformOutput=true must be 2-D",
                    ));
                }

                let rows = normalized_shape.first().copied().unwrap_or(1);
                let cols = normalized_shape.get(1).copied().unwrap_or(1);
                let expected = rows.checked_mul(cols).ok_or_else(|| {
                    arrayfun_internal("arrayfun: character output size exceeds platform limits")
                })?;

                if expected != chars.len() {
                    return Err(arrayfun_error_with_detail(
                        &ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE,
                        "callback returned the wrong number of characters",
                    ));
                }

                let mut row_major = vec!['\0'; expected];
                for col in 0..cols {
                    for row in 0..rows {
                        let col_major_idx = row + col * rows;
                        let row_major_idx = row * cols + col;
                        row_major[row_major_idx] = chars[col_major_idx];
                    }
                }

                let array = CharArray::new(row_major, rows, cols)
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::CharArray(array))
            }
        }
    }
}
