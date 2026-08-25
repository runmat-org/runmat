use std::fmt;

use runmat_value::{
    CellArray, CharArray, ComplexStorage, ComplexTensor, IntegerComplexStorage, IntegerStorage,
    LogicalArray, NumericScalar, NumericStorage, SparseTensor, StructValue, Tensor, Value,
};

use crate::mxarray::{
    MxApiMode, MxArray, MxArrayData, MxInterleavedStorage, MxNumeric, MxSparse, MxSparseValues,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MxConversionError {
    pub message: String,
}

impl MxConversionError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for MxConversionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for MxConversionError {}

pub fn value_to_mx(value: &Value, mode: MxApiMode) -> Result<MxArray, MxConversionError> {
    match value {
        Value::Num(value) => numeric_to_mx(NumericStorage::F64(vec![*value]), vec![1, 1]),
        Value::Int(value) => numeric_to_mx(
            NumericStorage::from_integer_storage(IntegerStorage::from_scalar(value.clone())),
            vec![1, 1],
        ),
        Value::Complex(real, imag) => complex_to_mx(
            NumericStorage::F64(vec![*real]),
            NumericStorage::F64(vec![*imag]),
            vec![1, 1],
            mode,
        ),
        Value::Bool(value) => {
            MxArray::logical(vec![u8::from(*value)], vec![1, 1]).map_err(MxConversionError::new)
        }
        Value::LogicalArray(value) => MxArray::logical(value.data.clone(), value.shape.clone())
            .map_err(MxConversionError::new),
        Value::String(value) => string_to_mx(value),
        Value::CharArray(value) => char_to_mx(value),
        Value::Tensor(value) => numeric_to_mx(
            value
                .clone()
                .into_numeric_storage()
                .map_err(MxConversionError::new)?,
            value.shape.clone(),
        ),
        Value::ComplexTensor(value) => {
            let (real, imag) = complex_components(value)?;
            complex_to_mx(real, imag, value.shape.clone(), mode)
        }
        Value::SparseTensor(value) => sparse_to_mx(value),
        Value::Cell(value) => cell_to_mx(value, mode),
        Value::Struct(value) => struct_to_mx(value, mode),
        other => Err(MxConversionError::new(format!(
            "{} values do not have a C Matrix API representation",
            value_kind(other)
        ))),
    }
}

pub fn value_from_mx(value: &MxArray) -> Result<Value, MxConversionError> {
    match value.data() {
        MxArrayData::Numeric(numeric) => numeric_from_mx(numeric, value.shape()),
        MxArrayData::Interleaved(interleaved) => {
            let (real, imag) = interleaved.values.components();
            complex_from_components(real, imag, value.shape())
        }
        MxArrayData::Logical(values) => {
            if values.len() == 1 {
                Ok(Value::Bool(values[0] != 0))
            } else {
                LogicalArray::new(values.clone(), value.shape().to_vec())
                    .map(Value::LogicalArray)
                    .map_err(MxConversionError::new)
            }
        }
        MxArrayData::Char(values) => char_from_mx(values, value.shape()),
        MxArrayData::Cell(values) => cell_from_mx(values, value.shape()),
        MxArrayData::Struct { fields, values } => struct_from_mx(fields, values, value.shape()),
        MxArrayData::Sparse(value) => sparse_from_mx(value),
    }
}

fn numeric_to_mx(storage: NumericStorage, shape: Vec<usize>) -> Result<MxArray, MxConversionError> {
    MxArray::numeric(storage, shape, None).map_err(MxConversionError::new)
}

fn complex_to_mx(
    real: NumericStorage,
    imag: NumericStorage,
    shape: Vec<usize>,
    mode: MxApiMode,
) -> Result<MxArray, MxConversionError> {
    match mode {
        MxApiMode::SeparateComplex => {
            MxArray::numeric(real, shape, Some(imag)).map_err(MxConversionError::new)
        }
        MxApiMode::InterleavedComplex => MxInterleavedStorage::from_components(&real, &imag)
            .and_then(|values| MxArray::interleaved(values, shape))
            .map_err(MxConversionError::new),
    }
}

fn complex_components(
    value: &ComplexTensor,
) -> Result<(NumericStorage, NumericStorage), MxConversionError> {
    match value.complex_storage() {
        ComplexStorage::F64(values) => Ok((
            NumericStorage::F64(values.iter().map(|value| value.0).collect()),
            NumericStorage::F64(values.iter().map(|value| value.1).collect()),
        )),
        ComplexStorage::F32(values) => Ok((
            NumericStorage::F32(values.iter().map(|value| value.0).collect()),
            NumericStorage::F32(values.iter().map(|value| value.1).collect()),
        )),
        ComplexStorage::Integer(values) => Ok((
            NumericStorage::from_integer_storage(values.real.clone()),
            NumericStorage::from_integer_storage(values.imag.clone()),
        )),
    }
}

fn string_to_mx(value: &str) -> Result<MxArray, MxConversionError> {
    let encoded = value.encode_utf16().collect::<Vec<_>>();
    MxArray::character(encoded.clone(), vec![1, encoded.len()]).map_err(MxConversionError::new)
}

fn char_to_mx(value: &CharArray) -> Result<MxArray, MxConversionError> {
    let mut encoded = Vec::with_capacity(value.data.len());
    for character in value.to_column_major() {
        let codepoint = u32::from(character);
        let unit = u16::try_from(codepoint).map_err(|_| {
            MxConversionError::new(
                "RunMat character arrays containing non-BMP scalars cannot preserve their shape in the UTF-16 C Matrix API",
            )
        })?;
        encoded.push(unit);
    }
    MxArray::character(encoded, value.shape().to_vec()).map_err(MxConversionError::new)
}

fn sparse_to_mx(value: &SparseTensor) -> Result<MxArray, MxConversionError> {
    let values = if value.is_logical() {
        MxSparseValues::Logical(vec![1; value.nnz()])
    } else {
        let dtype = value
            .numeric_dtype()
            .ok_or_else(|| MxConversionError::new("sparse value has no numeric class"))?;
        let mut storage = NumericStorage::zeros(dtype, value.nnz());
        for index in 0..value.nnz() {
            storage
                .set_value(
                    index,
                    value.numeric_value_at(index).ok_or_else(|| {
                        MxConversionError::new("sparse numeric storage is incomplete")
                    })?,
                )
                .map_err(MxConversionError::new)?;
        }
        MxSparseValues::Numeric(storage)
    };
    MxArray::sparse(MxSparse {
        rows: value.rows,
        cols: value.cols,
        col_ptrs: value.col_ptrs.clone(),
        row_indices: value.row_indices.clone(),
        values,
        nzmax: value.nnz(),
    })
    .map_err(MxConversionError::new)
}

fn cell_to_mx(value: &CellArray, mode: MxApiMode) -> Result<MxArray, MxConversionError> {
    let column_major = value.to_column_major();
    if let Some(fields) = uniform_struct_fields(&column_major) {
        let mut values = Vec::with_capacity(fields.len() * column_major.len());
        for field in &fields {
            for element in &column_major {
                let Value::Struct(element) = element else {
                    unreachable!("uniform struct check")
                };
                values.push(
                    element
                        .fields
                        .get(field)
                        .map(|value| value_to_mx(value, mode).map(Box::new))
                        .transpose()?,
                );
            }
        }
        return MxArray::structure(fields, values, value.shape.clone())
            .map_err(MxConversionError::new);
    }
    let values = column_major
        .iter()
        .map(|value| value_to_mx(value, mode).map(Box::new).map(Some))
        .collect::<Result<Vec<_>, _>>()?;
    MxArray::cell(values, value.shape.clone()).map_err(MxConversionError::new)
}

fn struct_to_mx(value: &StructValue, mode: MxApiMode) -> Result<MxArray, MxConversionError> {
    let fields = value.field_names().cloned().collect::<Vec<_>>();
    let values = fields
        .iter()
        .map(|field| {
            value_to_mx(
                value
                    .fields
                    .get(field)
                    .expect("field name came from the same struct"),
                mode,
            )
            .map(Box::new)
            .map(Some)
        })
        .collect::<Result<Vec<_>, _>>()?;
    MxArray::structure(fields, values, vec![1, 1]).map_err(MxConversionError::new)
}

fn uniform_struct_fields(values: &[Value]) -> Option<Vec<String>> {
    let Value::Struct(first) = values.first()? else {
        return None;
    };
    let fields = first.field_names().cloned().collect::<Vec<_>>();
    values
        .iter()
        .all(|value| {
            let Value::Struct(value) = value else {
                return false;
            };
            value.field_names().eq(fields.iter())
        })
        .then_some(fields)
}

fn numeric_from_mx(value: &MxNumeric, shape: &[usize]) -> Result<Value, MxConversionError> {
    if let Some(imag) = &value.imag {
        return complex_from_components(value.real.clone(), imag.clone(), shape);
    }
    if value.real.len() == 1 {
        return scalar_from_numeric(value.real.value_at(0).unwrap(), shape);
    }
    Tensor::from_numeric_storage(value.real.clone(), shape.to_vec())
        .map(Value::Tensor)
        .map_err(MxConversionError::new)
}

fn scalar_from_numeric(value: NumericScalar, shape: &[usize]) -> Result<Value, MxConversionError> {
    match value {
        NumericScalar::F64(value) => Ok(Value::Num(value)),
        NumericScalar::F32(value) => Tensor::from_f32(vec![value], shape.to_vec())
            .map(Value::Tensor)
            .map_err(MxConversionError::new),
        value => Ok(Value::Int(
            value
                .into_int_value()
                .expect("non-floating numeric scalar is integer"),
        )),
    }
}

fn complex_from_components(
    real: NumericStorage,
    imag: NumericStorage,
    shape: &[usize],
) -> Result<Value, MxConversionError> {
    if real.numeric_dtype() != imag.numeric_dtype() || real.len() != imag.len() {
        return Err(MxConversionError::new(
            "complex components have different classes or lengths",
        ));
    }
    if real.len() == 1 && matches!(real, NumericStorage::F64(_)) {
        let NumericScalar::F64(real) = real.value_at(0).unwrap() else {
            unreachable!()
        };
        let NumericScalar::F64(imag) = imag.value_at(0).unwrap() else {
            unreachable!()
        };
        return Ok(Value::Complex(real, imag));
    }
    let storage = match (real, imag) {
        (NumericStorage::F64(real), NumericStorage::F64(imag)) => {
            ComplexStorage::F64(real.into_iter().zip(imag).collect())
        }
        (NumericStorage::F32(real), NumericStorage::F32(imag)) => {
            ComplexStorage::F32(real.into_iter().zip(imag).collect())
        }
        (real, imag) => ComplexStorage::Integer(
            IntegerComplexStorage::new(
                real.into_integer_storage()
                    .map_err(|_| MxConversionError::new("real complex storage is not integer"))?,
                imag.into_integer_storage().map_err(|_| {
                    MxConversionError::new("imaginary complex storage is not integer")
                })?,
            )
            .map_err(MxConversionError::new)?,
        ),
    };
    ComplexTensor::from_complex_storage(storage, shape.to_vec())
        .map(Value::ComplexTensor)
        .map_err(MxConversionError::new)
}

fn char_from_mx(values: &[u16], shape: &[usize]) -> Result<Value, MxConversionError> {
    let mut characters = Vec::with_capacity(values.len());
    for value in values {
        let character = char::from_u32(u32::from(*value)).ok_or_else(|| {
            MxConversionError::new(
                "UTF-16 surrogate code units cannot be represented by the current RunMat character scalar",
            )
        })?;
        characters.push(character);
    }
    CharArray::from_column_major(characters, shape.to_vec())
        .map(Value::CharArray)
        .map_err(MxConversionError::new)
}

fn cell_from_mx(
    values: &[Option<Box<MxArray>>],
    shape: &[usize],
) -> Result<Value, MxConversionError> {
    let values = values
        .iter()
        .map(|value| {
            value
                .as_deref()
                .map(value_from_mx)
                .transpose()
                .map(|value| value.unwrap_or_else(|| Value::Tensor(Tensor::zeros(vec![0, 0]))))
        })
        .collect::<Result<Vec<_>, _>>()?;
    CellArray::from_column_major(values, shape.to_vec())
        .map(Value::Cell)
        .map_err(MxConversionError::new)
}

fn struct_from_mx(
    fields: &[String],
    values: &[Option<Box<MxArray>>],
    shape: &[usize],
) -> Result<Value, MxConversionError> {
    let numel = shape.iter().product::<usize>();
    let mut structures = Vec::with_capacity(numel);
    for element in 0..numel {
        let mut structure = StructValue::new();
        for (field_index, field) in fields.iter().enumerate() {
            let value = values[field_index * numel + element]
                .as_deref()
                .map(value_from_mx)
                .transpose()?
                .unwrap_or_else(|| Value::Tensor(Tensor::zeros(vec![0, 0])));
            structure.insert(field.clone(), value);
        }
        structures.push(Value::Struct(structure));
    }
    if numel == 1 {
        return Ok(structures.pop().expect("one struct element"));
    }
    CellArray::from_column_major(structures, shape.to_vec())
        .map(Value::Cell)
        .map_err(MxConversionError::new)
}

fn sparse_from_mx(value: &MxSparse) -> Result<Value, MxConversionError> {
    let nnz = value.col_ptrs.last().copied().unwrap_or(0);
    if nnz > value.nzmax {
        return Err(MxConversionError::new(
            "sparse column pointers exceed allocated nzmax",
        ));
    }
    let row_indices = value.row_indices[..nnz].to_vec();
    let result = match &value.values {
        MxSparseValues::Logical(_) => {
            SparseTensor::new_logical(value.rows, value.cols, value.col_ptrs.clone(), row_indices)
        }
        MxSparseValues::Numeric(NumericStorage::F64(values)) => SparseTensor::new(
            value.rows,
            value.cols,
            value.col_ptrs.clone(),
            row_indices,
            values[..nnz].to_vec(),
        ),
        MxSparseValues::Numeric(NumericStorage::F32(values)) => SparseTensor::new_f32(
            value.rows,
            value.cols,
            value.col_ptrs.clone(),
            row_indices,
            values[..nnz].to_vec(),
        ),
        MxSparseValues::Numeric(values) => SparseTensor::new_integer(
            value.rows,
            value.cols,
            value.col_ptrs.clone(),
            row_indices,
            truncate_numeric(values, nnz)
                .into_integer_storage()
                .map_err(|_| MxConversionError::new("sparse numeric class is not integer"))?,
        ),
    };
    result
        .map(Value::SparseTensor)
        .map_err(MxConversionError::new)
}

fn truncate_numeric(values: &NumericStorage, len: usize) -> NumericStorage {
    match values {
        NumericStorage::F64(values) => NumericStorage::F64(values[..len].to_vec()),
        NumericStorage::F32(values) => NumericStorage::F32(values[..len].to_vec()),
        NumericStorage::I8(values) => NumericStorage::I8(values[..len].to_vec()),
        NumericStorage::I16(values) => NumericStorage::I16(values[..len].to_vec()),
        NumericStorage::I32(values) => NumericStorage::I32(values[..len].to_vec()),
        NumericStorage::I64(values) => NumericStorage::I64(values[..len].to_vec()),
        NumericStorage::U8(values) => NumericStorage::U8(values[..len].to_vec()),
        NumericStorage::U16(values) => NumericStorage::U16(values[..len].to_vec()),
        NumericStorage::U32(values) => NumericStorage::U32(values[..len].to_vec()),
        NumericStorage::U64(values) => NumericStorage::U64(values[..len].to_vec()),
    }
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::StringArray(_) => "string array",
        Value::Symbolic(_) | Value::SymbolicArray(_) => "symbolic",
        Value::GpuTensor(_) => "GPU-resident",
        Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_) => "object",
        Value::Listener(_) => "listener",
        Value::OutputList(_) => "output-list",
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_) => "callable",
        Value::MException(_) => "exception",
        Value::Future(_) | Value::Task(_) | Value::Pool(_) | Value::Job(_) => "execution-handle",
        Value::Foreign(_) => "foreign",
        _ => "unsupported",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntValue, IntegerStorage};

    #[test]
    fn all_numeric_classes_round_trip_exactly_including_wide_uint64() {
        let cases = vec![
            NumericStorage::F64(vec![1.25, -2.5]),
            NumericStorage::F32(vec![1.25, -2.5]),
            NumericStorage::I8(vec![i8::MIN, i8::MAX]),
            NumericStorage::I16(vec![i16::MIN, i16::MAX]),
            NumericStorage::I32(vec![i32::MIN, i32::MAX]),
            NumericStorage::I64(vec![i64::MIN, i64::MAX]),
            NumericStorage::U8(vec![0, u8::MAX]),
            NumericStorage::U16(vec![0, u16::MAX]),
            NumericStorage::U32(vec![0, u32::MAX]),
            NumericStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ];
        for storage in cases {
            let original = Value::Tensor(
                Tensor::from_numeric_storage(storage, vec![2, 1]).expect("typed tensor"),
            );
            for mode in [MxApiMode::SeparateComplex, MxApiMode::InterleavedComplex] {
                let boundary = value_to_mx(&original, mode).expect("convert to mxArray");
                assert_eq!(value_from_mx(&boundary).expect("convert back"), original);
            }
        }
    }

    #[test]
    fn complex_integer_storage_round_trips_in_both_api_modes() {
        let original = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                    IntegerStorage::U64(vec![1, 2]),
                )
                .unwrap(),
                vec![1, 2],
            )
            .unwrap(),
        );
        for mode in [MxApiMode::SeparateComplex, MxApiMode::InterleavedComplex] {
            let boundary = value_to_mx(&original, mode).unwrap();
            assert_eq!(value_from_mx(&boundary).unwrap(), original);
        }
    }

    #[test]
    fn cell_backed_struct_arrays_use_field_major_mx_storage() {
        let mut first = StructValue::new();
        first.insert("id", Value::Int(IntValue::U64(u64::MAX)));
        let mut second = StructValue::new();
        second.insert("id", Value::Int(IntValue::U64(9_007_199_254_740_993)));
        let original = Value::Cell(
            CellArray::new(vec![Value::Struct(first), Value::Struct(second)], 1, 2).unwrap(),
        );
        let boundary = value_to_mx(&original, MxApiMode::InterleavedComplex).unwrap();
        assert!(matches!(boundary.data(), MxArrayData::Struct { .. }));
        assert_eq!(value_from_mx(&boundary).unwrap(), original);
    }
}
