use super::*;

pub(super) fn shape_or_scalar(shape: Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        vec![1, 1]
    } else {
        shape
    }
}

pub(super) fn matlab_value_from_f64(data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let shape = shape_or_scalar(shape);
    if data.len() == 1 {
        Ok(Value::Num(data[0]))
    } else {
        Tensor::new(data, shape)
            .map(Value::Tensor)
            .map_err(|msg| hdf5_error(H5READ_NAME, &ERR_IO, msg))
    }
}

pub(super) fn matlab_value_from_integer(
    storage: IntegerStorage,
    shape: Vec<usize>,
) -> BuiltinResult<Value> {
    let shape = shape_or_scalar(shape);
    if storage.len() == 1 {
        return Ok(Value::Int(integer_storage_scalar(&storage, 0)));
    }
    Tensor::new_integer(storage, shape)
        .map(Value::Tensor)
        .map_err(|msg| hdf5_error(H5READ_NAME, &ERR_IO, msg))
}

pub(super) fn integer_storage_from_scalar(value: IntValue) -> IntegerStorage {
    match value {
        IntValue::I8(value) => IntegerStorage::I8(vec![value]),
        IntValue::I16(value) => IntegerStorage::I16(vec![value]),
        IntValue::I32(value) => IntegerStorage::I32(vec![value]),
        IntValue::I64(value) => IntegerStorage::I64(vec![value]),
        IntValue::U8(value) => IntegerStorage::U8(vec![value]),
        IntValue::U16(value) => IntegerStorage::U16(vec![value]),
        IntValue::U32(value) => IntegerStorage::U32(vec![value]),
        IntValue::U64(value) => IntegerStorage::U64(vec![value]),
    }
}

pub(super) fn integer_storage_scalar(storage: &IntegerStorage, index: usize) -> IntValue {
    match storage {
        IntegerStorage::I8(values) => IntValue::I8(values[index]),
        IntegerStorage::I16(values) => IntValue::I16(values[index]),
        IntegerStorage::I32(values) => IntValue::I32(values[index]),
        IntegerStorage::I64(values) => IntValue::I64(values[index]),
        IntegerStorage::U8(values) => IntValue::U8(values[index]),
        IntegerStorage::U16(values) => IntValue::U16(values[index]),
        IntegerStorage::U32(values) => IntValue::U32(values[index]),
        IntegerStorage::U64(values) => IntValue::U64(values[index]),
    }
}

pub(super) fn matlab_value_from_bool(data: Vec<bool>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let bits: Vec<u8> = data.into_iter().map(u8::from).collect();
    let shape = shape_or_scalar(shape);
    if bits.len() == 1 {
        Ok(Value::Bool(bits[0] != 0))
    } else {
        LogicalArray::new(bits, shape)
            .map(Value::LogicalArray)
            .map_err(|msg| hdf5_error(H5READ_NAME, &ERR_IO, msg))
    }
}

pub(super) fn matlab_value_from_strings(
    data: Vec<String>,
    shape: Vec<usize>,
) -> BuiltinResult<Value> {
    let shape = shape_or_scalar(shape);
    if data.len() == 1 {
        Ok(Value::string(data[0].clone()))
    } else {
        StringArray::new(data, shape)
            .map(Value::StringArray)
            .map_err(|msg| hdf5_error(H5READ_NAME, &ERR_IO, msg))
    }
}

pub(super) fn numeric_vector_usize(
    value: &Value,
    builtin: &'static str,
    label: &str,
) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Num(v) => Ok(vec![numeric_usize_entry(*v, builtin, label)?]),
        Value::Int(v) => Ok(vec![int_usize_entry(v, builtin, label)?]),
        Value::Tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                storage
                    .exact_values()
                    .iter()
                    .map(|value| int_usize_entry(value, builtin, label))
                    .collect()
            } else {
                (0..checked_shape_len(t.shape(), builtin)?)
                    .map(|index| {
                        let value = match t
                            .numeric_value_at(index)
                            .expect("numeric vector index is in bounds")
                        {
                            NumericScalar::F64(value) => value,
                            NumericScalar::F32(value) => f64::from(value),
                            NumericScalar::I8(_)
                            | NumericScalar::I16(_)
                            | NumericScalar::I32(_)
                            | NumericScalar::I64(_)
                            | NumericScalar::U8(_)
                            | NumericScalar::U16(_)
                            | NumericScalar::U32(_)
                            | NumericScalar::U64(_) => {
                                unreachable!("integer vectors use the exact parser")
                            }
                        };
                        numeric_usize_entry(value, builtin, label)
                    })
                    .collect()
            }
        }
        _ => Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: expected {label} as a numeric vector"),
        )),
    }
}

pub(super) fn int_usize_entry(
    value: &IntValue,
    builtin: &'static str,
    label: &str,
) -> BuiltinResult<usize> {
    let Some(parsed) = value.try_to_usize() else {
        return Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: {label} entries must be positive finite integers"),
        ));
    };
    if parsed == 0 {
        return Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: {label} entries must be positive finite integers"),
        ));
    }
    Ok(parsed)
}

pub(super) fn numeric_usize_entry(
    value: f64,
    builtin: &'static str,
    label: &str,
) -> BuiltinResult<usize> {
    if !value.is_finite()
        || value < 1.0
        || value > usize::MAX as f64
        || (usize::BITS == 64 && value == usize::MAX as f64)
        || value.fract() != 0.0
    {
        Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: {label} entries must be positive finite integers"),
        ))
    } else {
        Ok(value as usize)
    }
}

pub(super) fn col_major_to_row_major<T: Clone>(data: &[T], shape: &[usize]) -> Vec<T> {
    let len = data.len();
    if shape.len() <= 1 || len <= 1 {
        return data.to_vec();
    }
    let mut out = Vec::with_capacity(len);
    for row_linear in 0..len {
        let idx = row_major_index_to_multi(row_linear, shape);
        let col_offset = multi_to_col_major(&idx, shape);
        out.push(data[col_offset].clone());
    }
    out
}

pub(super) fn row_major_to_col_major<T: Clone>(data: &[T], shape: &[usize]) -> Vec<T> {
    let len = data.len();
    if shape.len() <= 1 || len <= 1 {
        return data.to_vec();
    }
    let mut out = Vec::with_capacity(len);
    for col_linear in 0..len {
        let idx = col_major_index_to_multi(col_linear, shape);
        let row_offset = multi_to_row_major(&idx, shape);
        out.push(data[row_offset].clone());
    }
    out
}

pub(super) fn col_major_index_to_multi(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = Vec::with_capacity(shape.len());
    for &dim in shape {
        idx.push(linear % dim);
        linear /= dim;
    }
    idx
}

pub(super) fn row_major_index_to_multi(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0; shape.len()];
    for dim in (0..shape.len()).rev() {
        idx[dim] = linear % shape[dim];
        linear /= shape[dim];
    }
    idx
}

pub(super) fn multi_to_col_major(idx: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut offset = 0usize;
    for (&i, &dim) in idx.iter().zip(shape) {
        offset += i * stride;
        stride *= dim;
    }
    offset
}

pub(super) fn multi_to_row_major(idx: &[usize], shape: &[usize]) -> usize {
    let mut offset = 0usize;
    for (&i, &dim) in idx.iter().zip(shape) {
        offset = offset * dim + i;
    }
    offset
}

#[derive(Debug, Clone)]
pub(super) struct MatlabSelection {
    pub(super) start: Vec<usize>,
    pub(super) count: Vec<usize>,
    pub(super) stride: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(super) enum SelectionControls {
    Whole,
    Range {
        start: Vec<usize>,
        count: Vec<usize>,
        stride: Option<Vec<usize>>,
    },
}

pub(super) fn parse_selection_controls(
    rest: crate::call::FunctionArgumentView<'_>,
    builtin: &'static str,
) -> BuiltinResult<SelectionControls> {
    match rest.len() {
        0 => Ok(SelectionControls::Whole),
        2 | 3 => {
            let start = numeric_vector_usize(&rest[0], builtin, "start")?;
            let count = numeric_vector_usize(&rest[1], builtin, "count")?;
            let stride = if rest.len() == 3 {
                Some(numeric_vector_usize(&rest[2], builtin, "stride")?)
            } else {
                None
            };
            Ok(SelectionControls::Range {
                start,
                count,
                stride,
            })
        }
        _ => Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: expected start/count or start/count/stride selection"),
        )),
    }
}

pub(super) fn prepare_host_selection_controls(
    rest: crate::call::FunctionArgumentView<'_>,
    builtin: &'static str,
) -> BuiltinResult<Option<SelectionControls>> {
    if !matches!(rest.len(), 0 | 2 | 3) {
        return Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: expected start/count or start/count/stride selection"),
        ));
    }
    if !rest
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        return parse_selection_controls(rest, builtin).map(Some);
    }
    for (index, value) in rest.iter().enumerate() {
        if matches!(value, Value::GpuTensor(_)) {
            continue;
        }
        let label = match index {
            0 => "start",
            1 => "count",
            _ => "stride",
        };
        numeric_vector_usize(value, builtin, label)?;
    }
    Ok(None)
}

impl SelectionControls {
    pub(super) fn resolve(
        self,
        builtin: &'static str,
        source_shape: &[usize],
    ) -> BuiltinResult<Option<MatlabSelection>> {
        match self {
            Self::Whole => Ok(None),
            Self::Range {
                mut start,
                count,
                stride,
            } => {
                let rank = source_shape.len();
                let stride = stride.unwrap_or_else(|| vec![1; rank]);
                if start.len() != rank || count.len() != rank || stride.len() != rank {
                    return Err(hdf5_error(
                        builtin,
                        &ERR_ARGUMENT,
                        format!("{builtin}: start, count, and stride must match dataset rank"),
                    ));
                }
                for item in &mut start {
                    *item -= 1;
                }
                for dim in 0..rank {
                    if count[dim] == 0 {
                        return Err(hdf5_error(
                            builtin,
                            &ERR_ARGUMENT,
                            format!("{builtin}: count entries must be positive"),
                        ));
                    }
                    let last = count[dim]
                        .checked_sub(1)
                        .and_then(|count| count.checked_mul(stride[dim]))
                        .and_then(|offset| start[dim].checked_add(offset))
                        .ok_or_else(|| {
                            hdf5_error(
                                builtin,
                                &ERR_ARGUMENT,
                                format!("{builtin}: selection exceeds platform limits"),
                            )
                        })?;
                    if last >= source_shape[dim] {
                        return Err(hdf5_error(
                            builtin,
                            &ERR_ARGUMENT,
                            format!("{builtin}: selection exceeds dataset bounds"),
                        ));
                    }
                }
                checked_shape_len(&count, builtin)?;
                Ok(Some(MatlabSelection {
                    start,
                    count,
                    stride,
                }))
            }
        }
    }
}

pub(super) fn checked_shape_len(shape: &[usize], builtin: &'static str) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            hdf5_error(
                builtin,
                &ERR_ARGUMENT,
                format!("{builtin}: shape exceeds platform limits"),
            )
        })
    })
}
