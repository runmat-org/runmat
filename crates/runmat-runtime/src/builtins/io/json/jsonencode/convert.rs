use super::*;

pub(super) fn value_to_json(
    value: &Value,
    options: &JsonEncodeOptions,
) -> BuiltinResult<JsonValue> {
    match value {
        Value::Num(n) => number_to_json(*n, options),
        Value::Int(i) => Ok(JsonValue::Number(int_to_number(i))),
        Value::Bool(b) => Ok(JsonValue::Bool(*b)),
        Value::LogicalArray(logical) => logical_array_to_json(logical, options),
        Value::Tensor(tensor) => tensor_to_json(tensor, options),
        Value::SparseTensor(sparse) => {
            let total_elements = sparse.rows.checked_mul(sparse.cols).ok_or_else(|| {
                jsonencode_error_with(
                    &JSONENCODE_ERROR_INTERNAL,
                    "jsonencode: sparse matrix dimensions overflow",
                )
            })?;
            if total_elements > 10_000_000 {
                return Err(jsonencode_error_with(
                    &JSONENCODE_ERROR_INTERNAL,
                    format!("jsonencode: cannot densify sparse tensor {}x{} ({} elements exceeds safe threshold)", sparse.rows, sparse.cols, total_elements),
                ));
            }
            if sparse.is_logical() {
                let dense = sparse.to_dense_logical().map_err(|err| {
                    jsonencode_error_with(&JSONENCODE_ERROR_INTERNAL, format!("jsonencode: {err}"))
                })?;
                return logical_array_to_json(&dense, options);
            }
            let dense = sparse.to_dense().map_err(|err| {
                jsonencode_error_with(&JSONENCODE_ERROR_INTERNAL, format!("jsonencode: {err}"))
            })?;
            tensor_to_json(&dense, options)
        }
        Value::Complex(re, im) => complex_scalar_to_json(*re, *im, options),
        Value::ComplexTensor(ct) => complex_tensor_to_json(ct, options),
        Value::String(s) => Ok(string_element_to_json(s)),
        Value::Symbolic(expr) => Ok(JsonValue::String(expr.to_string().into())),
        Value::SymbolicArray(array) => symbolic_array_to_json(array, options),
        Value::StringArray(sa) => string_array_to_json(sa, options),
        Value::CharArray(ca) => char_array_to_json(ca, options),
        Value::Struct(sv) => struct_to_json(sv, options),
        Value::StructArray(array) => {
            let dimensions = (0..array.shape().len()).collect::<Vec<_>>();
            build_strided_array(array.shape(), &dimensions, |offset| {
                let element = array.get_linear(offset).ok_or_else(|| {
                    jsonencode_error_with(
                        &JSONENCODE_ERROR_INTERNAL,
                        "jsonencode: structure-array offset is out of bounds",
                    )
                })?;
                struct_fields_to_json(element.fields.iter(), options)
            })
        }
        Value::Cell(ca) => cell_array_to_json(ca, options),
        Value::ObjectArray(array) => array
            .data()
            .iter()
            .map(|value| value_to_json(value, options))
            .collect::<BuiltinResult<Vec<_>>>()
            .map(JsonValue::Array),
        Value::Object(obj) => object_to_json(obj, options),
        Value::GpuTensor(_) => Err(jsonencode_error(&JSONENCODE_ERROR_UNEXPECTED_GPU)),
        Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
        | Value::Foreign(_)
        | Value::OutputList(_) => Err(jsonencode_error(&JSONENCODE_ERROR_UNSUPPORTED_TYPE)),
    }
}

fn string_element_to_json(value: &runmat_value::StringElement) -> JsonValue {
    match value {
        runmat_value::StringElement::Text(text) => JsonValue::String(text.clone()),
        runmat_value::StringElement::Missing => JsonValue::Null,
    }
}

fn int_to_number(value: &IntValue) -> JsonNumber {
    match value {
        IntValue::I8(v) => JsonNumber::I64(*v as i64),
        IntValue::I16(v) => JsonNumber::I64(*v as i64),
        IntValue::I32(v) => JsonNumber::I64(*v as i64),
        IntValue::I64(v) => JsonNumber::I64(*v),
        IntValue::U8(v) => JsonNumber::U64(*v as u64),
        IntValue::U16(v) => JsonNumber::U64(*v as u64),
        IntValue::U32(v) => JsonNumber::U64(*v as u64),
        IntValue::U64(v) => JsonNumber::U64(*v),
    }
}

fn number_to_json(value: f64, options: &JsonEncodeOptions) -> BuiltinResult<JsonValue> {
    if !value.is_finite() {
        if options.convert_inf_and_nan {
            return Ok(JsonValue::Null);
        }
        return Err(jsonencode_error(&JSONENCODE_ERROR_INF_NAN));
    }
    Ok(JsonValue::Number(JsonNumber::Float(value)))
}

fn logical_array_to_json(
    logical: &LogicalArray,
    _options: &JsonEncodeOptions,
) -> BuiltinResult<JsonValue> {
    let keep_dims = compute_keep_dims(&logical.shape, true);
    if logical.shape.is_empty() || logical.data.is_empty() {
        return Ok(JsonValue::Array(Vec::new()));
    }
    if keep_dims.is_empty() {
        let first = logical.data.first().copied().unwrap_or(0) != 0;
        return Ok(JsonValue::Bool(first));
    }
    build_strided_array(&logical.shape, &keep_dims, |offset| {
        Ok(JsonValue::Bool(logical.data[offset] != 0))
    })
}

fn tensor_to_json(tensor: &Tensor, options: &JsonEncodeOptions) -> BuiltinResult<JsonValue> {
    if tensor::tensor_element_len(tensor) == 0 {
        return Ok(JsonValue::Array(Vec::new()));
    }
    let keep_dims = compute_keep_dims(tensor.shape(), true);
    if keep_dims.is_empty() {
        return tensor_value_to_json(tensor, 0, options);
    }
    build_strided_array(tensor.shape(), &keep_dims, |offset| {
        tensor_value_to_json(tensor, offset, options)
    })
}

fn tensor_value_to_json(
    tensor: &Tensor,
    offset: usize,
    options: &JsonEncodeOptions,
) -> BuiltinResult<JsonValue> {
    let value = tensor
        .numeric_value_at(offset)
        .expect("index within authoritative numeric storage");
    match value.into_int_value() {
        Some(value) => Ok(JsonValue::Number(integer_value_number(&value))),
        None => number_to_json(value.materialize_f64(), options),
    }
}

fn integer_value_number(value: &IntValue) -> JsonNumber {
    match value {
        IntValue::I8(value) => JsonNumber::I64(*value as i64),
        IntValue::I16(value) => JsonNumber::I64(*value as i64),
        IntValue::I32(value) => JsonNumber::I64(*value as i64),
        IntValue::I64(value) => JsonNumber::I64(*value),
        IntValue::U8(value) => JsonNumber::U64(*value as u64),
        IntValue::U16(value) => JsonNumber::U64(*value as u64),
        IntValue::U32(value) => JsonNumber::U64(*value as u64),
        IntValue::U64(value) => JsonNumber::U64(*value),
    }
}

fn integer_storage_number(storage: &IntegerStorage, offset: usize) -> JsonNumber {
    integer_value_number(
        &storage
            .value_at(offset)
            .expect("index within authoritative integer storage"),
    )
}

fn complex_scalar_to_json(
    real: f64,
    imag: f64,
    options: &JsonEncodeOptions,
) -> BuiltinResult<JsonValue> {
    let real_json = number_to_json(real, options)?;
    let imag_json = number_to_json(imag, options)?;
    Ok(JsonValue::Object(vec![
        ("real".to_string(), real_json),
        ("imag".to_string(), imag_json),
    ]))
}

fn complex_tensor_to_json(
    ct: &ComplexTensor,
    options: &JsonEncodeOptions,
) -> BuiltinResult<JsonValue> {
    if tensor::complex_tensor_element_len(ct) == 0 {
        return Ok(JsonValue::Array(Vec::new()));
    }
    let keep_dims = compute_keep_dims(ct.shape(), true);
    if keep_dims.is_empty() {
        return complex_tensor_value_to_json(ct, 0, options);
    }
    build_strided_array(ct.shape(), &keep_dims, |offset| {
        complex_tensor_value_to_json(ct, offset, options)
    })
}

fn complex_tensor_value_to_json(
    ct: &ComplexTensor,
    offset: usize,
    options: &JsonEncodeOptions,
) -> BuiltinResult<JsonValue> {
    if let Some(storage) = &ct.integer_storage() {
        return Ok(JsonValue::Object(vec![
            (
                "real".to_string(),
                JsonValue::Number(integer_storage_number(storage.real(), offset)),
            ),
            (
                "imag".to_string(),
                JsonValue::Number(integer_storage_number(storage.imag(), offset)),
            ),
        ]));
    }

    let (real, imag) = ct.materialize_f64()[offset];
    complex_scalar_to_json(real, imag, options)
}

fn string_array_to_json(
    sa: &StringArray,
    _options: &JsonEncodeOptions,
) -> BuiltinResult<JsonValue> {
    if sa.data().is_empty() {
        return Ok(JsonValue::Array(Vec::new()));
    }
    let keep_dims = compute_keep_dims(&sa.shape, true);
    if keep_dims.is_empty() {
        return Ok(string_element_to_json(&sa.data()[0]));
    }
    build_strided_array(&sa.shape, &keep_dims, |offset| {
        Ok(string_element_to_json(&sa.data()[offset]))
    })
}

fn symbolic_array_to_json(
    array: &SymbolicArray,
    _options: &JsonEncodeOptions,
) -> BuiltinResult<JsonValue> {
    if array.data().is_empty() {
        return Ok(JsonValue::Array(Vec::new()));
    }
    let keep_dims = compute_keep_dims(&array.shape, true);
    if keep_dims.is_empty() {
        return Ok(JsonValue::String(array.data()[0].to_string().into()));
    }
    build_strided_array(&array.shape, &keep_dims, |offset| {
        Ok(JsonValue::String(array.data()[offset].to_string().into()))
    })
}

fn char_array_to_json(ca: &CharArray, _options: &JsonEncodeOptions) -> BuiltinResult<JsonValue> {
    if ca.rows == 0 {
        return Ok(JsonValue::Array(Vec::new()));
    }

    if ca.cols == 0 {
        if ca.rows == 1 {
            return Ok(JsonValue::String(runmat_value::Utf8Text::default()));
        }
        let mut rows = Vec::with_capacity(ca.rows);
        for _ in 0..ca.rows {
            rows.push(JsonValue::String(runmat_value::Utf8Text::default()));
        }
        return Ok(JsonValue::Array(rows));
    }

    if ca.rows == 1 {
        return runmat_value::decode_utf16_code_units(ca.data())
            .map(|text| JsonValue::String(text.into()))
            .map_err(|source| {
                jsonencode_error_with(
                    &JSONENCODE_ERROR_INTERNAL,
                    format!("jsonencode: invalid UTF-16 character row: {source}"),
                )
            });
    }

    let mut rows = Vec::with_capacity(ca.rows);
    for r in 0..ca.rows {
        let start = r * ca.cols;
        let row_string = runmat_value::decode_utf16_code_units(&ca.data()[start..start + ca.cols])
            .map_err(|source| {
                jsonencode_error_with(
                    &JSONENCODE_ERROR_INTERNAL,
                    format!("jsonencode: invalid UTF-16 character row: {source}"),
                )
            })?;
        rows.push(JsonValue::String(row_string.into()));
    }
    Ok(JsonValue::Array(rows))
}

fn struct_to_json(sv: &StructValue, options: &JsonEncodeOptions) -> BuiltinResult<JsonValue> {
    struct_fields_to_json(sv.fields.iter(), options)
}

fn struct_fields_to_json<'a>(
    fields: impl Iterator<Item = (&'a str, &'a Value)>,
    options: &JsonEncodeOptions,
) -> BuiltinResult<JsonValue> {
    let mut map = BTreeMap::new();
    for (key, value) in fields {
        map.insert(key.to_owned(), value_to_json(value, options)?);
    }
    Ok(JsonValue::Object(map.into_iter().collect()))
}

fn object_to_json(obj: &ObjectInstance, options: &JsonEncodeOptions) -> BuiltinResult<JsonValue> {
    let mut map = BTreeMap::new();
    for (key, value) in obj.properties() {
        map.insert(key.to_owned(), value_to_json(value, options)?);
    }
    Ok(JsonValue::Object(map.into_iter().collect()))
}

fn cell_array_to_json(ca: &CellArray, options: &JsonEncodeOptions) -> BuiltinResult<JsonValue> {
    if ca.rows == 0 || ca.cols == 0 {
        return Ok(JsonValue::Array(Vec::new()));
    }

    if ca.rows == 1 && ca.cols == 1 {
        let value = ca.get(0, 0).map_err(|e| {
            jsonencode_error_with(
                &JSONENCODE_ERROR_INTERNAL,
                format!("{} ({e})", JSONENCODE_ERROR_INTERNAL.message),
            )
        })?;
        return Ok(JsonValue::Array(vec![value_to_json(&value, options)?]));
    }

    if ca.rows == 1 {
        let mut row = Vec::with_capacity(ca.cols);
        for c in 0..ca.cols {
            let element = ca.get(0, c).map_err(|e| {
                jsonencode_error_with(
                    &JSONENCODE_ERROR_INTERNAL,
                    format!("{} ({e})", JSONENCODE_ERROR_INTERNAL.message),
                )
            })?;
            row.push(value_to_json(&element, options)?);
        }
        return Ok(JsonValue::Array(row));
    }

    if ca.cols == 1 {
        let mut column = Vec::with_capacity(ca.rows);
        for r in 0..ca.rows {
            let element = ca.get(r, 0).map_err(|e| {
                jsonencode_error_with(
                    &JSONENCODE_ERROR_INTERNAL,
                    format!("{} ({e})", JSONENCODE_ERROR_INTERNAL.message),
                )
            })?;
            column.push(value_to_json(&element, options)?);
        }
        return Ok(JsonValue::Array(column));
    }

    let mut rows = Vec::with_capacity(ca.rows);
    for r in 0..ca.rows {
        let mut row = Vec::with_capacity(ca.cols);
        for c in 0..ca.cols {
            let element = ca.get(r, c).map_err(|e| {
                jsonencode_error_with(
                    &JSONENCODE_ERROR_INTERNAL,
                    format!("{} ({e})", JSONENCODE_ERROR_INTERNAL.message),
                )
            })?;
            row.push(value_to_json(&element, options)?);
        }
        rows.push(JsonValue::Array(row));
    }
    Ok(JsonValue::Array(rows))
}

fn compute_keep_dims(shape: &[usize], drop_singletons: bool) -> Vec<usize> {
    let mut keep = Vec::new();
    for (idx, &size) in shape.iter().enumerate() {
        if size != 1 || !drop_singletons {
            keep.push(idx);
        }
    }
    keep
}

fn compute_strides(shape: &[usize]) -> BuiltinResult<Vec<usize>> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut acc = 1usize;
    for &size in shape {
        strides.push(acc);
        acc = acc.checked_mul(size.max(1)).ok_or_else(|| {
            jsonencode_error_with(
                &JSONENCODE_ERROR_INTERNAL,
                "jsonencode: array shape exceeds platform limits",
            )
        })?;
    }
    Ok(strides)
}

fn build_strided_array<F>(
    shape: &[usize],
    keep_dims: &[usize],
    mut fetch: F,
) -> BuiltinResult<JsonValue>
where
    F: FnMut(usize) -> BuiltinResult<JsonValue>,
{
    if keep_dims.is_empty() {
        return fetch(0);
    }
    if keep_dims
        .iter()
        .any(|&idx| shape.get(idx).copied() == Some(0))
    {
        return Ok(JsonValue::Array(Vec::new()));
    }
    let strides = compute_strides(shape)?;
    let dims = keep_dims
        .iter()
        .map(|&idx| {
            shape.get(idx).copied().ok_or_else(|| {
                jsonencode_error_with(
                    &JSONENCODE_ERROR_INTERNAL,
                    "jsonencode: array dimension is out of bounds",
                )
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    build_nd_array(&dims, |indices| {
        let mut offset = 0usize;
        for (value, dim_idx) in indices.iter().zip(keep_dims.iter()) {
            offset = value
                .checked_mul(strides[*dim_idx])
                .and_then(|component| offset.checked_add(component))
                .ok_or_else(|| {
                    jsonencode_error_with(
                        &JSONENCODE_ERROR_INTERNAL,
                        "jsonencode: array offset exceeds platform limits",
                    )
                })?;
        }
        fetch(offset)
    })
}

fn build_nd_array<F>(dims: &[usize], mut fetch: F) -> BuiltinResult<JsonValue>
where
    F: FnMut(&[usize]) -> BuiltinResult<JsonValue>,
{
    if dims.is_empty() {
        return fetch(&[]);
    }
    if dims[0] == 0 {
        return Ok(JsonValue::Array(Vec::new()));
    }
    let mut indices = vec![0usize; dims.len()];
    build_nd_array_recursive(dims, 0, &mut indices, &mut fetch)
}

fn build_nd_array_recursive<F>(
    dims: &[usize],
    level: usize,
    indices: &mut [usize],
    fetch: &mut F,
) -> BuiltinResult<JsonValue>
where
    F: FnMut(&[usize]) -> BuiltinResult<JsonValue>,
{
    let size = dims[level];
    if size == 0 {
        return Ok(JsonValue::Array(Vec::new()));
    }
    if level + 1 == dims.len() {
        let mut items = Vec::with_capacity(size);
        for i in 0..size {
            indices[level] = i;
            items.push(fetch(indices)?);
        }
        return Ok(JsonValue::Array(items));
    }
    let mut items = Vec::with_capacity(size);
    for i in 0..size {
        indices[level] = i;
        items.push(build_nd_array_recursive(dims, level + 1, indices, fetch)?);
    }
    Ok(JsonValue::Array(items))
}
