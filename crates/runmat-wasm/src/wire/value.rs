use runmat_builtins::{NumericDType, ObjectInstance, StructValue, Value};
use runmat_core::{matlab_class_name, value_shape};
use serde_json::{json, Map as JsonMap, Value as JsonValue};

pub(crate) const MAX_DATA_PREVIEW: usize = 4096;
const MAX_STRUCT_FIELDS: usize = 64;
const MAX_OBJECT_FIELDS: usize = 64;
const MAX_OUTPUT_LIST_ITEMS: usize = 64;
const MAX_RECURSION_DEPTH: usize = 2;

pub(crate) fn value_to_json(value: &Value, depth: usize) -> JsonValue {
    if depth >= MAX_RECURSION_DEPTH {
        return json!({
            "kind": "display",
            "className": matlab_class_name(value),
            "shape": value_shape(value),
            "value": value.to_string(),
        });
    }

    match value {
        Value::Int(iv) => json!({
            "kind": "int",
            "className": iv.class_name(),
            "value": iv.to_i64(),
            "shape": scalar_shape(),
        }),
        Value::Num(n) => json!({
            "kind": "double",
            "value": n,
            "shape": scalar_shape(),
        }),
        Value::Complex(re, im) => json!({
            "kind": "complex",
            "real": re,
            "imag": im,
            "shape": scalar_shape(),
        }),
        Value::Bool(b) => json!({
            "kind": "logical",
            "value": b,
            "shape": scalar_shape(),
        }),
        Value::LogicalArray(arr) => {
            let (preview, truncated) = preview_slice(&arr.data, MAX_DATA_PREVIEW);
            let rows = arr.shape.first().copied().unwrap_or(0);
            let cols = arr.shape.get(1).copied().unwrap_or(0);
            json!({
                "kind": "logical-array",
                "shape": arr.shape,
                "rows": rows,
                "cols": cols,
                "preview": preview.iter().map(|v| *v != 0).collect::<Vec<bool>>(),
                "length": arr.data.len(),
                "truncated": truncated,
            })
        }
        Value::String(s) => json!({
            "kind": "string",
            "value": s,
            "shape": vec![1, s.chars().count()],
        }),
        Value::StringArray(sa) => {
            let (preview, truncated) = preview_slice(&sa.data, MAX_DATA_PREVIEW);
            json!({
                "kind": "string-array",
                "shape": sa.shape,
                "rows": sa.rows,
                "cols": sa.cols,
                "preview": preview,
                "length": sa.data.len(),
                "truncated": truncated,
            })
        }
        Value::CharArray(ca) => {
            let s: String = ca.data.iter().copied().collect();
            json!({
                "kind": "char-array",
                "rows": ca.rows,
                "cols": ca.cols,
                "shape": vec![ca.rows, ca.cols],
                "text": s,
            })
        }
        Value::Symbolic(expr) => json!({
            "kind": "symbolic",
            "className": "sym",
            "shape": scalar_shape(),
            "value": expr.to_string(),
        }),
        Value::SymbolicArray(array) => {
            let preview_values: Vec<String> = array.data.iter().map(ToString::to_string).collect();
            let (preview, truncated) = preview_slice(&preview_values, MAX_DATA_PREVIEW);
            json!({
                "kind": "symbolic-array",
                "className": "sym",
                "shape": array.shape,
                "rows": array.rows,
                "cols": array.cols,
                "preview": preview,
                "length": array.data.len(),
                "truncated": truncated,
            })
        }
        Value::Tensor(t) => {
            let (preview, truncated) = preview_slice(&t.data, MAX_DATA_PREVIEW);
            json!({
                "kind": "tensor",
                "shape": t.shape,
                "rows": t.rows,
                "cols": t.cols,
                "dtype": match t.dtype {
                    NumericDType::F64 => "double",
                    NumericDType::F32 => "single",
                    NumericDType::U8 => "uint8",
                    NumericDType::U16 => "uint16",
                },
                "preview": preview,
                "length": t.data.len(),
                "truncated": truncated,
            })
        }
        Value::ComplexTensor(t) => {
            let (preview, truncated) = preview_slice(&t.data, MAX_DATA_PREVIEW);
            let preview: Vec<JsonValue> = preview
                .into_iter()
                .map(|(re, im)| json!({ "real": re, "imag": im }))
                .collect();
            json!({
                "kind": "complex-tensor",
                "shape": t.shape,
                "rows": t.rows,
                "cols": t.cols,
                "preview": preview,
                "length": t.data.len(),
                "truncated": truncated,
            })
        }
        Value::SparseTensor(st) => {
            let (entry_preview, entry_preview_truncated) =
                sparse_entry_preview(st, MAX_DATA_PREVIEW);
            let (col_ptrs_preview, col_ptrs_truncated) =
                preview_slice(&st.col_ptrs, MAX_DATA_PREVIEW);
            let (row_indices_preview, row_indices_truncated) =
                preview_slice(&st.row_indices, MAX_DATA_PREVIEW);
            let (values_preview, values_truncated) = preview_slice(&st.values, MAX_DATA_PREVIEW);
            json!({
                "kind": "sparse-tensor",
                "shape": vec![st.rows, st.cols],
                "rows": st.rows,
                "cols": st.cols,
                "nnz": st.nnz(),
                "colPtrsPreview": col_ptrs_preview,
                "colPtrsLength": st.col_ptrs.len(),
                "colPtrsTruncated": col_ptrs_truncated,
                "rowIndicesPreview": row_indices_preview,
                "rowIndicesLength": st.row_indices.len(),
                "rowIndicesTruncated": row_indices_truncated,
                "valuesPreview": values_preview,
                "valuesLength": st.values.len(),
                "valuesTruncated": values_truncated,
                "preview": entry_preview,
                "entryPreviewTruncated": entry_preview_truncated,
                "truncated": entry_preview_truncated
                    || col_ptrs_truncated
                    || row_indices_truncated
                    || values_truncated,
            })
        }
        Value::Cell(ca) => json!({
            "kind": "cell",
            "shape": ca.shape,
            "rows": ca.rows,
            "cols": ca.cols,
            "length": ca.data.len(),
        }),
        Value::OutputList(values) => {
            let truncated = values.len() > MAX_OUTPUT_LIST_ITEMS;
            let items: Vec<JsonValue> = values
                .iter()
                .take(MAX_OUTPUT_LIST_ITEMS)
                .map(|v| value_to_json(v, depth + 1))
                .collect();
            json!({
                "kind": "output-list",
                "length": values.len(),
                "items": items,
                "truncated": truncated,
            })
        }
        Value::Struct(st) => struct_to_json(st, depth + 1),
        Value::GpuTensor(handle) => {
            let (rows, cols) = rows_cols_from_shape(&handle.shape);
            json!({
                "kind": "gpu-tensor",
                "shape": handle.shape,
                "rows": rows,
                "cols": cols,
                "deviceId": handle.device_id,
                "bufferId": handle.buffer_id,
            })
        }
        Value::Object(obj) => object_to_json(obj, depth + 1),
        Value::HandleObject(handle) => json!({
            "kind": "handle",
            "className": handle.class_name,
            "valid": handle.valid,
        }),
        Value::Listener(listener) => json!({
            "kind": "listener",
            "id": listener.id,
            "event": listener.event_name,
            "enabled": listener.enabled,
            "valid": listener.valid,
        }),
        Value::FunctionHandle(name) => json!({
            "kind": "function",
            "name": name,
        }),
        Value::ExternalFunctionHandle(name) => json!({
            "kind": "function",
            "name": name,
            "source": "external",
        }),
        Value::MethodFunctionHandle(name) => json!({
            "kind": "function",
            "name": name,
            "source": "method",
        }),
        Value::BoundFunctionHandle { name, function } => json!({
            "kind": "function",
            "name": name,
            "source": "bound",
            "functionId": function,
        }),
        Value::Closure(closure) => json!({
            "kind": "closure",
            "functionName": closure.function_name,
            "captureCount": closure.captures.len(),
        }),
        Value::ClassRef(name) => json!({
            "kind": "class-ref",
            "name": name,
        }),
        Value::MException(ex) => json!({
            "kind": "exception",
            "identifier": ex.identifier,
            "message": ex.message,
            "stack": ex.stack,
        }),
    }
}

fn struct_to_json(st: &StructValue, depth: usize) -> JsonValue {
    let mut fields = JsonMap::new();
    let mut truncated = false;
    for (idx, (name, field_value)) in st.fields.iter().enumerate() {
        if idx >= MAX_STRUCT_FIELDS {
            truncated = true;
            break;
        }
        fields.insert(name.clone(), value_to_json(field_value, depth));
    }
    json!({
        "kind": "struct",
        "fieldOrder": st.field_names().take(MAX_STRUCT_FIELDS).cloned().collect::<Vec<_>>(),
        "fields": fields,
        "totalFields": st.fields.len(),
        "truncated": truncated,
    })
}

fn object_to_json(obj: &ObjectInstance, depth: usize) -> JsonValue {
    let mut fields = JsonMap::new();
    let mut truncated = false;
    for (idx, (name, value)) in obj.properties.iter().enumerate() {
        if idx >= MAX_OBJECT_FIELDS {
            truncated = true;
            break;
        }
        fields.insert(name.clone(), value_to_json(value, depth));
    }
    json!({
        "kind": "object",
        "className": obj.class_name,
        "properties": fields,
        "propertyCount": obj.properties.len(),
        "truncated": truncated,
    })
}

fn scalar_shape() -> Vec<usize> {
    vec![1, 1]
}

fn rows_cols_from_shape(shape: &[usize]) -> (usize, usize) {
    let rows = shape.first().copied().unwrap_or(0);
    let cols = if shape.len() >= 2 {
        shape[1]
    } else if rows == 0 {
        0
    } else {
        1
    };
    (rows, cols)
}

fn sparse_entry_preview(
    st: &runmat_builtins::SparseTensor,
    limit: usize,
) -> (Vec<JsonValue>, bool) {
    let mut entries = Vec::with_capacity(st.nnz().min(limit));
    for col in 0..st.cols {
        let start = st.col_ptrs[col];
        let end = st.col_ptrs[col + 1];
        for idx in start..end {
            if entries.len() >= limit {
                return (entries, true);
            }
            entries.push(json!({
                "row": st.row_indices[idx] + 1,
                "col": col + 1,
                "value": st.values[idx],
            }));
        }
    }
    (entries, false)
}

fn preview_slice<T: Clone>(data: &[T], limit: usize) -> (Vec<T>, bool) {
    if data.len() > limit {
        (data[..limit].to_vec(), true)
    } else {
        (data.to_vec(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::SparseTensor;

    #[test]
    fn sparse_tensor_json_uses_bounded_storage_previews() {
        let rows = MAX_DATA_PREVIEW + 2;
        let cols = 1;
        let col_ptrs = vec![0, rows];
        let row_indices = (0..rows).collect::<Vec<_>>();
        let values = (0..rows).map(|idx| idx as f64).collect::<Vec<_>>();
        let sparse = SparseTensor::new(rows, cols, col_ptrs, row_indices, values).unwrap();

        let json = value_to_json(&Value::SparseTensor(sparse), 0);

        assert!(json.get("colPtrs").is_none());
        assert!(json.get("rowIndices").is_none());
        assert!(json.get("values").is_none());
        assert_eq!(json["rowIndicesLength"], rows);
        assert_eq!(json["valuesLength"], rows);
        assert_eq!(
            json["rowIndicesPreview"].as_array().unwrap().len(),
            MAX_DATA_PREVIEW
        );
        assert_eq!(
            json["valuesPreview"].as_array().unwrap().len(),
            MAX_DATA_PREVIEW
        );
        assert_eq!(json["rowIndicesTruncated"], true);
        assert_eq!(json["valuesTruncated"], true);
        assert_eq!(json["truncated"], true);
    }
}
