use runmat_core::{matlab_class_name, value_shape};
use runmat_value::{ComplexStorage, IntValue, NumericScalar, ObjectInstance, StructValue, Value};
use serde_json::{json, Map as JsonMap, Value as JsonValue};

pub(crate) const MAX_DATA_PREVIEW: usize = 4096;
const MAX_STRUCT_FIELDS: usize = 64;
const MAX_OBJECT_FIELDS: usize = 64;
const MAX_OBJECT_ARRAY_ITEMS: usize = 256;
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
            "value": integer_json_value(iv),
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
                "shape": ca.shape.clone(),
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
            let (preview, truncated, length) = if let Some(storage) = t.integer_storage() {
                let truncated = storage.len() > MAX_DATA_PREVIEW;
                let preview = (0..storage.len().min(MAX_DATA_PREVIEW))
                    .map(|index| {
                        integer_json_value(
                            &storage
                                .value_at(index)
                                .expect("integer storage index is valid"),
                        )
                    })
                    .collect::<Vec<_>>();
                (preview, truncated, storage.len())
            } else {
                let values = t.materialize_f64();
                let (preview, truncated) = preview_slice(&values, MAX_DATA_PREVIEW);
                (
                    preview.into_iter().map(JsonValue::from).collect(),
                    truncated,
                    t.len(),
                )
            };
            json!({
                "kind": "tensor",
                "shape": t.shape,
                "rows": t.rows,
                "cols": t.cols,
                "dtype": t.numeric_dtype().class_name(),
                "preview": preview,
                "length": length,
                "truncated": truncated,
            })
        }
        Value::ComplexTensor(t) => {
            let (preview, truncated, length, dtype) = match t.complex_storage() {
                ComplexStorage::Integer(storage) => {
                    let length = storage.len();
                    let preview: Vec<JsonValue> = (0..length.min(MAX_DATA_PREVIEW))
                        .map(|index| {
                            json!({
                                "real": integer_json_value(
                                    &storage.real.value_at(index).expect("integer storage index is valid"),
                                ),
                                "imag": integer_json_value(
                                    &storage.imag.value_at(index).expect("integer storage index is valid"),
                                ),
                            })
                        })
                        .collect();
                    (
                        preview,
                        length > MAX_DATA_PREVIEW,
                        length,
                        storage.class_name(),
                    )
                }
                ComplexStorage::F64(values) => {
                    let (preview, truncated) = preview_slice(values, MAX_DATA_PREVIEW);
                    let preview = preview
                        .into_iter()
                        .map(|(real, imag)| json!({ "real": real, "imag": imag }))
                        .collect();
                    (preview, truncated, values.len(), "double")
                }
                ComplexStorage::F32(values) => {
                    let (preview, truncated) = preview_slice(values, MAX_DATA_PREVIEW);
                    let preview = preview
                        .into_iter()
                        .map(|(real, imag)| json!({ "real": real, "imag": imag }))
                        .collect();
                    (preview, truncated, values.len(), "single")
                }
            };
            json!({
                "kind": "complex-tensor",
                "shape": t.shape,
                "rows": t.rows,
                "cols": t.cols,
                "dtype": dtype,
                "preview": preview,
                "length": length,
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
            let (values_preview, values_truncated) = sparse_values_preview(st, MAX_DATA_PREVIEW);
            json!({
                "kind": "sparse-tensor",
                "shape": vec![st.rows, st.cols],
                "rows": st.rows,
                "cols": st.cols,
                "dtype": st.class_name(),
                "nnz": st.nnz(),
                "colPtrsPreview": col_ptrs_preview,
                "colPtrsLength": st.col_ptrs.len(),
                "colPtrsTruncated": col_ptrs_truncated,
                "rowIndicesPreview": row_indices_preview,
                "rowIndicesLength": st.row_indices.len(),
                "rowIndicesTruncated": row_indices_truncated,
                "valuesPreview": values_preview,
                "valuesLength": st.nnz(),
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
        Value::ObjectArray(array) => {
            let truncated = array.len() > MAX_OBJECT_ARRAY_ITEMS;
            let items = array
                .data()
                .iter()
                .take(MAX_OBJECT_ARRAY_ITEMS)
                .map(|value| value_to_json(value, depth + 1))
                .collect::<Vec<_>>();
            let (rows, cols) = rows_cols_from_shape(array.shape());
            json!({
                "kind": "object-array",
                "className": array.class_name(),
                "shape": array.shape(),
                "rows": rows,
                "cols": cols,
                "length": array.len(),
                "items": items,
                "truncated": truncated,
            })
        }
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
        Value::Future(handle) => json!({
            "kind": "future",
            "id": handle.id.to_string(),
            "scopeId": handle.scope_id.to_string(),
            "requestedOutputs": handle.outputs.requested_outputs,
        }),
        Value::Task(handle) => json!({
            "kind": "task",
            "id": handle.id.to_string(),
            "scopeId": handle.scope_id.to_string(),
            "generation": handle.generation,
            "requestedOutputs": handle.outputs.requested_outputs,
        }),
        Value::Pool(handle) => json!({
            "kind": "pool",
            "id": handle.id.to_string(),
            "scopeId": handle.scope_id.to_string(),
            "generation": handle.generation,
        }),
        Value::Job(handle) => json!({
            "kind": "job",
            "id": handle.id.to_string(),
            "runId": handle.run_id.to_string(),
            "generation": handle.generation,
            "requestedOutputs": handle.outputs.requested_outputs,
        }),
        Value::Foreign(reference) => json!({
            "kind": "foreign",
            "family": reference.type_identity.family,
            "typeName": reference.type_identity.name,
            "typeVersion": reference.type_identity.version,
            "ownership": reference.ownership,
            "affinity": reference.affinity,
            "lifetime": reference.lifetime,
            "opaque": true,
        }),
    }
}

const MAX_SAFE_JS_INTEGER: i64 = 9_007_199_254_740_991;

pub(crate) fn integer_json_value(value: &IntValue) -> JsonValue {
    match value {
        IntValue::I64(value) if value.unsigned_abs() > MAX_SAFE_JS_INTEGER as u64 => {
            JsonValue::String(value.to_string())
        }
        IntValue::U64(value) if *value > MAX_SAFE_JS_INTEGER as u64 => {
            JsonValue::String(value.to_string())
        }
        IntValue::I8(value) => JsonValue::from(*value),
        IntValue::I16(value) => JsonValue::from(*value),
        IntValue::I32(value) => JsonValue::from(*value),
        IntValue::I64(value) => JsonValue::from(*value),
        IntValue::U8(value) => JsonValue::from(*value),
        IntValue::U16(value) => JsonValue::from(*value),
        IntValue::U32(value) => JsonValue::from(*value),
        IntValue::U64(value) => JsonValue::from(*value),
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

fn numeric_scalar_json_value(value: NumericScalar) -> JsonValue {
    match value {
        NumericScalar::F64(value) => JsonValue::from(value),
        NumericScalar::F32(value) => JsonValue::from(value),
        integer => integer_json_value(
            &integer
                .into_int_value()
                .expect("non-floating numeric scalar is an integer"),
        ),
    }
}

fn sparse_entry_preview(st: &runmat_value::SparseTensor, limit: usize) -> (Vec<JsonValue>, bool) {
    let mut entries = Vec::with_capacity(st.nnz().min(limit));
    for col in 0..st.cols {
        let start = st.col_ptrs[col];
        let end = st.col_ptrs[col + 1];
        for idx in start..end {
            if entries.len() >= limit {
                return (entries, true);
            }
            let value = numeric_scalar_json_value(
                st.numeric_value_at(idx)
                    .expect("sparse storage index is valid"),
            );
            entries.push(json!({
                "row": st.row_indices[idx] + 1,
                "col": col + 1,
                "value": value,
            }));
        }
    }
    (entries, false)
}

fn sparse_values_preview(st: &runmat_value::SparseTensor, limit: usize) -> (Vec<JsonValue>, bool) {
    let truncated = st.nnz() > limit;
    let preview = (0..st.nnz().min(limit))
        .map(|index| {
            numeric_scalar_json_value(
                st.numeric_value_at(index)
                    .expect("sparse storage index is valid"),
            )
        })
        .collect();
    (preview, truncated)
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
    use runmat_value::{
        ComplexTensor, IntegerComplexStorage, IntegerStorage, ObjectArray, ObjectInstance,
        SparseTensor, Tensor,
    };
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn integer_json_preserves_exact_64_bit_values_and_dtype() {
        let scalar = value_to_json(&Value::Int(IntValue::U64(u64::MAX)), 0);
        assert_eq!(scalar["value"], "18446744073709551615");

        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![42, u64::MAX]), vec![1, 2])
            .expect("tensor");
        let json = value_to_json(&Value::Tensor(tensor), 0);
        assert_eq!(json["dtype"], "uint64");
        assert_eq!(json["preview"], json!([42, "18446744073709551615"]));
        assert_eq!(json["length"], 2);
    }

    #[wasm_bindgen_test]
    fn native_single_tensor_json_preserves_dtype_and_values() {
        let tensor = Tensor::from_f32(vec![1.25, -3.5], vec![1, 2]).expect("single tensor");
        let json = value_to_json(&Value::Tensor(tensor), 0);

        assert_eq!(json["dtype"], "single");
        assert_eq!(json["preview"], json!([1.25, -3.5]));
        assert_eq!(json["length"], 2);
    }

    #[wasm_bindgen_test]
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

    #[test]
    fn object_array_json_preserves_class_shape_and_items() {
        let mut first = ObjectInstance::new("matlab.unittest.TestResult".into());
        first.properties.insert("Passed".into(), Value::Bool(true));
        let mut second = ObjectInstance::new("matlab.unittest.TestResult".into());
        second
            .properties
            .insert("Passed".into(), Value::Bool(false));
        let array = ObjectArray::row(
            "matlab.unittest.TestResult",
            vec![Value::Object(first), Value::Object(second)],
        )
        .unwrap();

        let json = value_to_json(&Value::ObjectArray(array), 0);

        assert_eq!(json["kind"], "object-array");
        assert_eq!(json["className"], "matlab.unittest.TestResult");
        assert_eq!(json["shape"], json!([1, 2]));
        assert_eq!(json["items"].as_array().unwrap().len(), 2);
        assert_eq!(json["truncated"], false);
    }

    #[test]
    fn foreign_json_is_opaque_and_does_not_expose_runtime_handle_identity() {
        let value = Value::Foreign(runmat_value::ForeignRef {
            host_identity: "private-host".into(),
            handle: 42,
            generation: 7,
            type_identity: runmat_value::ForeignTypeIdentity {
                family: "java".into(),
                name: "java.lang.StringBuilder".into(),
                version: 1,
            },
            ownership: runmat_value::ForeignOwnership::Shared,
            affinity: runmat_value::ForeignAffinity::OriginProcess,
            lifetime: runmat_value::ForeignLifetime::Session,
        });

        let json = value_to_json(&value, 0);
        assert_eq!(json["kind"], "foreign");
        assert_eq!(json["family"], "java");
        assert_eq!(json["typeName"], "java.lang.StringBuilder");
        assert_eq!(json["opaque"], true);
        assert!(json.get("hostIdentity").is_none());
        assert!(json.get("handle").is_none());
        assert!(json.get("generation").is_none());
    }

    #[wasm_bindgen_test]
    fn native_single_sparse_json_preserves_dtype_and_values() {
        let sparse = SparseTensor::new_f32(2, 2, vec![0, 1, 2], vec![1, 0], vec![1.25, 3.5])
            .expect("single sparse");
        let json = value_to_json(&Value::SparseTensor(sparse), 0);
        assert_eq!(json["dtype"], "single");
        assert_eq!(json["valuesPreview"], json!([1.25, 3.5]));
        assert_eq!(
            json["preview"],
            json!([
                {"row": 2, "col": 1, "value": 1.25},
                {"row": 1, "col": 2, "value": 3.5}
            ])
        );
    }

    #[wasm_bindgen_test]
    fn typed_complex_and_sparse_json_preserve_exact_64_bit_values() {
        let complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![u64::MAX]),
                IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
            )
            .expect("matching components"),
            vec![1, 1],
        )
        .expect("complex tensor");
        let complex_json = value_to_json(&Value::ComplexTensor(complex), 0);
        assert_eq!(complex_json["dtype"], "uint64");
        assert_eq!(
            complex_json["preview"],
            json!([{"real": "18446744073709551615", "imag": "9007199254740993"}])
        );

        let sparse = SparseTensor::new_integer(
            1,
            1,
            vec![0, 1],
            vec![0],
            IntegerStorage::U64(vec![u64::MAX]),
        )
        .expect("sparse tensor");
        let sparse_json = value_to_json(&Value::SparseTensor(sparse), 0);
        assert_eq!(
            sparse_json["valuesPreview"],
            json!(["18446744073709551615"])
        );
        assert_eq!(
            sparse_json["preview"],
            json!([{"row": 1, "col": 1, "value": "18446744073709551615"}])
        );
        assert_eq!(sparse_json["valuesLength"], 1);
    }
}
