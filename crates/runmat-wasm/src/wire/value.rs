mod aggregate;
mod numeric;
use aggregate::{object_to_json, struct_to_json};
pub(crate) use numeric::integer_json_value;
use numeric::{numeric_scalar_json_value, sparse_entry_preview, sparse_values_preview};

use runmat_core::{matlab_class_name, value_shape};
use runmat_value::{ComplexStorage, Value};
use serde_json::{json, Value as JsonValue};

pub(crate) const MAX_DATA_PREVIEW: usize = 4096;
const MAX_STRUCT_FIELDS: usize = 64;
const MAX_OBJECT_FIELDS: usize = 64;
const MAX_OBJECT_ARRAY_ITEMS: usize = 256;
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
                "shape": arr.shape.as_slice(),
                "rows": rows,
                "cols": cols,
                "preview": preview.iter().map(|v| *v != 0).collect::<Vec<bool>>(),
                "length": arr.data.len(),
                "truncated": truncated,
            })
        }
        Value::String(s) => json!({
            "kind": "string",
            "value": s.as_str(),
            "shape": vec![1, 1],
        }),
        Value::StringArray(sa) => {
            let preview = sa
                .data()
                .iter()
                .take(MAX_DATA_PREVIEW)
                .map(|value| value.as_str())
                .collect::<Vec<_>>();
            let truncated = sa.data().len() > MAX_DATA_PREVIEW;
            json!({
                "kind": "string-array",
                "shape": sa.shape(),
                "rows": sa.rows,
                "cols": sa.cols,
                "preview": preview,
                "length": sa.data().len(),
                "truncated": truncated,
            })
        }
        Value::CharArray(ca) => {
            let (preview, truncated) = preview_slice(ca.data(), MAX_DATA_PREVIEW);
            let code_units = preview.iter().map(|unit| unit.raw()).collect::<Vec<_>>();
            let text = (!truncated)
                .then(|| runmat_value::decode_utf16_code_units(ca.data()).ok())
                .flatten();
            json!({
                "kind": "char-array",
                "encoding": "utf16-code-units-v1",
                "rows": ca.rows,
                "cols": ca.cols,
                "shape": ca.shape.as_slice(),
                "codeUnits": code_units,
                "text": text,
                "length": ca.data().len(),
                "truncated": truncated,
            })
        }
        Value::Symbolic(expr) => json!({
            "kind": "symbolic",
            "className": "sym",
            "shape": scalar_shape(),
            "value": expr.to_string(),
        }),
        Value::SymbolicArray(array) => {
            let preview = array
                .data()
                .iter()
                .take(MAX_DATA_PREVIEW)
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            let truncated = array.data().len() > MAX_DATA_PREVIEW;
            json!({
                "kind": "symbolic-array",
                "className": "sym",
                "shape": array.shape.as_slice(),
                "rows": array.rows,
                "cols": array.cols,
                "preview": preview,
                "length": array.data().len(),
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
                let preview = (0..t.len().min(MAX_DATA_PREVIEW))
                    .map(|index| {
                        numeric_scalar_json_value(
                            t.numeric_value_at(index)
                                .expect("tensor storage index is valid"),
                        )
                    })
                    .collect();
                (preview, t.len() > MAX_DATA_PREVIEW, t.len())
            };
            json!({
                "kind": "tensor",
                "shape": t.shape(),
                "rows": t.rows(),
                "cols": t.cols(),
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
                                    &storage.real().value_at(index).expect("integer storage index is valid"),
                                ),
                                "imag": integer_json_value(
                                    &storage.imag().value_at(index).expect("integer storage index is valid"),
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
                        .map(|value| json!({ "real": value.0, "imag": value.1 }))
                        .collect();
                    (preview, truncated, values.len(), "double")
                }
                ComplexStorage::F32(values) => {
                    let (preview, truncated) = preview_slice(values, MAX_DATA_PREVIEW);
                    let preview = preview
                        .into_iter()
                        .map(|value| json!({ "real": value.0, "imag": value.1 }))
                        .collect();
                    (preview, truncated, values.len(), "single")
                }
            };
            json!({
                "kind": "complex-tensor",
                "shape": t.shape(),
                "rows": t.rows(),
                "cols": t.cols(),
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
            "shape": ca.shape(),
            "rows": ca.rows,
            "cols": ca.cols,
            "length": ca.data().len(),
        }),
        Value::OutputList(_) => json!({
            "kind": "unavailable",
            "identifier": "TransientSequenceNotPortable",
            "message": "transient output sequences cannot be stored or serialized as values",
        }),
        Value::Struct(st) => struct_to_json(st, depth + 1),
        // The existing product-value JSON contract has no typed structure-array
        // form. Keep its established display fallback until that boundary has
        // an explicit versioned aggregate-value schema.
        Value::StructArray(array) => json!({
            "kind": "display",
            "className": "struct",
            "shape": array.shape(),
            "value": array.to_string(),
        }),
        Value::GpuTensor(handle) => {
            let (rows, cols) = rows_cols_from_shape(handle.shape());
            json!({
                "kind": "gpu-tensor",
                "shape": handle.shape(),
                "rows": rows,
                "cols": cols,
                "deviceId": handle.device_id(),
                "bufferId": handle.buffer_id(),
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
                "className": array.class_name().display_name(),
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
            "className": handle.class_name.display_name(),
            "valid": handle.valid,
        }),
        Value::Listener(listener) => json!({
            "kind": "listener",
            "id": listener.id,
            "event": listener.event_name.as_str(),
            "enabled": listener.enabled,
            "valid": listener.valid,
        }),
        Value::ScopedFunctionHandle(handle) => json!({
            "kind": "function",
            "source": "invocation-scoped",
            "live": handle.is_live(),
            "portable": false,
        }),
        Value::FunctionHandle(name) => json!({
            "kind": "function",
            "name": name.name(),
        }),
        Value::ExternalFunctionHandle(name) => json!({
            "kind": "function",
            "name": name.name(),
            "source": "external",
        }),
        Value::MethodFunctionHandle(name) => json!({
            "kind": "function",
            "name": name.name(),
            "source": "method",
        }),
        Value::BoundFunctionHandle(handle) => json!({
            "kind": "function",
            "name": handle.name.as_str(),
            "source": "bound",
            "functionId": handle.function,
            "sourceRole": handle.source_role,
            "sourceFile": handle.source_file.as_deref(),
        }),
        Value::Closure(closure) => json!({
            "kind": "closure",
            "functionName": closure.function_name.as_str(),
            "captureCount": closure.captures.len(),
            "sourceRole": closure.source_role,
            "sourceFile": closure.source_file.as_deref(),
        }),
        Value::ClassRef(name) => json!({
            "kind": "class-ref",
            "name": name.display_name(),
        }),
        Value::MException(ex) => json!({
            "kind": "exception",
            "identifier": ex.identifier.as_str(),
            "message": ex.message.as_str(),
            "stack": JsonValue::Array(ex.stack.iter().map(|frame| json!(frame.as_str())).collect()),
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
        Value::Distributed(handle) => json!({
            "kind": "distributed",
            "className": "distributed",
            "shape": handle.global_shape,
            "partitionCount": handle.partition_count.0,
            "scheme": handle.scheme,
            "materializable": handle.materializable,
            "valueFact": handle.value,
        }),
        Value::Composite(handle) => json!({
            "kind": "composite",
            "className": "Composite",
            "shape": scalar_shape(),
            "labCount": handle.gang.labs.0,
            "valueFact": handle.value,
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

fn preview_slice<T: Clone>(data: &[T], limit: usize) -> (Vec<T>, bool) {
    if data.len() > limit {
        (data[..limit].to_vec(), true)
    } else {
        (data.to_vec(), false)
    }
}

#[cfg(test)]
mod tests;
