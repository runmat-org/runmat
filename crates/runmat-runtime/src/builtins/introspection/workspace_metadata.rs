//! Workspace variable metadata accounting; no payload readback from resident values.

use crate::builtins::common::shape::value_dimensions;
use crate::builtins::introspection::class_identity::class_name_for_value;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::handle_is_logical;
use runmat_value::Value;
use std::collections::HashSet;

#[derive(Debug)]
pub(super) struct WhosRecord {
    name: String,
    size: runmat_value::Tensor,
    bytes: usize,
    class_name: String,
    is_global: bool,
    is_sparse: bool,
    is_complex: bool,
    nesting: String,
    persistent: bool,
}

impl WhosRecord {
    pub(super) fn name(&self) -> &str {
        &self.name
    }

    pub(super) async fn from_value(
        name: String,
        value: &Value,
        is_global: bool,
    ) -> BuiltinResult<Self> {
        let dims = value_dimensions(value).await.map_err(|mut error| {
            error.context = error.context.with_builtin("whos");
            error
        })?;
        let size_tensor = dims_to_tensor(&dims)?;
        let mut seen = HashSet::new();
        let bytes = value_memory_bytes(value, &mut seen)?;
        let class_name = class_name_for_value(value);
        let is_complex = matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
            || matches!(value, Value::GpuTensor(handle)
                if handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved);
        Ok(Self {
            name,
            size: size_tensor,
            bytes,
            class_name,
            is_global,
            is_sparse: matches!(value, Value::SparseTensor(_)),
            is_complex,
            nesting: String::new(),
            persistent: false,
        })
    }

    pub(super) fn into_value(self) -> BuiltinResult<Value> {
        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        st.fields
            .insert("name".to_string(), Value::string(self.name));
        st.fields
            .insert("size".to_string(), Value::Tensor(self.size));
        st.fields.insert(
            "bytes".to_string(),
            Value::Num(bytes_as_presentation_double(self.bytes)),
        );
        st.fields
            .insert("class".to_string(), Value::string(self.class_name));
        st.fields
            .insert("global".to_string(), Value::Bool(self.is_global));
        st.fields
            .insert("sparse".to_string(), Value::Bool(self.is_sparse));
        st.fields
            .insert("complex".to_string(), Value::Bool(self.is_complex));
        st.fields
            .insert("nesting".to_string(), Value::string(self.nesting));
        st.fields
            .insert("persistent".to_string(), Value::Bool(self.persistent));
        Ok(Value::Struct(st.finish()))
    }
}

fn bytes_as_presentation_double(bytes: usize) -> f64 {
    // The public `bytes` field is a double. Keep exact accounting in `usize`
    // through recursion, then perform the documented presentation conversion.
    bytes as f64
}

fn whos_gc_error(role: &'static str, cause: runmat_gc::GcError) -> RuntimeError {
    build_runtime_error(format!("whos: invalid {role}: {cause}"))
        .with_builtin("whos")
        .with_source(cause)
        .build()
}

fn dims_to_tensor(dims: &[usize]) -> BuiltinResult<runmat_value::Tensor> {
    let mut normalized = dims.to_vec();
    if normalized.is_empty() {
        normalized = vec![1, 1];
    }
    let data: Vec<f64> = normalized.iter().map(|dim| *dim as f64).collect();
    runmat_value::Tensor::new(data, vec![1, normalized.len()]).map_err(|err| {
        build_runtime_error(format!("whos: failed to materialize size vector: {err}"))
            .with_builtin("whos")
            .build()
    })
}

pub(super) fn value_memory_bytes(value: &Value, seen: &mut HashSet<usize>) -> BuiltinResult<usize> {
    let bytes = match value {
        Value::Num(_) => 8,
        Value::Int(i) => match i {
            runmat_value::IntValue::I8(_) | runmat_value::IntValue::U8(_) => 1,
            runmat_value::IntValue::I16(_) | runmat_value::IntValue::U16(_) => 2,
            runmat_value::IntValue::I32(_) | runmat_value::IntValue::U32(_) => 4,
            runmat_value::IntValue::I64(_) | runmat_value::IntValue::U64(_) => 8,
        },
        Value::Bool(_) => 1,
        Value::LogicalArray(la) => la.data.len(),
        Value::CharArray(ca) => ca.data().len().saturating_mul(2),
        Value::String(runmat_value::StringElement::Missing) => 0,
        Value::String(runmat_value::StringElement::Text(s)) => {
            s.encode_utf16().count().saturating_mul(2)
        }
        Value::StringArray(sa) => sa
            .data()
            .iter()
            .map(|s| {
                s.as_str()
                    .map_or(0, |text| text.encode_utf16().count().saturating_mul(2))
            })
            .fold(0usize, usize::saturating_add),
        Value::Symbolic(expr) => expr.to_string().encode_utf16().count().saturating_mul(2),
        Value::SymbolicArray(array) => array
            .data()
            .iter()
            .map(|expr| expr.to_string().encode_utf16().count().saturating_mul(2))
            .fold(0usize, usize::saturating_add),
        // `whos` reports the authoritative MATLAB array payload size, not any
        // transitional compatibility representation.
        Value::Tensor(t) => t.len().saturating_mul(t.numeric_dtype().byte_size()),
        Value::SparseTensor(t) => {
            let value_bytes = t.nnz().saturating_mul(t.value_byte_size());
            value_bytes
                .saturating_add(
                    t.row_indices
                        .len()
                        .saturating_mul(std::mem::size_of::<usize>()),
                )
                .saturating_add(
                    t.col_ptrs
                        .len()
                        .saturating_mul(std::mem::size_of::<usize>()),
                )
        }
        Value::Complex(_, _) => 16,
        Value::ComplexTensor(t) => t
            .len()
            .saturating_mul(t.numeric_dtype().byte_size())
            .saturating_mul(2),
        Value::Cell(ca) => {
            let mut total = 0usize;
            for handle in ca.data() {
                let ptr = handle as *const Value as usize;
                if seen.insert(ptr) {
                    let value = handle;
                    total = total.saturating_add(value_memory_bytes(value, seen)?);
                }
            }
            total
        }
        Value::Struct(st) => {
            let mut total = 0usize;
            for value in st.fields.values() {
                total = total.saturating_add(value_memory_bytes(value, seen)?);
            }
            total
        }
        Value::StructArray(array) => {
            let mut total = 0usize;
            let mut error = None;
            array.for_each_value(|value| {
                if error.is_none() {
                    match value_memory_bytes(value, seen) {
                        Ok(bytes) => total = total.saturating_add(bytes),
                        Err(err) => error = Some(err),
                    }
                }
            });
            if let Some(error) = error {
                return Err(error);
            }
            total
        }
        Value::GpuTensor(handle) => {
            let elem_size = if handle_is_logical(handle) {
                1
            } else {
                let lanes = if handle.physical().storage()
                    == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                {
                    2
                } else {
                    1
                };
                handle.physical().element_type().element_size() * lanes
            };
            let count = handle
                .shape()
                .iter()
                .fold(1usize, |acc, dim| acc.saturating_mul(*dim));
            count.saturating_mul(elem_size)
        }
        Value::Object(obj) => {
            let mut total = 0usize;
            for value in obj.properties().values() {
                total = total.saturating_add(value_memory_bytes(value, seen)?);
            }
            total
        }
        Value::ObjectArray(array) => {
            let mut total = 0usize;
            for value in array.data() {
                total = total.saturating_add(value_memory_bytes(value, seen)?);
            }
            total
        }
        Value::HandleObject(handle) => {
            let ptr = runmat_gc::gc_handle_addr(&handle.target);
            if seen.insert(ptr) {
                runmat_gc::gc_with_value(&handle.target, |inner| value_memory_bytes(inner, seen))
                    .map_err(|err| whos_gc_error("handle object target", err))??
            } else {
                0
            }
        }
        Value::Listener(listener) => {
            let mut total = 0usize;
            let target_ptr = runmat_gc::gc_handle_addr(&listener.target);
            if seen.insert(target_ptr) {
                total = total.saturating_add(
                    runmat_gc::gc_with_value(&listener.target, |value| {
                        value_memory_bytes(value, seen)
                    })
                    .map_err(|err| whos_gc_error("listener target", err))??,
                );
            }
            let callback_ptr = runmat_gc::gc_handle_addr(&listener.callback);
            if seen.insert(callback_ptr) {
                total = total.saturating_add(
                    runmat_gc::gc_with_value(&listener.callback, |value| {
                        value_memory_bytes(value, seen)
                    })
                    .map_err(|err| whos_gc_error("listener callback", err))??,
                );
            }
            total
        }
        Value::Closure(closure) => {
            let mut total = 0usize;
            for value in closure.captures.iter() {
                total = total.saturating_add(value_memory_bytes(value, seen)?);
            }
            total
        }
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::ScopedFunctionHandle(_) => 0,
        Value::ClassRef(name) => name.display_name().len().saturating_mul(2),
        Value::MException(exc) => {
            let base = exc
                .message
                .len()
                .saturating_mul(2)
                .saturating_add(exc.identifier.len().saturating_mul(2));
            exc.stack.iter().fold(base, |acc, frame| {
                acc.saturating_add(frame.len().saturating_mul(2))
            })
        }
        Value::OutputList(values) => {
            let mut total = 0usize;
            for value in values {
                total = total.saturating_add(value_memory_bytes(value, seen)?);
            }
            total
        }
        Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_) => 0,
        Value::Foreign(reference) => std::mem::size_of_val(reference)
            .saturating_add(reference.host_identity.len())
            .saturating_add(reference.type_identity.family.len())
            .saturating_add(reference.type_identity.name.len()),
    };
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_bytes_remain_exact_until_the_public_double_field() {
        assert_eq!(bytes_as_presentation_double(8), 8.0);
        #[cfg(target_pointer_width = "64")]
        {
            let wide = usize::try_from(9_007_199_254_740_993_u64).expect("64-bit usize");
            assert_eq!(wide, 9_007_199_254_740_993_usize);
            assert_eq!(bytes_as_presentation_double(wide), 9_007_199_254_740_992.0);
        }
    }

    #[test]
    fn invalid_gc_target_retains_whos_owner_and_original_cause() {
        let error = whos_gc_error(
            "handle object target",
            runmat_gc::GcError::InvalidPointer("stale target".into()),
        );
        assert_eq!(error.context.builtin.as_deref(), Some("whos"));
        assert!(error.message().contains("invalid handle object target"));
        assert!(error.source.is_some());
    }
}
