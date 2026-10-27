//! Class-preserving numeric JSON scalars and bounded sparse previews.
use runmat_value::{IntValue, NumericScalar};
use serde_json::{json, Value as JsonValue};

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

pub(super) fn numeric_scalar_json_value(value: NumericScalar) -> JsonValue {
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

pub(super) fn sparse_entry_preview(
    st: &runmat_value::SparseTensor,
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

pub(super) fn sparse_values_preview(
    st: &runmat_value::SparseTensor,
    limit: usize,
) -> (Vec<JsonValue>, bool) {
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
