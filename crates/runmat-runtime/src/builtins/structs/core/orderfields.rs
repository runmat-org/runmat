//! MATLAB-compatible `orderfields` builtin.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;

use runmat_builtins::{CellArray, StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::structs::core::orderfields")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "orderfields",
    op_kind: GpuOpKind::Custom("orderfields"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only metadata manipulation; struct values that live on the GPU remain resident.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::structs::core::orderfields")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "orderfields",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Reordering fields is a metadata operation and does not participate in fusion planning.",
};

#[runtime_builtin(
    name = "orderfields",
    category = "structs/core",
    summary = "Reorder structure field definitions alphabetically or using a supplied order.",
    keywords = "orderfields,struct,reorder fields,alphabetical,struct array",
    builtin_path = "crate::builtins::structs::core::orderfields"
)]
fn orderfields_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    evaluate(value, &rest).map(|eval| eval.into_ordered_value())
}

/// Evaluate the `orderfields` builtin once and expose both outputs.
pub fn evaluate(value: Value, rest: &[Value]) -> Result<OrderFieldsEvaluation, String> {
    if rest.len() > 1 {
        return Err("orderfields: expected at most two input arguments".to_string());
    }
    let order_arg = rest.first();

    match value {
        Value::Struct(struct_value) => {
            let original: Vec<String> = struct_value.field_names().cloned().collect();
            let order = resolve_order(&struct_value, order_arg)?;
            let permutation = permutation_from(&original, &order)?;
            let permutation = permutation_tensor(permutation)?;
            let reordered = reorder_struct(&struct_value, &order)?;
            Ok(OrderFieldsEvaluation::new(
                Value::Struct(reordered),
                permutation,
            ))
        }
        Value::Cell(cell) => {
            if cell.data.is_empty() {
                let permutation = permutation_tensor(Vec::new())?;
                if let Some(arg) = order_arg {
                    if let Some(reference) = extract_reference_struct(arg)? {
                        if reference.fields.is_empty() {
                            return Ok(OrderFieldsEvaluation::new(Value::Cell(cell), permutation));
                        } else {
                            return Err("orderfields: empty struct arrays cannot adopt a non-empty reference order".to_string());
                        }
                    }
                    if let Some(names) = extract_name_list(arg)? {
                        if names.is_empty() {
                            return Ok(OrderFieldsEvaluation::new(Value::Cell(cell), permutation));
                        }
                        return Err(
                            "orderfields: struct array has no fields to reorder".to_string()
                        );
                    }
                    if let Value::Tensor(tensor) = arg {
                        if tensor.data.is_empty() {
                            return Ok(OrderFieldsEvaluation::new(Value::Cell(cell), permutation));
                        }
                        return Err(
                            "orderfields: struct array has no fields to reorder".to_string()
                        );
                    }
                    return Err("orderfields: struct array has no fields to reorder".to_string());
                }
                return Ok(OrderFieldsEvaluation::new(Value::Cell(cell), permutation));
            }
            let first = extract_struct_from_cell(&cell, 0)?;
            let original: Vec<String> = first.field_names().cloned().collect();
            let order = resolve_order(&first, order_arg)?;
            let permutation = permutation_from(&original, &order)?;
            let permutation = permutation_tensor(permutation)?;
            let reordered = reorder_struct_array(&cell, &order)?;
            Ok(OrderFieldsEvaluation::new(
                Value::Cell(reordered),
                permutation,
            ))
        }
        other => Err(format!(
            "orderfields: first argument must be a struct or struct array (got {other:?})"
        )),
    }
}

pub struct OrderFieldsEvaluation {
    ordered: Value,
    permutation: Tensor,
}

impl OrderFieldsEvaluation {
    fn new(ordered: Value, permutation: Tensor) -> Self {
        Self {
            ordered,
            permutation,
        }
    }

    pub fn into_ordered_value(self) -> Value {
        self.ordered
    }

    pub fn permutation_value(&self) -> Value {
        tensor::tensor_into_value(self.permutation.clone())
    }

    pub fn into_values(self) -> (Value, Value) {
        let perm = tensor::tensor_into_value(self.permutation);
        (self.ordered, perm)
    }
}

fn reorder_struct_array(array: &CellArray, order: &[String]) -> Result<CellArray, String> {
    let mut reordered_elems = Vec::with_capacity(array.data.len());
    for (index, handle) in array.data.iter().enumerate() {
        let value = unsafe { &*handle.as_raw() };
        let Value::Struct(st) = value else {
            return Err(format!(
                "orderfields: struct array element {} is not a struct",
                index + 1
            ));
        };
        ensure_same_field_set(order, st)?;
        let reordered = reorder_struct(st, order)?;
        reordered_elems.push(Value::Struct(reordered));
    }
    CellArray::new_with_shape(reordered_elems, array.shape.clone())
        .map_err(|e| format!("orderfields: failed to rebuild struct array: {e}"))
}

fn reorder_struct(struct_value: &StructValue, order: &[String]) -> Result<StructValue, String> {
    let mut reordered = StructValue::new();
    for name in order {
        let value = struct_value
            .fields
            .get(name)
            .ok_or_else(|| missing_field(name))?
            .clone();
        reordered.fields.insert(name.clone(), value);
    }
    Ok(reordered)
}

fn resolve_order(
    struct_value: &StructValue,
    order_arg: Option<&Value>,
) -> Result<Vec<String>, String> {
    let mut current: Vec<String> = struct_value.field_names().cloned().collect();
    if let Some(arg) = order_arg {
        if let Some(reference) = extract_reference_struct(arg)? {
            let reference_names: Vec<String> = reference.field_names().cloned().collect();
            ensure_same_field_set(&reference_names, struct_value)?;
            return Ok(reference_names);
        }

        if let Some(names) = extract_name_list(arg)? {
            ensure_same_field_set(&names, struct_value)?;
            return Ok(names);
        }

        if let Some(permutation) = extract_indices(&current, arg)? {
            return Ok(permutation);
        }

        return Err("orderfields: unrecognised ordering argument".to_string());
    }

    sort_field_names(&mut current);
    Ok(current)
}

fn permutation_from(original: &[String], order: &[String]) -> Result<Vec<f64>, String> {
    let mut index_map = HashMap::with_capacity(original.len());
    for (idx, name) in original.iter().enumerate() {
        index_map.insert(name.as_str(), idx);
    }
    let mut indices = Vec::with_capacity(order.len());
    for name in order {
        let Some(position) = index_map.get(name.as_str()) else {
            return Err(missing_field(name));
        };
        indices.push((*position as f64) + 1.0);
    }
    Ok(indices)
}

fn permutation_tensor(indices: Vec<f64>) -> Result<Tensor, String> {
    let rows = indices.len();
    let shape = vec![rows, 1];
    Tensor::new(indices, shape).map_err(|e| format!("orderfields: {e}"))
}

fn sort_field_names(names: &mut [String]) {
    names.sort_by(|a, b| {
        let lower_a = a.to_ascii_lowercase();
        let lower_b = b.to_ascii_lowercase();
        match lower_a.cmp(&lower_b) {
            Ordering::Equal => a.cmp(b),
            other => other,
        }
    });
}

fn extract_reference_struct(value: &Value) -> Result<Option<StructValue>, String> {
    match value {
        Value::Struct(st) => Ok(Some(st.clone())),
        Value::Cell(cell) => {
            let mut first: Option<StructValue> = None;
            for (index, handle) in cell.data.iter().enumerate() {
                let value = unsafe { &*handle.as_raw() };
                if let Value::Struct(st) = value {
                    if first.is_none() {
                        first = Some(st.clone());
                    }
                } else if first.is_some() {
                    return Err(format!(
                        "orderfields: reference struct array element {} is not a struct",
                        index + 1
                    ));
                } else {
                    return Ok(None);
                }
            }
            Ok(first)
        }
        _ => Ok(None),
    }
}

fn extract_name_list(arg: &Value) -> Result<Option<Vec<String>>, String> {
    match arg {
        Value::Cell(cell) => {
            let mut names = Vec::with_capacity(cell.data.len());
            for (index, handle) in cell.data.iter().enumerate() {
                let value = unsafe { &*handle.as_raw() };
                let text = scalar_string(value).ok_or_else(|| {
                    format!(
                        "orderfields: cell array element {} must be a string or character vector",
                        index + 1
                    )
                })?;
                if text.is_empty() {
                    return Err("orderfields: field names must be nonempty".to_string());
                }
                names.push(text);
            }
            Ok(Some(names))
        }
        Value::StringArray(sa) => Ok(Some(sa.data.clone())),
        Value::CharArray(ca) => {
            if ca.rows == 0 {
                return Ok(Some(Vec::new()));
            }
            let mut names = Vec::with_capacity(ca.rows);
            for row in 0..ca.rows {
                let start = row * ca.cols;
                let end = start + ca.cols;
                let mut text: String = ca.data[start..end].iter().collect();
                while text.ends_with(' ') {
                    text.pop();
                }
                if text.is_empty() {
                    return Err("orderfields: field names must be nonempty".to_string());
                }
                names.push(text);
            }
            Ok(Some(names))
        }
        _ => Ok(None),
    }
}

fn extract_indices(current: &[String], arg: &Value) -> Result<Option<Vec<String>>, String> {
    let Value::Tensor(tensor) = arg else {
        return Ok(None);
    };
    if tensor.data.is_empty() && current.is_empty() {
        return Ok(Some(Vec::new()));
    }
    if tensor.data.len() != current.len() {
        return Err("orderfields: index vector must permute every field exactly once".to_string());
    }
    let mut seen = HashSet::with_capacity(current.len());
    let mut order = Vec::with_capacity(current.len());
    for value in &tensor.data {
        if !value.is_finite() || value.fract() != 0.0 {
            return Err("orderfields: index vector must contain integers".to_string());
        }
        let idx = *value as isize;
        if idx < 1 || idx as usize > current.len() {
            return Err("orderfields: index vector element out of range".to_string());
        }
        let zero_based = (idx as usize) - 1;
        if !seen.insert(zero_based) {
            return Err("orderfields: index vector contains duplicate positions".to_string());
        }
        order.push(current[zero_based].clone());
    }
    Ok(Some(order))
}

fn ensure_same_field_set(order: &[String], original: &StructValue) -> Result<(), String> {
    if order.len() != original.fields.len() {
        return Err("orderfields: field names must match the struct exactly".to_string());
    }
    let mut seen = HashSet::with_capacity(order.len());
    let original_set: HashSet<&str> = original.field_names().map(|s| s.as_str()).collect();
    for name in order {
        if !original_set.contains(name.as_str()) {
            return Err(format!(
                "orderfields: unknown field '{name}' in requested order"
            ));
        }
        if !seen.insert(name.as_str()) {
            return Err(format!(
                "orderfields: duplicate field '{name}' in requested order"
            ));
        }
    }
    Ok(())
}

fn extract_struct_from_cell(cell: &CellArray, index: usize) -> Result<StructValue, String> {
    let value = unsafe { &*cell.data[index].as_raw() };
    match value {
        Value::Struct(st) => Ok(st.clone()),
        other => Err(format!(
            "orderfields: expected struct array contents to be structs (found {other:?})"
        )),
    }
}

fn scalar_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => {
            let mut text: String = ca.data.iter().collect();
            while text.ends_with(' ') {
                text.pop();
            }
            Some(text)
        }
        _ => None,
    }
}

fn missing_field(name: &str) -> String {
    format!("orderfields: field '{name}' does not exist on the struct")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{CellArray, CharArray, StringArray, Tensor};

    fn field_order(struct_value: &StructValue) -> Vec<String> {
        struct_value.field_names().cloned().collect()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn default_sorts_alphabetically() {
        let mut st = StructValue::new();
        st.fields.insert("beta".to_string(), Value::Num(2.0));
        st.fields.insert("alpha".to_string(), Value::Num(1.0));
        st.fields.insert("gamma".to_string(), Value::Num(3.0));

        let result = orderfields_builtin(Value::Struct(st), Vec::new()).expect("orderfields");
        let Value::Struct(sorted) = result else {
            panic!("expected struct result");
        };
        assert_eq!(
            field_order(&sorted),
            vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reorder_with_cell_name_list() {
        let mut st = StructValue::new();
        st.fields.insert("a".to_string(), Value::Num(1.0));
        st.fields.insert("b".to_string(), Value::Num(2.0));
        st.fields.insert("c".to_string(), Value::Num(3.0));
        let names = CellArray::new(
            vec![Value::from("c"), Value::from("a"), Value::from("b")],
            1,
            3,
        )
        .expect("cell");

        let reordered =
            orderfields_builtin(Value::Struct(st), vec![Value::Cell(names)]).expect("orderfields");
        let Value::Struct(result) = reordered else {
            panic!("expected struct result");
        };
        assert_eq!(
            field_order(&result),
            vec!["c".to_string(), "a".to_string(), "b".to_string()]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reorder_with_string_array_names() {
        let mut st = StructValue::new();
        st.fields.insert("alpha".to_string(), Value::Num(1.0));
        st.fields.insert("beta".to_string(), Value::Num(2.0));
        st.fields.insert("gamma".to_string(), Value::Num(3.0));

        let strings = StringArray::new(
            vec!["gamma".into(), "alpha".into(), "beta".into()],
            vec![1, 3],
        )
        .expect("string array");

        let result = orderfields_builtin(Value::Struct(st), vec![Value::StringArray(strings)])
            .expect("orderfields");
        let Value::Struct(sorted) = result else {
            panic!("expected struct result");
        };
        assert_eq!(
            field_order(&sorted),
            vec!["gamma".to_string(), "alpha".to_string(), "beta".to_string()]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reorder_with_char_array_names() {
        let mut st = StructValue::new();
        st.fields.insert("cat".to_string(), Value::Num(1.0));
        st.fields.insert("ant".to_string(), Value::Num(2.0));
        st.fields.insert("bat".to_string(), Value::Num(3.0));

        let data = vec!['b', 'a', 't', 'c', 'a', 't', 'a', 'n', 't'];
        let char_array = CharArray::new(data, 3, 3).expect("char array");

        let result = orderfields_builtin(Value::Struct(st), vec![Value::CharArray(char_array)])
            .expect("order");
        let Value::Struct(sorted) = result else {
            panic!("expected struct result");
        };
        assert_eq!(
            field_order(&sorted),
            vec!["bat".to_string(), "cat".to_string(), "ant".to_string()]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reorder_with_reference_struct() {
        let mut source = StructValue::new();
        source.fields.insert("y".to_string(), Value::Num(2.0));
        source.fields.insert("x".to_string(), Value::Num(1.0));

        let mut reference = StructValue::new();
        reference.fields.insert("x".to_string(), Value::Num(0.0));
        reference.fields.insert("y".to_string(), Value::Num(0.0));

        let result = orderfields_builtin(
            Value::Struct(source),
            vec![Value::Struct(reference.clone())],
        )
        .expect("orderfields");
        let Value::Struct(reordered) = result else {
            panic!("expected struct result");
        };
        assert_eq!(
            field_order(&reordered),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reorder_with_index_vector() {
        let mut st = StructValue::new();
        st.fields.insert("first".to_string(), Value::Num(1.0));
        st.fields.insert("second".to_string(), Value::Num(2.0));
        st.fields.insert("third".to_string(), Value::Num(3.0));

        let permutation = Tensor::new(vec![3.0, 1.0, 2.0], vec![1, 3]).expect("tensor permutation");
        let result = orderfields_builtin(Value::Struct(st), vec![Value::Tensor(permutation)])
            .expect("orderfields");
        let Value::Struct(reordered) = result else {
            panic!("expected struct result");
        };
        assert_eq!(
            field_order(&reordered),
            vec![
                "third".to_string(),
                "first".to_string(),
                "second".to_string()
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn index_vector_must_be_integers() {
        let mut st = StructValue::new();
        st.fields.insert("one".to_string(), Value::Num(1.0));
        st.fields.insert("two".to_string(), Value::Num(2.0));

        let permutation = Tensor::new(vec![1.0, 1.5], vec![1, 2]).expect("tensor");
        let err =
            orderfields_builtin(Value::Struct(st), vec![Value::Tensor(permutation)]).unwrap_err();
        assert!(
            err.contains("index vector must contain integers"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn permutation_vector_matches_original_positions() {
        let mut st = StructValue::new();
        st.fields.insert("beta".to_string(), Value::Num(2.0));
        st.fields.insert("alpha".to_string(), Value::Num(1.0));
        st.fields.insert("gamma".to_string(), Value::Num(3.0));

        let eval = evaluate(Value::Struct(st), &[]).expect("evaluate");
        let perm = eval.permutation_value();
        match perm {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 1.0, 3.0]),
            other => panic!("expected tensor permutation, got {other:?}"),
        }
        let Value::Struct(ordered) = eval.into_ordered_value() else {
            panic!("expected struct result");
        };
        assert_eq!(
            field_order(&ordered),
            vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reorder_struct_array() {
        let mut first = StructValue::new();
        first.fields.insert("b".to_string(), Value::Num(1.0));
        first.fields.insert("a".to_string(), Value::Num(2.0));
        let mut second = StructValue::new();
        second.fields.insert("b".to_string(), Value::Num(3.0));
        second.fields.insert("a".to_string(), Value::Num(4.0));
        let array = CellArray::new_with_shape(
            vec![Value::Struct(first), Value::Struct(second)],
            vec![1, 2],
        )
        .expect("struct array");
        let names =
            CellArray::new(vec![Value::from("a"), Value::from("b")], 1, 2).expect("cell names");

        let result =
            orderfields_builtin(Value::Cell(array), vec![Value::Cell(names)]).expect("orderfields");
        let Value::Cell(reordered) = result else {
            panic!("expected cell array");
        };
        for handle in &reordered.data {
            let Value::Struct(st) = (unsafe { &*handle.as_raw() }) else {
                panic!("expected struct element");
            };
            assert_eq!(field_order(st), vec!["a".to_string(), "b".to_string()]);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_array_permutation_reuses_order() {
        let mut first = StructValue::new();
        first.fields.insert("z".to_string(), Value::Num(1.0));
        first.fields.insert("x".to_string(), Value::Num(2.0));
        first.fields.insert("y".to_string(), Value::Num(3.0));

        let mut second = StructValue::new();
        second.fields.insert("z".to_string(), Value::Num(4.0));
        second.fields.insert("x".to_string(), Value::Num(5.0));
        second.fields.insert("y".to_string(), Value::Num(6.0));

        let array = CellArray::new_with_shape(
            vec![Value::Struct(first), Value::Struct(second)],
            vec![1, 2],
        )
        .expect("struct array");

        let eval = evaluate(Value::Cell(array), &[]).expect("evaluate");
        let perm = eval.permutation_value();
        match perm {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 3.0, 1.0]),
            other => panic!("expected tensor permutation, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_unknown_field() {
        let mut st = StructValue::new();
        st.fields.insert("alpha".to_string(), Value::Num(1.0));
        st.fields.insert("beta".to_string(), Value::Num(2.0));
        let err = orderfields_builtin(
            Value::Struct(st),
            vec![Value::Cell(
                CellArray::new(vec![Value::from("beta"), Value::from("gamma")], 1, 2)
                    .expect("cell"),
            )],
        )
        .unwrap_err();
        assert!(
            err.contains("unknown field 'gamma'"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn duplicate_field_names_rejected() {
        let mut st = StructValue::new();
        st.fields.insert("alpha".to_string(), Value::Num(1.0));
        st.fields.insert("beta".to_string(), Value::Num(2.0));

        let names =
            CellArray::new(vec![Value::from("alpha"), Value::from("alpha")], 1, 2).expect("cell");
        let err = orderfields_builtin(Value::Struct(st), vec![Value::Cell(names)]).unwrap_err();
        assert!(
            err.contains("duplicate field 'alpha'"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reference_struct_mismatch_errors() {
        let mut source = StructValue::new();
        source.fields.insert("x".to_string(), Value::Num(1.0));
        source.fields.insert("y".to_string(), Value::Num(2.0));

        let mut reference = StructValue::new();
        reference.fields.insert("x".to_string(), Value::Num(0.0));

        let err =
            orderfields_builtin(Value::Struct(source), vec![Value::Struct(reference)]).unwrap_err();
        assert!(
            err.contains("field names must match the struct exactly"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invalid_order_argument_type_errors() {
        let mut st = StructValue::new();
        st.fields.insert("x".to_string(), Value::Num(1.0));

        let err = orderfields_builtin(Value::Struct(st), vec![Value::Num(1.0)]).unwrap_err();
        assert!(
            err.contains("unrecognised ordering argument"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_struct_array_nonempty_reference_errors() {
        let empty = CellArray::new(Vec::new(), 0, 0).expect("empty struct array");
        let mut reference = StructValue::new();
        reference
            .fields
            .insert("field".to_string(), Value::Num(1.0));

        let err =
            orderfields_builtin(Value::Cell(empty), vec![Value::Struct(reference)]).unwrap_err();
        assert!(
            err.contains("empty struct arrays cannot adopt a non-empty reference order"),
            "unexpected error: {err}"
        );
    }
}
