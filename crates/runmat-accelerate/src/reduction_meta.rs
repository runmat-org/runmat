use crate::graph::{AccelGraph, AccelNode, AccelOpCategory, ValueId};
use runmat_builtins::{IntValue, Tensor, Type, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionBehavior {
    SumLike,
    MeanLike, // sum-like with 1/reduce_len post-scale
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReductionAxes {
    Unspecified,
    All,
    Explicit(Vec<usize>),
}

#[derive(Debug, Clone)]
pub struct ReductionSignature {
    pub data_input: ValueId,
    pub dim_arg: Option<ValueId>,
    pub behavior: ReductionBehavior,
    pub axes: ReductionAxes,
}

/// Attempt to derive a generic reduction signature from a builtin reduction node without name checks.
/// Heuristics:
/// - Data input: the first input whose type is Tensor; otherwise fall back to the first input.
/// - Dim arg: the first input (after data) that is a scalar numeric or int constant (by ValueId; resolution is done by callers).
/// - Behavior: inferred via a minimal registry keyed by builtin name for post-scale choices (mean -> MeanLike). This is centralized here.
pub fn detect_reduction_signature(
    graph: &AccelGraph,
    node: &AccelNode,
) -> Option<ReductionSignature> {
    if node.category != AccelOpCategory::Reduction {
        return None;
    }
    let (name_opt, inputs) = match &node.label {
        crate::graph::AccelNodeLabel::Builtin { name } => {
            (Some(name.as_str()), node.inputs.as_slice())
        }
        _ => (None, node.inputs.as_slice()),
    };
    if inputs.is_empty() {
        return None;
    }

    // 1) Pick data input: first tensor-typed input, else inputs[0]
    let mut data_input = inputs[0];
    for &vid in inputs {
        if let Some(info) = graph.value(vid) {
            if matches!(info.ty, Type::Tensor { .. }) {
                data_input = vid;
                break;
            }
        }
    }

    // 2) Pick dim argument if present: first scalar numeric/int constant input after data input
    let mut dim_arg: Option<ValueId> = None;
    for &vid in inputs {
        if vid == data_input {
            continue;
        }
        if let Some(info) = graph.value(vid) {
            // constants resolved by callers; here we only pass the ValueId through
            if matches!(info.origin, crate::graph::ValueOrigin::Constant) {
                // allow numeric or integer constants (type system may already have Num/Int)
                if matches!(info.ty, Type::Num | Type::Int) {
                    dim_arg = Some(vid);
                    break;
                }
            }
        }
    }

    // 3) Behavior via centralized minimal registry
    let behavior = name_opt
        .map(|n| match n.to_ascii_lowercase().as_str() {
            "mean" => ReductionBehavior::MeanLike,
            // Add more here as behavior needs expand; default to SumLike when unsure
            "sum" => ReductionBehavior::SumLike,
            _ => ReductionBehavior::SumLike,
        })
        .unwrap_or(ReductionBehavior::SumLike);

    let mut axes = ReductionAxes::Unspecified;
    // Inspect the dimension argument (if constant) first
    if let Some(dim_vid) = dim_arg {
        if let Some(value) = graph.value(dim_vid).and_then(|info| info.constant.clone()) {
            if value_is_all_keyword(&value) {
                axes = ReductionAxes::All;
            } else if let Some(dims) = parse_dims_from_value(&value) {
                axes = ReductionAxes::Explicit(dims);
            }
        }
    }
    // Fallback: look for any constant input resembling a dimension/all keyword
    if matches!(axes, ReductionAxes::Unspecified) {
        for &vid in inputs {
            if vid == data_input {
                continue;
            }
            if let Some(value) = graph.value(vid).and_then(|info| info.constant.clone()) {
                if value_is_all_keyword(&value) {
                    axes = ReductionAxes::All;
                    break;
                } else if let Some(dims) = parse_dims_from_value(&value) {
                    axes = ReductionAxes::Explicit(dims);
                    break;
                }
            }
        }
    }

    Some(ReductionSignature {
        data_input,
        dim_arg,
        behavior,
        axes,
    })
}

pub fn value_is_all_keyword(value: &Value) -> bool {
    match value {
        Value::String(s) => s.eq_ignore_ascii_case("all"),
        Value::CharArray(ca) => {
            if ca.rows == 1 {
                let candidate: String = ca.data.iter().collect();
                candidate.trim().eq_ignore_ascii_case("all")
            } else {
                false
            }
        }
        Value::StringArray(sa) => sa.data.len() == 1 && sa.data[0].eq_ignore_ascii_case("all"),
        _ => false,
    }
}

fn parse_dims_from_value(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::Int(int_val) => parse_single_int(int_val),
        Value::Num(n) => parse_single_float(*n),
        Value::Tensor(t) => parse_tensor_dims(t),
        _ => None,
    }
}

fn parse_single_int(int_val: &IntValue) -> Option<Vec<usize>> {
    let raw = int_val.to_i64();
    if raw >= 1 {
        Some(vec![raw as usize])
    } else {
        None
    }
}

fn parse_single_float(value: f64) -> Option<Vec<usize>> {
    if !value.is_finite() {
        return None;
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON || rounded < 1.0 {
        return None;
    }
    Some(vec![rounded as usize])
}

fn parse_tensor_dims(tensor: &Tensor) -> Option<Vec<usize>> {
    if tensor.data.is_empty() {
        return None;
    }
    let mut dims = Vec::with_capacity(tensor.data.len());
    for value in &tensor.data {
        if let Some(parsed) = parse_single_float(*value) {
            dims.extend(parsed);
        } else {
            return None;
        }
    }
    if dims.is_empty() {
        None
    } else {
        Some(dims)
    }
}
