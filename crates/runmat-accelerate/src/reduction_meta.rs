use crate::graph::{AccelGraph, AccelNode, AccelOpCategory, ValueId};
use runmat_builtins::Type;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionBehavior {
    SumLike,
    MeanLike, // sum-like with 1/reduce_len post-scale
}

#[derive(Debug, Clone, Copy)]
pub struct ReductionSignature {
    pub data_input: ValueId,
    pub dim_arg: Option<ValueId>,
    pub behavior: ReductionBehavior,
}

/// Attempt to derive a generic reduction signature from a builtin reduction node without name checks.
/// Heuristics:
/// - Data input: the first input whose type is Tensor; otherwise fall back to the first input.
/// - Dim arg: the first input (after data) that is a scalar numeric or int constant (by ValueId; resolution is done by callers).
/// - Behavior: inferred via a minimal registry keyed by builtin name for post-scale choices (mean -> MeanLike). This is centralized here.
pub fn detect_reduction_signature(graph: &AccelGraph, node: &AccelNode) -> Option<ReductionSignature> {
    if node.category != AccelOpCategory::Reduction {
        return None;
    }
    let (name_opt, inputs) = match &node.label {
        crate::graph::AccelNodeLabel::Builtin { name } => (Some(name.as_str()), node.inputs.as_slice()),
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

    Some(ReductionSignature {
        data_input,
        dim_arg,
        behavior,
    })
}


