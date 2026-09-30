//! Pure reachability admission for asynchronous resident reverse. Snapshot
//! immutable records while the tape is borrowed, then release it before any
//! provider future is polled.

use super::super::super::tape::NodeValue;
use super::*;
use std::{collections::HashSet, rc::Rc};

pub(super) struct ResidentGraph {
    pub loss: runmat_accelerate_api::GpuTensorHandle,
    pub executions: Vec<(usize, Rc<super::super::super::network::ResidentTapeRecord>)>,
}

pub(super) fn admit(loss_node: usize) -> BuiltinResult<ResidentGraph> {
    let tape = active_tape().ok_or_else(|| {
        deep_learning_error(
            "dlgradient",
            "dlgradient: no active automatic differentiation tape",
        )
    })?;
    let tape = tape.borrow();
    let nodes = &tape.nodes;
    let Some(loss) = nodes.get(loss_node) else {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: invalid loss node",
        ));
    };
    let NodeValue::Resident(loss_value) = &loss.value else {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: resident reverse requires a resident loss",
        ));
    };
    if !matches!(loss.kind, NodeKind::ResidentNetwork { .. }) {
        return Err(unsupported_error(
            "dlgradient",
            "dlgradient: resident loss requires an executed resident network trace",
        ));
    }
    let mut seen = HashSet::new();
    let mut pending = vec![loss_node];
    let mut executions = Vec::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = nodes.get(id).ok_or_else(|| {
            deep_learning_error("dlgradient", "dlgradient: resident tape parent is missing")
        })?;
        match &node.kind {
            NodeKind::Leaf => {}
            NodeKind::ResidentNetwork { execution } => {
                let NodeValue::Resident(output) = &node.value else {
                    return Err(deep_learning_error(
                        "dlgradient",
                        "dlgradient: resident tape node has host output",
                    ));
                };
                if output.view_identity() != execution.trace.output.view_identity() {
                    return Err(deep_learning_error(
                        "dlgradient",
                        "dlgradient: resident tape output differs from executed trace",
                    ));
                }
                if output.shape() != execution.presented_axes_shape() {
                    return Err(deep_learning_error(
                        "dlgradient",
                        "dlgradient: presented resident axes differ from tape output",
                    ));
                }
                let output_dtype = node.value.numeric_dtype().ok_or_else(|| {
                    deep_learning_error(
                        "dlgradient",
                        "dlgradient: resident execution has no floating class",
                    )
                })?;
                for (parent, shape, dtype) in execution.parent_bindings() {
                    if parent >= id {
                        return Err(deep_learning_error(
                            "dlgradient",
                            "dlgradient: resident tape parent order is invalid",
                        ));
                    }
                    let recorded = nodes.get(parent).ok_or_else(|| {
                        deep_learning_error(
                            "dlgradient",
                            "dlgradient: resident tape parent is missing",
                        )
                    })?;
                    if dtype != output_dtype
                        || recorded.value.shape() != shape
                        || recorded.value.numeric_dtype() != Some(dtype)
                    {
                        return Err(deep_learning_error(
                            "dlgradient",
                            "dlgradient: resident parent geometry or class changed",
                        ));
                    }
                    if let NodeValue::Resident(parent_value) = &recorded.value {
                        crate::builtins::common::gpu_helpers::exact_provider_for_binary_inputs(
                            output,
                            parent_value,
                        )
                        .map_err(|_| {
                            deep_learning_error(
                                "dlgradient",
                                "dlgradient: resident parent owner differs from network output",
                            )
                        })?;
                    }
                    pending.push(parent);
                }
                executions.push((id, execution.clone()));
            }
            _ => {
                return Err(unsupported_error(
                    "dlgradient",
                    "dlgradient: connected operation has no resident adjoint",
                ));
            }
        }
    }
    executions.sort_unstable_by(|(lhs, _), (rhs, _)| rhs.cmp(lhs));
    Ok(ResidentGraph {
        loss: loss_value.clone(),
        executions,
    })
}
