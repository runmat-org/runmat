//! Host RHS expansion shared by direct Native stores and resident-RHS adapters.

use super::{checked_assignment_element_count, mex, validated_assignment_rhs_shape, RuntimeError};
use crate::builtins::common::tensor::{tensor_element_len, tensor_value_f64};
use runmat_value::Value;

pub(super) fn linear(rhs: &Value, count: usize) -> Result<Vec<f64>, RuntimeError> {
    match rhs {
        Value::Num(n) => Ok(vec![*n; count]),
        Value::Int(int_val) => Ok(vec![int_val.to_f64(); count]),
        Value::Bool(b) => Ok(vec![if *b { 1.0 } else { 0.0 }; count]),
        Value::Tensor(t) => {
            let len = tensor_element_len(t);
            if len == count {
                Ok(t.materialize_f64())
            } else if len == 1 {
                Ok(vec![tensor_value_f64(t, 0); count])
            } else {
                Err(mex("ShapeMismatch", "shape mismatch for slice assign"))
            }
        }
        Value::LogicalArray(la) => {
            if la.data.len() == count {
                Ok(la
                    .data
                    .iter()
                    .map(|b| if *b != 0 { 1.0 } else { 0.0 })
                    .collect())
            } else if la.data.len() == 1 {
                let val = if la.data[0] != 0 { 1.0 } else { 0.0 };
                Ok(vec![val; count])
            } else {
                Err(mex("ShapeMismatch", "shape mismatch for slice assign"))
            }
        }
        other => Err(mex(
            "InvalidSliceAssignmentRhs",
            format!("slice assign: unsupported RHS type {:?}", other),
        )),
    }
}

pub(super) fn multidimensional(
    rhs: &Value,
    selection_lengths: &[usize],
) -> Result<Vec<f64>, RuntimeError> {
    enum RhsView {
        Scalar(f64),
        Tensor {
            data: Vec<f64>,
            shape: Vec<usize>,
            strides: Vec<usize>,
        },
    }
    let view = match rhs {
        Value::Num(n) => RhsView::Scalar(*n),
        Value::Int(iv) => RhsView::Scalar(iv.to_f64()),
        Value::Bool(b) => RhsView::Scalar(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) => {
            let shape = validated_assignment_rhs_shape(
                t.shape(),
                tensor_element_len(t),
                selection_lengths,
            )?;
            let mut strides = vec![1usize; selection_lengths.len()];
            for d in 1..selection_lengths.len() {
                strides[d] = strides[d - 1] * shape[d - 1].max(1);
            }
            let data = t.materialize_f64();
            if data.len() != checked_assignment_element_count(&shape)? {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            RhsView::Tensor {
                data,
                shape,
                strides,
            }
        }
        Value::LogicalArray(la) => {
            let shape =
                validated_assignment_rhs_shape(&la.shape, la.data.len(), selection_lengths)?;
            let mut strides = vec![1usize; selection_lengths.len()];
            for d in 1..selection_lengths.len() {
                strides[d] = strides[d - 1] * shape[d - 1].max(1);
            }
            if la.data.len() != checked_assignment_element_count(&shape)? {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            let data = la
                .data
                .iter()
                .map(|b| if *b != 0 { 1.0 } else { 0.0 })
                .collect();
            RhsView::Tensor {
                data,
                shape,
                strides,
            }
        }
        other => {
            return Err(mex(
                "InvalidSliceAssignmentRhs",
                format!("slice assign: unsupported RHS type {:?}", other),
            ))
        }
    };

    let total = checked_assignment_element_count(selection_lengths)?;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; selection_lengths.len()];
    if total == 0 || selection_lengths.is_empty() {
        return Ok(out);
    }
    loop {
        match &view {
            RhsView::Scalar(val) => out.push(*val),
            RhsView::Tensor {
                data,
                shape,
                strides,
            } => {
                let mut rlin = 0usize;
                for d in 0..idx.len() {
                    let rhs_len = shape[d];
                    let pos = if rhs_len == 1 { 0 } else { idx[d] };
                    rlin += pos * strides[d];
                }
                out.push(
                    *data
                        .get(rlin)
                        .ok_or_else(|| mex("ShapeMismatch", "shape mismatch for slice assign"))?,
                );
            }
        }
        let mut d = 0usize;
        while d < idx.len() {
            idx[d] += 1;
            if idx[d] < selection_lengths[d].max(1) {
                break;
            }
            idx[d] = 0;
            d += 1;
        }
        if d == idx.len() {
            break;
        }
    }
    Ok(out)
}
