//! Existing scalar sparse geometry feeding the same canonical CSC owners.
use super::*;
use crate::context::RuntimeContext;
use crate::indexing::{
    plan::admission::Owner,
    read_slice::sparse::output::Element,
    write_slice::sparse::{
        delete::{rebuild, Axis},
        merge,
    },
};
use runmat_value::ComplexElement;

pub(super) async fn assign(
    context: Option<&RuntimeContext>,
    source: SparseTensor,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    if source.integer_storage().is_some() {
        crate::compatibility::ensure_sparse_integer_extension_enabled("indexed assignment")?;
    }
    if delete {
        if !is_empty_tensor(rhs) {
            return Err(mex(
                "DeletionRequiresEmptyRhs",
                "Indexed deletion requires empty RHS",
            ));
        }
        let (axis, index) = match indices {
            [index] => {
                let total = source
                    .rows
                    .checked_mul(source.cols)
                    .ok_or_else(|| mex("IndexOutOfBounds", "Index out of bounds"))?;
                if *index == 0 || *index > total {
                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                }
                if source.rows == 1 {
                    (Axis::Columns, index - 1)
                } else if source.cols == 1 {
                    (Axis::Rows, index - 1)
                } else {
                    return Err(mex(
                        "UnsupportedDeletion",
                        "Linear sparse deletion is only supported for vectors",
                    ));
                }
            }
            [row, column] => {
                if *row == 0 || *row > source.rows || *column == 0 || *column > source.cols {
                    return Err(mex("SubscriptOutOfBounds", "Subscript out of bounds"));
                }
                if source.rows == 1 {
                    (Axis::Columns, column - 1)
                } else if source.cols == 1 {
                    (Axis::Rows, row - 1)
                } else {
                    return Err(mex(
                        "UnsupportedDeletion",
                        "Sparse deletion requires selecting complete rows or columns",
                    ));
                }
            }
            _ => {
                return Err(mex(
                    "UnsupportedDeletion",
                    "Sparse scalar deletion is only supported for vector indices",
                ))
            }
        };
        return rebuild(context, source, axis, &[index]);
    }
    let (rows, cols, row, column) = match indices {
        [index] => {
            let total = source
                .rows
                .checked_mul(source.cols)
                .ok_or_else(|| mex("IndexOutOfBounds", "Index out of bounds"))?;
            if *index == 0 {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            if *index > total {
                if source.rows == 1 || (source.rows == 0 && source.cols == 0) {
                    (1, *index, 0, index - 1)
                } else if source.cols == 1 {
                    (*index, 1, index - 1, 0)
                } else {
                    return Err(mex(
                        "IndexOutOfBounds",
                        "Linear sparse growth is only supported for vectors",
                    ));
                }
            } else {
                (
                    source.rows,
                    source.cols,
                    (index - 1) % source.rows,
                    (index - 1) / source.rows,
                )
            }
        }
        [row, column] => {
            if *row == 0 || *column == 0 {
                return Err(mex("SubscriptOutOfBounds", "Subscript out of bounds"));
            }
            (
                source.rows.max(*row),
                source.cols.max(*column),
                row - 1,
                column - 1,
            )
        }
        _ => {
            return Err(mex(
                "UnsupportedAssignmentRank",
                "Only 1D/2D scalar assignment supported",
            ))
        }
    };
    cols.checked_add(1)
        .ok_or_else(|| map_assignment_shape_error("SparseTensor expanded column count overflow"))?;
    let integer_gpu = source.integer_storage().is_some()
        && matches!(rhs, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some());
    let gathered;
    let rhs = if let Value::GpuTensor(handle) = rhs {
        if !integer_gpu && !handle.shape().iter().all(|&extent| extent == 1) {
            return Err(mex("ScalarRequired", "RHS must be scalar"));
        }
        gathered = owner
            .await_stage(
                crate::indexing::write_slice::gather_assignment_rhs_in_context(
                    context,
                    rhs,
                    source.integer_storage().is_some(),
                ),
            )
            .await??;
        if integer_gpu && !matches!(&gathered, Value::Tensor(tensor) if tensor.len() == 1) {
            return Err(mex(
                "InvalidAssignmentRhs",
                "integer gpuArray assignment rhs must be scalar",
            ));
        }
        &gathered
    } else {
        rhs
    };
    let element = if source.is_logical() {
        Element::Logical(
            rhs_to_real_numeric_scalar(context, rhs)
                .await?
                .materialize_f64()
                != 0.0,
        )
    } else if let Some(storage) = source.integer_storage() {
        Element::Numeric(
            integer_assignment::scalar_value(
                storage,
                &rhs_to_integer_assignment_scalar(context, rhs).await?,
            )
            .into(),
        )
    } else if source.is_complex() {
        let (real, imaginary) = rhs_to_complex_numeric_scalar(context, rhs).await?;
        if source.numeric_dtype() == Some(NumericDType::F32) {
            Element::Complex32(ComplexElement(
                real.materialize_f32(),
                imaginary.materialize_f32(),
            ))
        } else {
            Element::Complex64(ComplexElement(
                real.materialize_f64(),
                imaginary.materialize_f64(),
            ))
        }
    } else {
        let value = rhs_to_real_numeric_scalar(context, rhs).await?;
        Element::Numeric(if source.numeric_dtype() == Some(NumericDType::F32) {
            NumericScalar::F32(value.materialize_f32())
        } else {
            NumericScalar::F64(value.materialize_f64())
        })
    };
    owner.check()?;
    let linear = column
        .checked_mul(rows)
        .and_then(|offset| offset.checked_add(row))
        .ok_or_else(|| {
            map_assignment_shape_error("SparseTensor assignment linear index overflow")
        })?;
    merge(context, source, rows, cols, &[(linear, 0)], |_, _| element)
}
