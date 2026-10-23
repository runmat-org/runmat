use crate::builtins::common::tensor::{is_scalar_tensor, tensor_element_len};
use crate::indexing::integer_assignment::{
    self, ComplexIntegerAssignmentValue, IntegerAssignmentValue,
};
use crate::indexing::write_slice::{
    delete_complex_at_index_in_context, delete_tensor_at_index_in_context,
    download_integer_tensor_in_context,
};
use crate::runtime_error::semantic_error as mex;
use crate::RuntimeError;
use runmat_value::{
    ComplexTensor, IntValue, IntegerStorage, NumericDType, NumericScalar, SparseTensor, Tensor,
    Value,
};

mod dense_scalar;
mod sparse_scalar;
use dense_scalar::ScalarShape;

fn map_assignment_shape_error(err: impl std::fmt::Display) -> RuntimeError {
    mex("ShapeMismatch", format!("assignment: {err}"))
}

fn dense_linear_dimensions(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (0, 0),
        [cols] => (1, *cols),
        [rows, cols, ..] => (*rows, *cols),
    }
}

fn is_empty_tensor(value: &Value) -> bool {
    matches!(value, Value::Tensor(t) if {
        let (rows, cols) = dense_linear_dimensions(t.shape());
        tensor_element_len(t) == 0 || rows == 0 || cols == 0
    }) || matches!(value, Value::ComplexTensor(t) if t.is_empty() || t.rows() == 0 || t.cols() == 0)
}

fn integer_storage_scalar(storage: &IntegerStorage) -> IntValue {
    match storage {
        IntegerStorage::I8(values) => IntValue::I8(values[0]),
        IntegerStorage::I16(values) => IntValue::I16(values[0]),
        IntegerStorage::I32(values) => IntValue::I32(values[0]),
        IntegerStorage::I64(values) => IntValue::I64(values[0]),
        IntegerStorage::U8(values) => IntValue::U8(values[0]),
        IntegerStorage::U16(values) => IntValue::U16(values[0]),
        IntegerStorage::U32(values) => IntValue::U32(values[0]),
        IntegerStorage::U64(values) => IntValue::U64(values[0]),
    }
}

async fn rhs_to_integer_assignment_scalar(
    context: Option<&crate::context::RuntimeContext>,
    rhs: &Value,
) -> Result<IntegerAssignmentValue, RuntimeError> {
    match rhs {
        Value::Int(value) => Ok(IntegerAssignmentValue::Exact(value.clone())),
        Value::Num(value) => Ok(IntegerAssignmentValue::Float(*value)),
        Value::Bool(value) => Ok(IntegerAssignmentValue::Float(if *value {
            1.0
        } else {
            0.0
        })),
        Value::Tensor(tensor) if is_scalar_tensor(tensor) => {
            let value = tensor
                .numeric_value_at(0)
                .expect("scalar tensor must contain one numeric value");
            Ok(match value.into_int_value() {
                Some(value) => IntegerAssignmentValue::Exact(value),
                None => IntegerAssignmentValue::Float(value.materialize_f64()),
            })
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok(IntegerAssignmentValue::Float(if array.data[0] == 0 {
                0.0
            } else {
                1.0
            }))
        }
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_some() =>
        {
            let tensor = download_integer_tensor_in_context(context, handle).await?;
            let storage = tensor
                .integer_storage()
                .expect("exact integer GPU gather must retain integer storage");
            if storage.len() != 1 {
                return Err(mex(
                    "InvalidAssignmentRhs",
                    "integer gpuArray assignment rhs must be scalar",
                ));
            }
            Ok(IntegerAssignmentValue::Exact(integer_storage_scalar(
                storage,
            )))
        }
        _ => rhs_to_real_numeric_scalar(context, rhs)
            .await
            .map(|value| IntegerAssignmentValue::Float(value.materialize_f64())),
    }
}

async fn rhs_to_complex_integer_assignment_scalar(
    context: Option<&crate::context::RuntimeContext>,
    rhs: &Value,
) -> Result<ComplexIntegerAssignmentValue, RuntimeError> {
    let scalar = |real, imag| ComplexIntegerAssignmentValue {
        real: IntegerAssignmentValue::Float(real),
        imag: IntegerAssignmentValue::Float(imag),
    };
    match rhs {
        Value::Complex(real, imag) => Ok(scalar(*real, *imag)),
        Value::ComplexTensor(tensor) if tensor.len() == 1 => {
            if let Some(storage) = &tensor.integer_storage() {
                return Ok(ComplexIntegerAssignmentValue {
                    real: IntegerAssignmentValue::Exact(
                        storage.real().value_at(0).expect("scalar component"),
                    ),
                    imag: IntegerAssignmentValue::Exact(
                        storage.imag().value_at(0).expect("scalar component"),
                    ),
                });
            }
            let (real, imag) = tensor
                .numeric_value_at(0)
                .expect("validated scalar complex storage");
            Ok(scalar(real.materialize_f64(), imag.materialize_f64()))
        }
        _ => Ok(ComplexIntegerAssignmentValue {
            real: rhs_to_integer_assignment_scalar(context, rhs).await?,
            imag: IntegerAssignmentValue::Float(0.0),
        }),
    }
}

fn delete_tensor_linear(
    context: Option<&crate::context::RuntimeContext>,
    t: Tensor,
    idx: usize,
) -> Result<Value, RuntimeError> {
    let (rows, cols) = dense_linear_dimensions(t.shape());
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| mex("IndexOutOfBounds", "Index out of bounds"))?;
    if idx == 0 || idx > total {
        return Err(mex("IndexOutOfBounds", "Index out of bounds"));
    }
    if !(rows == 1 || cols == 1) {
        return Err(mex(
            "UnsupportedDeletion",
            "Linear deletion is only supported for vectors",
        ));
    }
    delete_tensor_at_index_in_context(context, t, idx - 1)
}

fn delete_complex_linear(
    context: Option<&crate::context::RuntimeContext>,
    t: ComplexTensor,
    idx: usize,
) -> Result<Value, RuntimeError> {
    let total = t
        .rows()
        .checked_mul(t.cols())
        .ok_or_else(|| mex("IndexOutOfBounds", "Index out of bounds"))?;
    if idx == 0 || idx > total {
        return Err(mex("IndexOutOfBounds", "Index out of bounds"));
    }
    if !(t.rows() == 1 || t.cols() == 1) {
        return Err(mex(
            "UnsupportedDeletion",
            "Linear deletion is only supported for vectors",
        ));
    }
    delete_complex_at_index_in_context(context, t, idx - 1)
}

pub async fn rhs_to_real_scalar(rhs: &Value) -> Result<f64, RuntimeError> {
    let context = crate::context::legacy::active();
    rhs_to_real_numeric_scalar(context.as_ref(), rhs)
        .await
        .map(|value| value.materialize_f64())
}

async fn rhs_to_real_numeric_scalar(
    context: Option<&crate::context::RuntimeContext>,
    rhs: &Value,
) -> Result<NumericScalar, RuntimeError> {
    if let Value::GpuTensor(handle) = rhs {
        if !handle.shape().iter().all(|&extent| extent == 1) {
            return Err(mex("ScalarRequired", "RHS must be scalar"));
        }
        let owner = super::plan::admission::Owner::new(context)?;
        let host = owner
            .await_stage(super::resident::gather_in_context(context, handle))
            .await??;
        return rhs_to_real_numeric_scalar_host(&host);
    }
    rhs_to_real_numeric_scalar_host(rhs)
}

fn rhs_to_real_numeric_scalar_host(rhs: &Value) -> Result<NumericScalar, RuntimeError> {
    match rhs {
        Value::Num(x) => Ok(NumericScalar::F64(*x)),
        Value::Int(value) => Ok(NumericScalar::from(value.clone())),
        Value::Tensor(t2) => {
            if is_scalar_tensor(t2) {
                Ok(t2.numeric_value_at(0).expect("validated scalar storage"))
            } else {
                Err(mex("ScalarRequired", "RHS must be scalar"))
            }
        }
        _ => <f64>::try_from(rhs)
            .map(NumericScalar::F64)
            .map_err(|_| mex("NumericRequired", "RHS must be numeric")),
    }
}

pub async fn rhs_to_complex_scalar(rhs: &Value) -> Result<(f64, f64), RuntimeError> {
    let context = crate::context::legacy::active();
    rhs_to_complex_numeric_scalar(context.as_ref(), rhs)
        .await
        .map(|(real, imag)| (real.materialize_f64(), imag.materialize_f64()))
}

async fn rhs_to_complex_numeric_scalar(
    context: Option<&crate::context::RuntimeContext>,
    rhs: &Value,
) -> Result<(NumericScalar, NumericScalar), RuntimeError> {
    if let Value::GpuTensor(handle) = rhs {
        if !handle.shape().iter().all(|&extent| extent == 1) {
            return Err(mex("ScalarRequired", "RHS must be scalar"));
        }
        let owner = super::plan::admission::Owner::new(context)?;
        let host = owner
            .await_stage(super::resident::gather_in_context(context, handle))
            .await??;
        return rhs_to_complex_numeric_scalar_host(&host);
    }
    rhs_to_complex_numeric_scalar_host(rhs)
}

fn rhs_to_complex_numeric_scalar_host(
    rhs: &Value,
) -> Result<(NumericScalar, NumericScalar), RuntimeError> {
    let zero = NumericScalar::F64(0.0);
    match rhs {
        Value::Complex(re, im) => Ok((NumericScalar::F64(*re), NumericScalar::F64(*im))),
        Value::Num(n) => Ok((NumericScalar::F64(*n), zero)),
        Value::Int(i) => Ok((NumericScalar::from(i.clone()), zero)),
        Value::Bool(b) => Ok((NumericScalar::F64(if *b { 1.0 } else { 0.0 }), zero)),
        Value::Tensor(t) if is_scalar_tensor(t) => Ok((
            t.numeric_value_at(0).expect("validated scalar storage"),
            zero,
        )),
        Value::ComplexTensor(t) if t.len() == 1 => Ok(t
            .numeric_value_at(0)
            .expect("validated scalar complex storage")),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok((
            NumericScalar::F64(if array.data[0] != 0 { 1.0 } else { 0.0 }),
            zero,
        )),
        _ => Err(mex("NumericRequired", "RHS must be numeric")),
    }
}

pub async fn assign_tensor_scalar(
    t: Tensor,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    assign_tensor_scalar_in_context(context.as_ref(), t, indices, rhs, delete).await
}

pub async fn assign_tensor_scalar_in_context(
    context: Option<&crate::context::RuntimeContext>,
    t: Tensor,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    if indices.len() == 1 {
        let total = t.len();
        let idx = indices[0];
        if idx == 0 {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        if delete {
            if idx > total {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            if !is_empty_tensor(rhs) {
                return Err(mex(
                    "DeletionRequiresEmptyRhs",
                    "Indexed deletion requires empty RHS",
                ));
            }
            return delete_tensor_linear(context, t, idx);
        }
        if matches!(rhs, Value::Complex(_, _) | Value::ComplexTensor(_)) {
            return dense_scalar::assign_complex(
                context,
                dense_scalar::ComplexSource::Real(t),
                indices,
                rhs,
                false,
            )
            .await;
        }
        let (rows, cols) = dense_linear_dimensions(t.shape());
        let shape = if idx > total {
            if !(rows == 1 || cols == 1) {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            if rows == 1 {
                ScalarShape::Growth([1, idx])
            } else {
                ScalarShape::Growth([idx, 1])
            }
        } else {
            ScalarShape::Retained(t.shared_shape())
        };
        dense_scalar::assign_real(context, t, idx - 1, shape, rhs).await
    } else if indices.len() == 2 {
        let i = indices[0];
        let mut j = indices[1];
        let (rows, cols) = dense_linear_dimensions(t.shape());
        if j == 0 {
            j = 1;
        }
        if j > cols {
            j = cols;
        }
        if i == 0 || i > rows || j == 0 {
            return Err(mex("SubscriptOutOfBounds", "Subscript out of bounds"));
        }
        if delete {
            return Err(mex(
                "UnsupportedDeletion",
                "Indexed deletion is only supported for linear vector indices",
            ));
        }
        if matches!(rhs, Value::Complex(_, _) | Value::ComplexTensor(_)) {
            return dense_scalar::assign_complex(
                context,
                dense_scalar::ComplexSource::Real(t),
                indices,
                rhs,
                false,
            )
            .await;
        }
        let idx = (j - 1)
            .checked_mul(rows)
            .and_then(|column| column.checked_add(i - 1))
            .ok_or_else(|| mex("SubscriptOutOfBounds", "Subscript out of bounds"))?;
        let shape = ScalarShape::Retained(t.shared_shape());
        dense_scalar::assign_real(context, t, idx, shape, rhs).await
    } else {
        Err(mex(
            "UnsupportedAssignmentRank",
            "Only 1D/2D scalar assignment supported",
        ))
    }
}

/// Assigns one sparse matrix element, growing scalar-indexed vectors and
/// matrices without densifying its CSC representation. Selector assignment
/// deliberately remains in the slice-assignment implementation.
pub async fn assign_sparse_scalar(
    source: SparseTensor,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    assign_sparse_scalar_in_context(context.as_ref(), source, indices, rhs, delete).await
}
pub async fn assign_sparse_scalar_in_context(
    context: Option<&crate::context::RuntimeContext>,
    source: SparseTensor,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    sparse_scalar::assign(context, source, indices, rhs, delete).await
}

pub async fn assign_complex_scalar(
    t: ComplexTensor,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    assign_complex_scalar_in_context(context.as_ref(), t, indices, rhs, delete).await
}

pub async fn assign_complex_scalar_in_context(
    context: Option<&crate::context::RuntimeContext>,
    t: ComplexTensor,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    dense_scalar::assign_complex(
        context,
        dense_scalar::ComplexSource::Complex(t),
        indices,
        rhs,
        delete,
    )
    .await
}

pub async fn assign_gpu_scalar(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    assign_gpu_scalar_in_context(context.as_ref(), handle, indices, rhs, delete).await
}

pub async fn assign_gpu_scalar_in_context(
    context: Option<&crate::context::RuntimeContext>,
    handle: &runmat_accelerate_api::GpuTensorHandle,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    let owner = super::plan::admission::Owner::new(context)?;
    let host = owner
        .await_stage(super::resident::gather_numeric_in_context(context, handle))
        .await??;
    let updated = match host {
        Value::Tensor(tensor) => {
            owner
                .await_stage(assign_tensor_scalar_in_context(
                    context, tensor, indices, rhs, delete,
                ))
                .await??
        }
        Value::ComplexTensor(tensor) => {
            owner
                .await_stage(assign_complex_scalar_in_context(
                    context, tensor, indices, rhs, delete,
                ))
                .await??
        }
        _ => {
            return Err(mex(
                "AccelerationOperationFailed",
                "Invalid resident assignment carrier",
            ))
        }
    };
    let protected = match rhs {
        Value::GpuTensor(rhs) => Some(rhs),
        _ => None,
    };
    owner.check()?;
    super::resident::restore_in_context(context, handle, updated, protected.as_slice())
}

#[cfg(test)]
mod tests {
    use super::{
        assign_complex_scalar, assign_sparse_scalar, assign_tensor_scalar,
        map_assignment_shape_error,
    };
    use futures::executor::block_on;
    use runmat_value::{
        ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, NumericDType,
        NumericStorage, Tensor, Value,
    };

    fn scalar_rhs_selected_context_preserves_native_projection_and_error_order() {
        use crate::context::RuntimeContext;
        use runmat_accelerate_api::{
            test_support::TestAllocationOwner, GpuTensorStorage, HostNumericDataOwned,
            HostNumericTensorOwned,
        };
        use std::rc::Rc;
        let make_context =
            || RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
        let ambient = make_context();
        let _ambient = ambient.enter();
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let selected = make_context().with_cancellation_token(cancellation.token());
        assert!(cancellation.cancel(runmat_execution::CancellationReason::User));
        assert!(selected.is_cancelled());
        let owner = TestAllocationOwner::new();
        let gpu = |data, shape| {
            Value::GpuTensor(
                owner
                    .allocate_numeric(HostNumericTensorOwned {
                        data,
                        shape,
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap()
                    .handle,
            )
        };
        let scalar = gpu(HostNumericDataOwned::F64(vec![1.]), vec![1, 1]);
        for error in [
            block_on(super::rhs_to_real_numeric_scalar(Some(&selected), &scalar)).unwrap_err(),
            block_on(super::rhs_to_complex_numeric_scalar(
                Some(&selected),
                &scalar,
            ))
            .unwrap_err(),
        ] {
            assert!(error
                .identifier()
                .is_some_and(|id| id.ends_with(":Cancelled")));
        }
        // Floating GPU rank validation precedes download/control; exact
        // integer assignment retains its original download-before-width gate.
        let many = gpu(HostNumericDataOwned::F64(vec![1., 2.]), vec![1, 2]);
        assert_eq!(
            block_on(super::rhs_to_real_numeric_scalar(Some(&selected), &many))
                .unwrap_err()
                .identifier(),
            Some("RunMat:ScalarRequired")
        );
        assert_eq!(
            block_on(super::rhs_to_complex_numeric_scalar(Some(&selected), &many))
                .unwrap_err()
                .identifier(),
            Some("RunMat:ScalarRequired")
        );
        let integers = gpu(
            HostNumericDataOwned::U64(vec![u64::MAX, 9_007_199_254_740_993]),
            vec![1, 2],
        );
        let error = block_on(super::rhs_to_integer_assignment_scalar(
            Some(&selected),
            &integers,
        ))
        .err()
        .expect("selected cancellation precedes integer width validation");
        assert!(error
            .identifier()
            .is_some_and(|id| id.ends_with(":Cancelled")));
        assert!(!ambient.is_cancelled());

        // The synchronous host projection keeps the original native scalar,
        // not a float round trip or a second recursive future allocation.
        let bits = 0x7fc1_2345;
        let single =
            Value::Tensor(Tensor::from_f32(vec![f32::from_bits(bits)], vec![1, 1]).unwrap());
        let runmat_value::NumericScalar::F32(value) =
            super::rhs_to_real_numeric_scalar_host(&single).unwrap()
        else {
            panic!("native single projection")
        };
        assert_eq!(value.to_bits(), bits);
        let exact = Value::Int(IntValue::U64(u64::MAX));
        assert!(matches!(
            super::rhs_to_real_numeric_scalar_host(&exact).unwrap(),
            runmat_value::NumericScalar::U64(u64::MAX)
        ));
        assert!(matches!(
            super::rhs_to_complex_numeric_scalar_host(&exact).unwrap(),
            (
                runmat_value::NumericScalar::U64(u64::MAX),
                runmat_value::NumericScalar::F64(0.)
            )
        ));
    }

    #[test]
    fn integer_linear_assignment_preserves_exact_uint64_rhs() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![1, 2]), vec![1, 2]).expect("tensor");
        let result = block_on(assign_tensor_scalar(
            tensor,
            &[2],
            &Value::Int(IntValue::U64(u64::MAX)),
            false,
        ))
        .expect("assignment");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![1, u64::MAX]))
        );
    }

    #[test]
    fn native_single_linear_growth_assignment_and_deletion_preserve_class() {
        let tensor = Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).expect("single tensor");
        let Value::Tensor(grown) = block_on(assign_tensor_scalar(
            tensor,
            &[4],
            &Value::Num(1.234_567_890_123),
            false,
        ))
        .expect("single growth assignment") else {
            panic!("expected tensor");
        };
        assert_eq!(grown.numeric_dtype(), NumericDType::F32);
        assert_eq!(grown.shape(), [1, 4]);
        assert_eq!(
            grown.clone().into_numeric_storage(),
            Ok(NumericStorage::F32(vec![
                1.0,
                2.0,
                0.0,
                1.234_567_890_123_f64 as f32,
            ]))
        );

        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty"));
        let Value::Tensor(deleted) =
            block_on(assign_tensor_scalar(grown, &[2], &empty, true)).expect("single deletion")
        else {
            panic!("expected tensor");
        };
        assert_eq!(deleted.numeric_dtype(), NumericDType::F32);
        assert_eq!(deleted.shape(), [1, 3]);
        assert_eq!(
            deleted.into_numeric_storage(),
            Ok(NumericStorage::F32(vec![
                1.0,
                0.0,
                1.234_567_890_123_f64 as f32,
            ]))
        );
    }

    #[test]
    fn native_complex_single_linear_growth_assignment_and_deletion_preserve_class() {
        let tensor = ComplexTensor::from_f32(vec![(1.0, -1.0), (2.0, -2.0)], vec![1, 2]).unwrap();
        let Value::ComplexTensor(grown) = block_on(assign_complex_scalar(
            tensor,
            &[4],
            &Value::Complex(1.234_567_890_123, -9.876_543_210_987),
            false,
        ))
        .expect("complex single growth assignment") else {
            panic!("expected complex tensor");
        };
        assert_eq!(grown.numeric_dtype(), NumericDType::F32);
        assert_eq!(grown.shape(), [1, 4]);
        assert_eq!(
            grown.as_f32_slice().map(|values| values
                .iter()
                .copied()
                .map(<(f32, f32)>::from)
                .collect::<Vec<_>>()),
            Some(vec![
                (1.0, -1.0),
                (2.0, -2.0),
                (0.0, 0.0),
                (1.234_567_890_123_f64 as f32, -9.876_543_210_987_f64 as f32,),
            ])
        );

        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty"));
        let Value::ComplexTensor(deleted) =
            block_on(assign_complex_scalar(grown, &[2], &empty, true))
                .expect("complex single deletion")
        else {
            panic!("expected complex tensor");
        };
        assert_eq!(deleted.numeric_dtype(), NumericDType::F32);
        assert_eq!(deleted.shape(), [1, 3]);
        assert_eq!(
            deleted.as_f32_slice().map(|values| values
                .iter()
                .copied()
                .map(<(f32, f32)>::from)
                .collect::<Vec<_>>()),
            Some(vec![
                (1.0, -1.0),
                (0.0, 0.0),
                (1.234_567_890_123_f64 as f32, -9.876_543_210_987_f64 as f32,),
            ])
        );
    }

    #[test]
    fn integer_scalar_complex_assignment_promotes_without_losing_wide_components() {
        let tensor = Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![1, 2])
            .expect("tensor");
        let rhs = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::I64(vec![i64::MAX]),
                    IntegerStorage::I64(vec![i64::MIN]),
                )
                .expect("integer complex storage"),
                vec![1, 1],
            )
            .expect("rhs"),
        );

        let result = block_on(assign_tensor_scalar(tensor, &[2], &rhs, false))
            .expect("typed complex integer assignment");

        let Value::ComplexTensor(output) = result else {
            panic!("integer tensor should promote to typed complex integer tensor");
        };
        assert_eq!(
            output
                .integer_storage()
                .as_ref()
                .map(|storage| (storage.real(), storage.imag())),
            Some((
                &IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                &IntegerStorage::I64(vec![0, i64::MIN]),
            ))
        );
    }

    #[test]
    fn linear_assignment_reads_typed_integer_rhs_without_mirror() {
        let tensor = Tensor::new(vec![0.0, 0.0], vec![1, 2]).expect("tensor");
        let rhs = Tensor::new_integer(IntegerStorage::U16(vec![11]), vec![1, 1]).expect("rhs");

        let result = block_on(assign_tensor_scalar(
            tensor,
            &[2],
            &Value::Tensor(rhs),
            false,
        ))
        .expect("assignment");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.materialize_f64(), vec![0.0, 11.0]);
    }

    #[test]
    fn integer_linear_assignment_reads_typed_integer_rhs_without_mirror() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![1, 2]).expect("tensor");
        let rhs = Tensor::new_integer(IntegerStorage::I16(vec![7]), vec![1, 1]).expect("rhs");

        let result = block_on(assign_tensor_scalar(
            tensor,
            &[2],
            &Value::Tensor(rhs),
            false,
        ))
        .expect("assignment");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I16(vec![1, 7]))
        );
    }

    #[test]
    fn integer_subscript_assignment_rounds_and_saturates() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I8(vec![0, 0, 0, 0]), vec![2, 2]).expect("tensor");
        let rounded = block_on(assign_tensor_scalar(
            tensor,
            &[2, 1],
            &Value::Num(3.5),
            false,
        ))
        .expect("assignment");
        let Value::Tensor(rounded) = rounded else {
            panic!("expected tensor");
        };
        let saturated = block_on(assign_tensor_scalar(
            rounded,
            &[1, 2],
            &Value::Int(IntValue::U64(u64::MAX)),
            false,
        ))
        .expect("assignment");

        let Value::Tensor(output) = saturated else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I8(vec![0, 4, i8::MAX, 0]))
        );
    }

    #[test]
    fn integer_vector_growth_and_deletion_preserve_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![7]), vec![1, 1]).expect("tensor");
        let grown = block_on(assign_tensor_scalar(
            tensor,
            &[3],
            &Value::Int(IntValue::I16(9)),
            false,
        ))
        .expect("growth");
        let Value::Tensor(grown) = grown else {
            panic!("expected tensor");
        };
        assert_eq!(
            grown.integer_storage(),
            Some(&IntegerStorage::I16(vec![7, 0, 9]))
        );

        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty"));
        let deleted = block_on(assign_tensor_scalar(grown, &[2], &empty, true)).expect("delete");
        let Value::Tensor(output) = deleted else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I16(vec![7, 9]))
        );
        assert_eq!(output.shape(), [1, 2]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn dense_rank_zero_and_rank_one_linear_assignment_keep_cached_geometry() {
        let rank_zero = Tensor::new(vec![7.0], vec![]).expect("rank-zero tensor");
        let growth = block_on(assign_tensor_scalar(
            rank_zero.clone(),
            &[2],
            &Value::Num(9.0),
            false,
        ))
        .expect_err("rank-zero tensor is not a cached-geometry vector");
        assert_eq!(growth.identifier(), Some("RunMat:IndexOutOfBounds"));

        let subscript = block_on(assign_tensor_scalar(
            rank_zero.clone(),
            &[1, 1],
            &Value::Num(9.0),
            false,
        ))
        .expect_err("rank-zero cached geometry has no rows");
        assert_eq!(subscript.identifier(), Some("RunMat:SubscriptOutOfBounds"));

        let deletion = block_on(assign_tensor_scalar(
            rank_zero.clone(),
            &[1],
            &Value::Tensor(Tensor::zeros(vec![0, 0])),
            true,
        ))
        .expect_err("rank-zero cached geometry has no deletable linear index");
        assert_eq!(deletion.identifier(), Some("RunMat:IndexOutOfBounds"));

        let rank_one = Tensor::new(vec![1.0, 2.0], vec![2]).expect("rank-one tensor");
        let Value::Tensor(grown) = block_on(assign_tensor_scalar(
            rank_one,
            &[3],
            &Value::Num(9.0),
            false,
        ))
        .expect("rank-one linear growth") else {
            panic!("expected tensor");
        };
        assert_eq!(grown.shape(), [1, 3]);
        assert_eq!(grown.materialize_f64(), vec![1.0, 2.0, 9.0]);

        let Value::Tensor(deleted) = block_on(assign_tensor_scalar(
            grown,
            &[2],
            &Value::Tensor(rank_zero),
            true,
        ))
        .expect("rank-zero RHS keeps legacy cached-empty deletion admission") else {
            panic!("expected tensor");
        };
        assert_eq!(deleted.shape(), [1, 2]);
        assert_eq!(deleted.materialize_f64(), vec![1.0, 9.0]);
    }

    #[test]
    fn integer_complex_vector_growth_uses_checked_owner_shape() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![u64::MAX]),
            IntegerStorage::U64(vec![1]),
        )
        .expect("paired storage");
        let tensor = ComplexTensor::new_integer(storage, vec![1, 1]).expect("complex tensor");
        let rhs = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![9_007_199_254_740_993]),
                    IntegerStorage::U64(vec![u64::MAX]),
                )
                .expect("rhs storage"),
                vec![1, 1],
            )
            .expect("rhs tensor"),
        );

        let Value::ComplexTensor(grown) =
            block_on(assign_complex_scalar(tensor, &[3], &rhs, false)).expect("growth")
        else {
            panic!("expected complex tensor");
        };
        assert_eq!(grown.shape(), [1, 3]);
        let expected = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![u64::MAX, 0, 9_007_199_254_740_993]),
            IntegerStorage::U64(vec![1, 0, u64::MAX]),
        )
        .expect("expected storage");
        assert_eq!(grown.integer_storage(), Some(&expected));
    }

    #[test]
    fn sparse_integer_assignment_preserves_exact_uint64_and_elides_zero() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let sparse = runmat_value::SparseTensor::new_integer(
            2,
            2,
            vec![0, 0, 0],
            vec![],
            IntegerStorage::U64(vec![]),
        )
        .expect("sparse");
        let Value::SparseTensor(inserted) = block_on(assign_sparse_scalar(
            sparse,
            &[2, 2],
            &Value::Int(IntValue::U64(u64::MAX)),
            false,
        ))
        .expect("insert") else {
            panic!("expected sparse output");
        };
        assert_eq!(inserted.row_indices, vec![1]);
        assert_eq!(inserted.col_ptrs, vec![0, 0, 1]);
        assert_eq!(
            inserted.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX]))
        );

        let Value::SparseTensor(removed) = block_on(assign_sparse_scalar(
            inserted,
            &[4],
            &Value::Num(0.0),
            false,
        ))
        .expect("remove") else {
            panic!("expected sparse output");
        };
        assert_eq!(removed.col_ptrs, vec![0, 0, 0]);
        assert_eq!(
            removed.integer_storage(),
            Some(&IntegerStorage::U64(vec![]))
        );
    }

    #[test]
    fn sparse_single_scalar_assignment_preserves_class_and_growth() {
        let sparse = runmat_value::SparseTensor::zeros_f32(1, 1);
        let Value::SparseTensor(updated) = block_on(assign_sparse_scalar(
            sparse,
            &[2, 2],
            &Value::Num(1.0 / 3.0),
            false,
        ))
        .expect("single sparse assignment") else {
            panic!("expected sparse output");
        };
        assert_eq!(updated.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(updated.shape(), vec![2, 2]);
        assert_eq!(updated.col_ptrs, vec![0, 0, 1]);
        assert_eq!(updated.row_indices, vec![1]);
        assert_eq!(updated.as_f32_slice(), Some(&[(1.0 / 3.0) as f32][..]));
    }

    #[test]
    fn sparse_logical_scalar_assignment_preserves_class_and_growth() {
        let sparse = runmat_value::SparseTensor::zeros_logical(1, 1);
        let Value::SparseTensor(updated) = block_on(assign_sparse_scalar(
            sparse,
            &[2, 2],
            &Value::Num(-3.0),
            false,
        ))
        .expect("logical sparse assignment") else {
            panic!("expected sparse output");
        };
        assert!(updated.is_logical());
        assert_eq!(updated.shape(), vec![2, 2]);
        assert_eq!(updated.col_ptrs, vec![0, 0, 1]);
        assert_eq!(updated.row_indices, vec![1]);
    }

    #[test]
    fn sparse_complex_scalar_assignment_preserves_complex_storage_and_elides_zero() {
        let sparse = runmat_value::SparseTensor::zeros_complex(2, 2);
        let Value::SparseTensor(updated) = block_on(assign_sparse_scalar(
            sparse,
            &[2, 1],
            &Value::Complex(-0.0, 4.5),
            false,
        ))
        .expect("complex sparse assignment") else {
            panic!("expected sparse output");
        };
        assert!(updated.is_complex());
        assert_eq!(updated.complex_at(1, 0), Some((-0.0, 4.5)));

        let Value::SparseTensor(cleared) = block_on(assign_sparse_scalar(
            updated,
            &[2, 1],
            &Value::Complex(0.0, 0.0),
            false,
        ))
        .expect("zero complex sparse assignment") else {
            panic!("expected sparse output");
        };
        assert!(cleared.is_complex());
        assert_eq!(cleared.nnz(), 0);
    }

    #[test]
    fn sparse_integer_assignment_preserves_every_integer_class() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let cases = vec![
            (IntegerStorage::I8(vec![]), IntValue::I8(i8::MIN)),
            (IntegerStorage::I16(vec![]), IntValue::I16(i16::MIN)),
            (IntegerStorage::I32(vec![]), IntValue::I32(i32::MIN)),
            (IntegerStorage::I64(vec![]), IntValue::I64(i64::MIN)),
            (IntegerStorage::U8(vec![]), IntValue::U8(u8::MAX)),
            (IntegerStorage::U16(vec![]), IntValue::U16(u16::MAX)),
            (IntegerStorage::U32(vec![]), IntValue::U32(u32::MAX)),
            (IntegerStorage::U64(vec![]), IntValue::U64(u64::MAX)),
        ];

        for (storage, value) in cases {
            let sparse = runmat_value::SparseTensor::new_integer(1, 1, vec![0, 0], vec![], storage)
                .expect("sparse");
            let Value::SparseTensor(updated) = block_on(assign_sparse_scalar(
                sparse,
                &[1, 1],
                &Value::Int(value.clone()),
                false,
            ))
            .expect("assignment") else {
                panic!("expected sparse output");
            };
            assert_eq!(updated.integer_at(0, 0), Some(value));
        }
    }

    #[test]
    fn sparse_integer_assignment_rounds_and_saturates_numeric_rhs() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let sparse = runmat_value::SparseTensor::new_integer(
            1,
            2,
            vec![0, 0, 0],
            vec![],
            IntegerStorage::I8(vec![]),
        )
        .expect("sparse");
        let Value::SparseTensor(rounded) = block_on(assign_sparse_scalar(
            sparse,
            &[1, 1],
            &Value::Num(3.5),
            false,
        ))
        .expect("round") else {
            panic!("expected sparse output");
        };
        let Value::SparseTensor(saturated) = block_on(assign_sparse_scalar(
            rounded,
            &[1, 2],
            &Value::Int(IntValue::U64(u64::MAX)),
            false,
        ))
        .expect("saturate") else {
            panic!("expected sparse output");
        };
        assert_eq!(
            saturated.integer_storage(),
            Some(&IntegerStorage::I8(vec![4, i8::MAX]))
        );
    }

    #[test]
    fn sparse_scalar_assignment_reports_stable_negative_path_errors() {
        let sparse = runmat_value::SparseTensor::zeros(1, 1);
        let out_of_bounds = block_on(assign_sparse_scalar(
            sparse.clone(),
            &[0],
            &Value::Num(1.0),
            false,
        ))
        .expect_err("out-of-bounds assignment must fail");
        assert_eq!(out_of_bounds.identifier(), Some("RunMat:IndexOutOfBounds"));

        let subscript_out_of_bounds = block_on(assign_sparse_scalar(
            sparse.clone(),
            &[1, 0],
            &Value::Num(1.0),
            false,
        ))
        .expect_err("invalid sparse subscript must fail");
        assert_eq!(
            subscript_out_of_bounds.identifier(),
            Some("RunMat:SubscriptOutOfBounds")
        );

        let deletion = block_on(assign_sparse_scalar(
            runmat_value::SparseTensor::zeros(2, 2),
            &[1],
            &Value::Tensor(Tensor::zeros(vec![0, 0])),
            true,
        ))
        .expect_err("matrix linear deletion must remain unsupported");
        assert_eq!(deletion.identifier(), Some("RunMat:UnsupportedDeletion"));

        let nonempty_deletion = block_on(assign_sparse_scalar(
            sparse.clone(),
            &[1],
            &Value::Num(1.0),
            true,
        ))
        .expect_err("deletion with a nonempty RHS must fail");
        assert_eq!(
            nonempty_deletion.identifier(),
            Some("RunMat:DeletionRequiresEmptyRhs")
        );

        let rank = block_on(assign_sparse_scalar(
            sparse,
            &[1, 1, 1],
            &Value::Num(1.0),
            false,
        ))
        .expect_err("rank-three scalar assignment must fail");
        assert_eq!(rank.identifier(), Some("RunMat:UnsupportedAssignmentRank"));
    }

    #[test]
    fn matlab_mode_rejects_sparse_integer_indexed_assignment() {
        let sparse = runmat_value::SparseTensor::new_integer(
            1,
            1,
            vec![0, 1],
            vec![0],
            IntegerStorage::U8(vec![7]),
        )
        .expect("sparse");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(assign_sparse_scalar(
            sparse,
            &[1],
            &Value::Int(IntValue::U8(9)),
            false,
        ))
        .expect_err("MATLAB mode must reject sparse integer assignment");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:SparseIntegerExtension")
        );
    }

    #[test]
    fn assignment_shape_error_mapping_reports_identifier() {
        let err = map_assignment_shape_error("invalid shape");
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn dense_constructor_rejects_invalid_shape_and_assignment_keeps_rhs_error() {
        let invalid = Tensor::new(vec![1.0, 2.0], vec![3])
            .expect_err("the value constructor must reject mismatched shape");
        assert_eq!(
            map_assignment_shape_error(invalid).identifier(),
            Some("RunMat:ShapeMismatch")
        );

        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
        let error = block_on(assign_tensor_scalar(
            tensor.clone(),
            &[1],
            &Value::String("not numeric".into()),
            false,
        ))
        .expect_err("valid value must report the RHS error");
        assert_eq!(error.identifier(), Some("RunMat:NumericRequired"));
        assert_eq!(tensor.shape(), [1, 2]);
        assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0]);
    }

    #[test]
    fn complex_constructors_reject_invalid_shapes_and_assignment_keeps_rhs_error() {
        let invalid_floating = ComplexTensor::new(vec![(1.0, -1.0)], vec![2])
            .expect_err("the complex value constructor must reject mismatched shape");
        assert_eq!(
            map_assignment_shape_error(invalid_floating).identifier(),
            Some("RunMat:ShapeMismatch")
        );
        let floating = ComplexTensor::new(vec![(1.0, -1.0)], vec![1, 1]).expect("complex tensor");
        let rhs_error = block_on(assign_complex_scalar(
            floating.clone(),
            &[1],
            &Value::String("not numeric".into()),
            false,
        ))
        .expect_err("rhs conversion remains first");
        assert_eq!(rhs_error.identifier(), Some("RunMat:NumericRequired"));
        assert_eq!(floating.shape(), [1, 1]);
        assert_eq!(floating.materialize_f64(), vec![(1.0, -1.0)]);

        let storage =
            IntegerComplexStorage::new(IntegerStorage::U64(vec![1]), IntegerStorage::U64(vec![2]))
                .expect("paired storage");
        let invalid_exact = ComplexTensor::new_integer(storage.clone(), vec![2])
            .expect_err("the exact complex value constructor must reject mismatched shape");
        assert_eq!(
            map_assignment_shape_error(invalid_exact).identifier(),
            Some("RunMat:ShapeMismatch")
        );
        let exact = ComplexTensor::new_integer(storage.clone(), vec![1, 1])
            .expect("integer complex tensor");
        let exact_error = block_on(assign_complex_scalar(
            exact.clone(),
            &[1],
            &Value::String("not numeric".into()),
            false,
        ))
        .expect_err("valid exact value must report the RHS error");
        assert_eq!(exact_error.identifier(), Some("RunMat:NumericRequired"));
        assert_eq!(exact.shape(), [1, 1]);
        assert_eq!(exact.integer_storage(), Some(&storage));

        // Keep the original class-specific RHS-vs-growth failure order,
        // including direct real-to-complex promotion without a mirror.
        let nonscalar =
            Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 0.0); 2], vec![1, 2]).unwrap());
        let floating = ComplexTensor::new(vec![(0.0, 0.0); 4], vec![2, 2]).unwrap();
        let error = block_on(assign_complex_scalar(floating, &[5], &nonscalar, false)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:NumericRequired"));
        let exact = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![0; 4]),
                IntegerStorage::U64(vec![0; 4]),
            )
            .unwrap(),
            vec![2, 2],
        )
        .unwrap();
        let error = block_on(assign_complex_scalar(exact, &[5], &nonscalar, false)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:IndexOutOfBounds"));
        let floating = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let error = block_on(assign_tensor_scalar(floating, &[5], &nonscalar, false)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:NumericRequired"));
        let exact = Tensor::new_integer(IntegerStorage::U64(vec![0; 4]), vec![2, 2]).unwrap();
        let error = block_on(assign_tensor_scalar(exact, &[5], &nonscalar, false)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:IndexOutOfBounds"));
    }

    #[test]
    fn assignment_acceleration_error_mapping_reports_identifier() {
        let err = crate::indexing::resident::failure(
            "gather rhs",
            anyhow::anyhow!("provider failed").into(),
        );
        assert_eq!(err.identifier(), Some("RunMat:AccelerationOperationFailed"));
        scalar_rhs_selected_context_preserves_native_projection_and_error_order();
    }
}
