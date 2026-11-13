//! Scalar assignment retains its own cached geometry while sharing final
//! payload ownership and unique mutation with planned dense assignment.
use super::{
    dense_linear_dimensions, map_assignment_shape_error, mex,
    rhs_to_complex_integer_assignment_scalar, rhs_to_complex_numeric_scalar,
    rhs_to_integer_assignment_scalar, rhs_to_real_numeric_scalar,
};
use crate::context::RuntimeContext;
use crate::indexing::{
    integer_assignment,
    plan::admission::{self, Owner},
    read_slice::payload::output,
};
use crate::RuntimeError;
use runmat_value::{ComplexTensor, IntegerStorage, NumericScalar, SharedShape, Tensor, Value};

pub(super) enum ScalarShape {
    Retained(SharedShape),
    Growth([usize; 2]),
}
impl ScalarShape {
    fn as_slice(&self) -> &[usize] {
        match self {
            Self::Retained(shape) => shape,
            Self::Growth(shape) => shape,
        }
    }
    fn into_shared(self, context: Option<&RuntimeContext>) -> Result<SharedShape, RuntimeError> {
        match self {
            Self::Retained(shape) => Ok(shape),
            Self::Growth(shape) => admission::shape(context, 2, |dimension| Ok(shape[dimension])),
        }
    }
}

async fn gathered_rhs(
    context: Option<&RuntimeContext>,
    owner: &Owner<'_>,
    rhs: &Value,
    integer: bool,
) -> Result<Option<Value>, RuntimeError> {
    let Value::GpuTensor(handle) = rhs else {
        return Ok(None);
    };
    let exact_integer = integer && runmat_accelerate_api::handle_integer_type(handle).is_some();
    if !exact_integer && !handle.shape().iter().all(|&extent| extent == 1) {
        return Err(mex("ScalarRequired", "RHS must be scalar"));
    }
    let host = owner
        .await_stage(
            crate::indexing::write_slice::gather_assignment_rhs_in_context(context, rhs, integer),
        )
        .await??;
    if exact_integer {
        let Value::Tensor(tensor) = &host else {
            unreachable!("integer download validates real tensor class")
        };
        if tensor.len() != 1 {
            return Err(mex(
                "InvalidAssignmentRhs",
                "integer gpuArray assignment rhs must be scalar",
            ));
        }
    }
    Ok(Some(host))
}

fn update_unique_real(
    tensor: &mut Tensor,
    index: usize,
    value: NumericScalar,
    owner: &Owner<'_>,
) -> Result<Option<()>, RuntimeError> {
    let Some(mut output) = tensor.try_unique_numeric_mutation() else {
        return Ok(None);
    };
    output
        .set_value(index, value)
        .map_err(map_assignment_shape_error)?;
    owner.check()?;
    Ok(Some(()))
}

fn update_unique_complex(
    tensor: &mut ComplexTensor,
    index: usize,
    value: (NumericScalar, NumericScalar),
    owner: &Owner<'_>,
) -> Result<Option<()>, RuntimeError> {
    let Some(mut output) = tensor.try_unique_numeric_mutation() else {
        return Ok(None);
    };
    output
        .set_value(index, value.0, value.1)
        .map_err(map_assignment_shape_error)?;
    owner.check()?;
    Ok(Some(()))
}

pub(super) async fn assign_real(
    context: Option<&RuntimeContext>,
    mut tensor: Tensor,
    index: usize,
    shape: ScalarShape,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    tensor
        .validate_shape()
        .map_err(map_assignment_shape_error)?;
    let host = gathered_rhs(context, &owner, rhs, tensor.integer_storage().is_some()).await?;
    let rhs = host.as_ref().unwrap_or(rhs);
    let value = if let Some(prototype) = tensor.integer_storage() {
        let value = owner
            .await_stage(rhs_to_integer_assignment_scalar(context, rhs))
            .await??;
        NumericScalar::from(integer_assignment::scalar_value(prototype, &value))
    } else {
        owner
            .await_stage(rhs_to_real_numeric_scalar(context, rhs))
            .await??
    };
    owner.check()?;
    if shape.as_slice() == tensor.shape()
        && update_unique_real(&mut tensor, index, value, &owner)?.is_some()
    {
        return Ok(Value::Tensor(tensor));
    }
    let shape = shape.into_shared(context)?;
    let result = output::real(context, tensor.numeric_dtype(), shape, |output| {
        // Scalar growth has always kept linear storage order, not the
        // planned multidimensional leading-dimension rebase policy.
        for source in 0..tensor.len() {
            owner.check()?;
            output.set(
                source,
                tensor
                    .numeric_value_at(source)
                    .expect("validated scalar assignment source"),
            )?;
        }
        owner.check()?;
        output.set(index, value)
    })?;
    owner.check()?;
    Ok(Value::Tensor(result))
}

pub(super) enum ComplexSource {
    Real(Tensor),
    Complex(ComplexTensor),
}
impl ComplexSource {
    fn len(&self) -> usize {
        match self {
            Self::Real(value) => value.len(),
            Self::Complex(value) => value.len(),
        }
    }
    fn dimensions(&self) -> (usize, usize) {
        match self {
            Self::Real(value) => dense_linear_dimensions(value.shape()),
            Self::Complex(value) => (value.rows(), value.cols()),
        }
    }
    fn shape(&self) -> &[usize] {
        match self {
            Self::Real(value) => value.shape(),
            Self::Complex(value) => value.shape(),
        }
    }
    fn shared_shape(&self) -> SharedShape {
        match self {
            Self::Real(value) => value.shared_shape(),
            Self::Complex(value) => value.shared_shape(),
        }
    }
    fn dtype(&self) -> runmat_value::NumericDType {
        match self {
            Self::Real(value) => value.numeric_dtype(),
            Self::Complex(value) => value.numeric_dtype(),
        }
    }
    fn validate(&self) -> Result<(), RuntimeError> {
        match self {
            Self::Real(value) => value.validate_shape(),
            Self::Complex(value) => value.validate_shape(),
        }
        .map_err(map_assignment_shape_error)
    }
    fn prototypes(&self) -> Option<(&IntegerStorage, &IntegerStorage)> {
        match self {
            Self::Real(value) => value.integer_storage().map(|storage| (storage, storage)),
            Self::Complex(value) => value
                .integer_storage()
                .map(|storage| (storage.real(), storage.imag())),
        }
    }
    fn value_at(&self, index: usize) -> (NumericScalar, NumericScalar) {
        match self {
            Self::Real(value) => {
                let zero = value
                    .integer_storage()
                    .map_or(NumericScalar::F64(0.0), |prototype| {
                        NumericScalar::from(prototype.cast_f64_assignment(0.0))
                    });
                (
                    value
                        .numeric_value_at(index)
                        .expect("validated real promotion source"),
                    zero,
                )
            }
            Self::Complex(value) => value
                .numeric_value_at(index)
                .expect("validated complex assignment source"),
        }
    }
    fn linear_shape(&self, index: usize) -> Result<ScalarShape, RuntimeError> {
        if index > self.len() {
            let (rows, cols) = self.dimensions();
            if !(rows == 1 || cols == 1) {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            Ok(ScalarShape::Growth(if rows == 1 {
                [1, index]
            } else {
                [index, 1]
            }))
        } else {
            Ok(ScalarShape::Retained(self.shared_shape()))
        }
    }
}

pub(super) async fn assign_complex(
    context: Option<&RuntimeContext>,
    mut source: ComplexSource,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    // The old real promotion validated its source before entering the
    // complex scalar driver; retain that error ordering without a mirror.
    if matches!(&source, ComplexSource::Real(_)) {
        source.validate()?;
    }
    let (index, linear) = if indices.len() == 1 {
        let index = indices[0];
        if index == 0 {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        if delete {
            if index > source.len() {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            if !super::is_empty_tensor(rhs) {
                return Err(mex(
                    "DeletionRequiresEmptyRhs",
                    "Indexed deletion requires empty RHS",
                ));
            }
            return match source {
                ComplexSource::Real(value) => super::delete_tensor_linear(context, value, index),
                ComplexSource::Complex(value) => {
                    super::delete_complex_linear(context, value, index)
                }
            };
        }
        (index - 1, true)
    } else if indices.len() == 2 {
        let (rows, cols) = source.dimensions();
        let i = indices[0];
        let mut j = indices[1];
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
        let index = (j - 1)
            .checked_mul(rows)
            .and_then(|column| column.checked_add(i - 1))
            .ok_or_else(|| mex("SubscriptOutOfBounds", "Subscript out of bounds"))?;
        (index, false)
    } else {
        return Err(mex(
            "UnsupportedAssignmentRank",
            "Only 1D/2D scalar assignment supported",
        ));
    };
    let integer = source.prototypes().is_some();
    // Integer complex geometry was checked before RHS conversion; floating
    // complex decoded RHS first. Keep both original observable orderings.
    let early_shape = if integer {
        Some(
            source.linear_shape(
                index
                    .checked_add(1)
                    .ok_or_else(|| mex("IndexOutOfBounds", "Index out of bounds"))?,
            )?,
        )
    } else {
        None
    };
    let host = gathered_rhs(context, &owner, rhs, integer).await?;
    let rhs = host.as_ref().unwrap_or(rhs);
    let (real, imag) = if let Some((real, imag)) = source.prototypes() {
        let value = owner
            .await_stage(rhs_to_complex_integer_assignment_scalar(context, rhs))
            .await??;
        source.validate()?;
        (
            NumericScalar::from(integer_assignment::scalar_value(real, &value.real)),
            NumericScalar::from(integer_assignment::scalar_value(imag, &value.imag)),
        )
    } else {
        let value = owner
            .await_stage(rhs_to_complex_numeric_scalar(context, rhs))
            .await??;
        source.validate()?;
        value
    };
    let shape = if let Some(shape) = early_shape {
        shape
    } else if linear {
        source.linear_shape(
            index
                .checked_add(1)
                .ok_or_else(|| mex("IndexOutOfBounds", "Index out of bounds"))?,
        )?
    } else {
        ScalarShape::Retained(source.shared_shape())
    };
    owner.check()?;
    if shape.as_slice() == source.shape() {
        let updated = match &mut source {
            ComplexSource::Real(_) => false,
            ComplexSource::Complex(tensor) => {
                update_unique_complex(tensor, index, (real, imag), &owner)?.is_some()
            }
        };
        if updated {
            let ComplexSource::Complex(tensor) = source else {
                unreachable!("only existing complex payload can be updated uniquely")
            };
            return Ok(Value::ComplexTensor(tensor));
        }
    }
    let result = output::complex(context, source.dtype(), shape.as_slice(), |output| {
        for index in 0..source.len() {
            owner.check()?;
            let (real, imag) = source.value_at(index);
            output.set(index, real, imag)?;
        }
        owner.check()?;
        output.set(index, real, imag)
    })?;
    owner.check()?;
    Ok(Value::ComplexTensor(result))
}
