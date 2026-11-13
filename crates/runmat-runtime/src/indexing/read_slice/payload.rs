//! Borrowed numeric selection into the canonical final native output owner.
use super::map_slice_shape_error;
use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::indexing::plan::admission::Owner;
use crate::RuntimeError;
use runmat_value::{ComplexTensor, NumericScalar, SharedShape, Tensor, Value};

pub(in crate::indexing) mod output;

fn allocation(context: Option<&RuntimeContext>, source: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        context,
        "RunMat:Indexing:PayloadAllocation",
        "indexed numeric output",
        source,
    )
}
fn receipt(
    context: Option<&RuntimeContext>,
    raw: usize,
    claims: usize,
) -> Result<Option<HostAllocationReceipt>, RuntimeError> {
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(raw).map_err(|_| allocation(context, HostAllocationError::Overflow))?,
        claims,
    )
    .map_err(|source| allocation(context, source))?;
    HostAllocationReceipt::admit(
        context,
        HostAllocationPlan::new(0, retained).map_err(|source| allocation(context, source))?,
    )
    .map_err(|source| allocation(context, source))
}
fn validate_shape(shape: &[usize], count: usize) -> Result<(), RuntimeError> {
    if shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        != Some(count)
    {
        return Err(map_slice_shape_error(
            "selected element count does not match output shape",
        ));
    }
    Ok(())
}
fn out_of_bounds() -> RuntimeError {
    crate::runtime_error::semantic_error(
        "IndexOutOfBounds",
        "Slice error: numeric index out of bounds",
    )
}

pub(super) fn real(
    tensor: &Tensor,
    count: usize,
    index: impl Fn(usize) -> usize,
    shape: impl FnOnce() -> Result<SharedShape, RuntimeError>,
    context: Option<&RuntimeContext>,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    if count == 1 {
        let value = tensor
            .numeric_value_at(index(0))
            .ok_or_else(out_of_bounds)?;
        match value {
            NumericScalar::F64(value) => return Ok(Value::Num(value)),
            NumericScalar::F32(_) => {}
            value => return Ok(Value::Int(value.into_int_value().expect("integer class"))),
        }
    }
    let shape = if count == 1 {
        crate::indexing::plan::admission::shape(context, 2, |_| Ok(1))?
    } else {
        shape()?
    };
    let result = real_tensor(
        tensor,
        count,
        |position| Ok(index(position)),
        shape,
        context,
    )?;
    owner.check()?;
    Ok(Value::Tensor(result))
}

/// Tensor-only construction: deletion never enters read scalar presentation.
pub(in crate::indexing) fn real_tensor(
    tensor: &Tensor,
    count: usize,
    mut index: impl FnMut(usize) -> Result<usize, RuntimeError>,
    shape: SharedShape,
    context: Option<&RuntimeContext>,
) -> Result<Tensor, RuntimeError> {
    let owner = Owner::new(context)?;
    validate_shape(&shape, count)?;
    output::real(context, tensor.numeric_dtype(), shape, |output| {
        for position in 0..count {
            owner.check()?;
            output.set(
                position,
                tensor
                    .numeric_value_at(index(position)?)
                    .ok_or_else(out_of_bounds)?,
            )?;
        }
        owner.check()
    })
}

pub(super) fn complex(
    tensor: &ComplexTensor,
    indices: &[u32],
    shape: SharedShape,
    context: Option<&RuntimeContext>,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    if tensor.integer_storage().is_none() && indices.len() == 1 {
        let (real, imag) = tensor
            .numeric_value_at(indices[0] as usize)
            .ok_or_else(out_of_bounds)?;
        return Ok(Value::Complex(
            real.materialize_f64(),
            imag.materialize_f64(),
        ));
    }
    let output = complex_tensor(
        tensor,
        indices.len(),
        |position| Ok(indices[position] as usize),
        &shape,
        context,
    )?;
    owner.check()?;
    Ok(Value::ComplexTensor(output))
}

pub(in crate::indexing) fn complex_tensor(
    tensor: &ComplexTensor,
    count: usize,
    mut index: impl FnMut(usize) -> Result<usize, RuntimeError>,
    shape: &[usize],
    context: Option<&RuntimeContext>,
) -> Result<ComplexTensor, RuntimeError> {
    let owner = Owner::new(context)?;
    validate_shape(shape, count)?;
    output::complex(context, tensor.numeric_dtype(), shape, |output| {
        for position in 0..count {
            owner.check()?;
            let (real, imag) = tensor
                .numeric_value_at(index(position)?)
                .ok_or_else(out_of_bounds)?;
            output.set(position, real, imag)?;
        }
        owner.check()
    })
}
