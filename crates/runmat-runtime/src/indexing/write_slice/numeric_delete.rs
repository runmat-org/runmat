//! Native survivor filling with the shared deletion geometry and final owners.
use super::{
    canonicalize_delete_positions_desc, deleted_vector_dimensions, dense_delete_dimensions,
    is_empty_delete_rhs, validated_delete_position, IndexPlan,
};
use crate::context::RuntimeContext;
use crate::indexing::plan::admission::{self, Owner, Scratch};
use crate::indexing::read_slice::payload;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_value::{ComplexTensor, Tensor, Value};

enum Positions {
    Single(usize),
    Many(Scratch<usize>),
}
impl Positions {
    fn as_slice(&self) -> &[usize] {
        match self {
            Self::Single(index) => std::slice::from_ref(index),
            Self::Many(values) => values.as_slice(),
        }
    }
}

struct Deletion {
    positions: Positions,
    unique: usize,
    shape: [usize; 2],
    len: usize,
}

impl Deletion {
    fn prepare(
        context: Option<&RuntimeContext>,
        owner: &Owner<'_>,
        dimensions: (usize, usize),
        total: usize,
        plan: &IndexPlan,
        rhs: &Value,
    ) -> Result<Option<Self>, RuntimeError> {
        let (rows, cols) = dimensions;
        owner.check()?;
        if !is_empty_delete_rhs(rhs) {
            return Err(semantic_error(
                "DeletionRequiresEmptyRhs",
                "Indexed deletion requires empty RHS",
            ));
        }
        if plan.indices.is_empty() {
            return Ok(None);
        }
        if !(rows == 1 || cols == 1) {
            return Err(semantic_error(
                "UnsupportedDeletion",
                "Linear deletion is only supported for vectors",
            ));
        }
        if plan.indices.len() == 1 {
            let index = validated_delete_position(plan.indices[0], total)?;
            return Self::single(owner, dimensions, total, index).map(Some);
        }
        let mut positions = Scratch::new(context, plan.indices.len())?;
        for (&index, position) in plan.indices.iter().zip(positions.as_mut_slice()) {
            owner.check()?;
            *position = validated_delete_position(index, total)?;
        }
        owner.check()?;
        // Retain the existing indivisible sort; no per-comparison cancellation claim.
        let unique = canonicalize_delete_positions_desc(positions.as_mut_slice());
        owner.check()?;
        let len = total
            .checked_sub(unique)
            .ok_or_else(|| semantic_error("IndexOutOfBounds", "Index out of bounds"))?;
        Ok(Some(Self {
            positions: Positions::Many(positions),
            unique,
            shape: deleted_vector_dimensions(rows, len),
            len,
        }))
    }

    fn single(
        owner: &Owner<'_>,
        dimensions: (usize, usize),
        total: usize,
        index: usize,
    ) -> Result<Self, RuntimeError> {
        owner.check()?;
        let (rows, cols) = dimensions;
        if !(rows == 1 || cols == 1) {
            return Err(semantic_error(
                "UnsupportedDeletion",
                "Linear deletion is only supported for vectors",
            ));
        }
        if index >= total {
            return Err(semantic_error("IndexOutOfBounds", "Index out of bounds"));
        }
        let len = total - 1;
        Ok(Self {
            positions: Positions::Single(index),
            unique: 1,
            shape: deleted_vector_dimensions(rows, len),
            len,
        })
    }

    fn survivors<'a>(
        &'a self,
        owner: &'a Owner<'a>,
    ) -> impl FnMut(usize) -> Result<usize, RuntimeError> + 'a {
        let mut removed = self.positions.as_slice()[..self.unique]
            .iter()
            .rev()
            .copied()
            .peekable();
        let mut source = 0;
        move |_| {
            owner.check()?;
            while removed.peek() == Some(&source) {
                removed.next();
                source += 1;
                owner.check()?;
            }
            let index = source;
            source += 1;
            Ok(index)
        }
    }
}

pub(in crate::indexing) fn delete_tensor_at_index_in_context(
    context: Option<&RuntimeContext>,
    value: Tensor,
    index: usize,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    let deletion = Deletion::single(
        &owner,
        dense_delete_dimensions(value.shape()),
        value.len(),
        index,
    )?;
    let shape = admission::shape(context, 2, |dimension| Ok(deletion.shape[dimension]))?;
    let output = payload::real_tensor(
        &value,
        deletion.len,
        deletion.survivors(&owner),
        shape,
        context,
    )?;
    owner.check()?;
    Ok(Value::Tensor(output))
}

pub(in crate::indexing) fn delete_complex_at_index_in_context(
    context: Option<&RuntimeContext>,
    value: ComplexTensor,
    index: usize,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    let deletion = Deletion::single(&owner, (value.rows(), value.cols()), value.len(), index)?;
    let output = payload::complex_tensor(
        &value,
        deletion.len,
        deletion.survivors(&owner),
        &deletion.shape,
        context,
    )?;
    owner.check()?;
    Ok(Value::ComplexTensor(output))
}

pub fn delete_tensor_with_plan_in_context(
    context: Option<&RuntimeContext>,
    value: Tensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    let (rows, cols) = dense_delete_dimensions(value.shape());
    let Some(deletion) = Deletion::prepare(context, &owner, (rows, cols), value.len(), plan, rhs)?
    else {
        owner.check()?;
        return Ok(Value::Tensor(value));
    };
    let shape = admission::shape(context, 2, |dimension| Ok(deletion.shape[dimension]))?;
    let output = payload::real_tensor(
        &value,
        deletion.len,
        deletion.survivors(&owner),
        shape,
        context,
    )?;
    owner.check()?;
    Ok(Value::Tensor(output))
}

pub fn delete_complex_with_plan_in_context(
    context: Option<&RuntimeContext>,
    value: ComplexTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    // Preserve ComplexTensor's original cached/raw rank-zero/rank-one policy.
    let Some(deletion) = Deletion::prepare(
        context,
        &owner,
        (value.rows(), value.cols()),
        value.len(),
        plan,
        rhs,
    )?
    else {
        owner.check()?;
        return Ok(Value::ComplexTensor(value));
    };
    let output = payload::complex_tensor(
        &value,
        deletion.len,
        deletion.survivors(&owner),
        &deletion.shape,
        context,
    )?;
    owner.check()?;
    Ok(Value::ComplexTensor(output))
}
