use super::{
    primitive_output,
    scalar::{checked_shape_len, index_out_of_bounds, shape_error},
};
use crate::{
    context::RuntimeContext,
    indexing::plan::{admission::Owner, IndexPlan},
    runtime_error::semantic_error,
    RuntimeError,
};
use runmat_value::{LogicalArray, Value};

pub(super) fn read(
    context: Option<&RuntimeContext>,
    source: &[u8],
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    for (ordinal, &index) in plan.indices.iter().enumerate() {
        if ordinal % 4096 == 0 {
            owner.check()?;
        }
        source.get(index as usize).ok_or_else(index_out_of_bounds)?;
    }
    if let [index] = plan.indices.as_slice() {
        owner.check()?;
        return Ok(Value::Bool(source[*index as usize] != 0));
    }
    let expected = plan
        .output_shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| {
            shape_error(format!(
                "LogicalArray shape {:?} overflows usize",
                plan.output_shape
            ))
        })?;
    if expected != plan.indices.len() {
        return Err(shape_error(format!(
            "LogicalArray data length {} doesn't match shape {:?} ({} elements)",
            plan.indices.len(),
            plan.output_shape,
            expected
        )));
    }
    primitive_output::logical(context, &plan.output_shape, |output| {
        for (ordinal, &index) in plan.indices.iter().enumerate() {
            if ordinal % 4096 == 0 {
                owner.check()?;
            }
            output[ordinal] = u8::from(source[index as usize] != 0);
        }
        Ok(())
    })
    .map(Value::LogicalArray)
}

enum Rhs<'a> {
    Scalar(u8),
    Array(&'a [u8]),
}
impl Rhs<'_> {
    fn at(&self, index: usize) -> u8 {
        match self {
            Self::Scalar(v) => *v,
            Self::Array(v) => u8::from(v[index] != 0),
        }
    }
}
fn rhs<'a>(value: &'a Value, count: usize) -> Result<Rhs<'a>, RuntimeError> {
    let scalar = match value {
        Value::Bool(v) => u8::from(*v),
        Value::Num(v) => u8::from(*v != 0.),
        Value::Int(v) => u8::from(v.to_f64() != 0.),
        Value::LogicalArray(v) if v.len() == count => return Ok(Rhs::Array(&v.data)),
        Value::Tensor(v) if v.len() == 1 => u8::from(
            v.numeric_value_at(0)
                .ok_or_else(|| {
                    semantic_error(
                        "AssignmentTypeMismatch",
                        "logical assignment requires a numeric or logical value",
                    )
                })?
                .materialize_f64()
                != 0.,
        ),
        _ => return Err(semantic_error(
            "AssignmentTypeMismatch",
            "logical assignment requires a numeric or logical scalar, or a matching logical array",
        )),
    };
    Ok(Rhs::Scalar(scalar))
}

pub(super) fn assign(
    context: Option<&RuntimeContext>,
    mut value: LogicalArray,
    plan: &IndexPlan,
    replacement: &Value,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    let target = checked_shape_len(&plan.base_shape)?;
    let replacement = rhs(replacement, plan.indices.len())?;
    let count = value.len().max(target);
    validate(&owner, plan, count)?;
    if target <= value.len() {
        if let Some(bytes) = value.data.try_as_mut_slice_unique() {
            scatter(&owner, bytes, plan, &replacement)?;
            owner.check()?;
            return Ok(Value::LogicalArray(value));
        }
    }
    let shape = if target > value.len() {
        &plan.base_shape
    } else {
        &value.shape
    };
    let output = primitive_output::logical(context, shape, |bytes| {
        for (offset, chunk) in value.data.chunks(4096).enumerate() {
            owner.check()?;
            let start = offset * 4096;
            for (target, &source) in bytes[start..start + chunk.len()].iter_mut().zip(chunk) {
                *target = u8::from(source != 0);
            }
        }
        scatter(&owner, bytes, plan, &replacement)
    })?;
    Ok(Value::LogicalArray(output))
}

pub(super) fn assign_bool(
    context: Option<&RuntimeContext>,
    value: bool,
    plan: &IndexPlan,
    replacement: &Value,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    let target = checked_shape_len(&plan.base_shape)?;
    let replacement = rhs(replacement, plan.indices.len())?;
    validate(&owner, plan, target.max(1))?;
    let shape = if target > 1 {
        plan.base_shape.as_slice()
    } else {
        &[1, 1]
    };
    primitive_output::logical(context, shape, |bytes| {
        bytes[0] = u8::from(value);
        scatter(&owner, bytes, plan, &replacement)
    })
    .map(Value::LogicalArray)
}

fn validate(owner: &Owner<'_>, plan: &IndexPlan, count: usize) -> Result<(), RuntimeError> {
    for (ordinal, &index) in plan.indices.iter().enumerate() {
        if ordinal % 4096 == 0 {
            owner.check()?;
        }
        if index as usize >= count {
            return Err(index_out_of_bounds());
        }
    }
    Ok(())
}
fn scatter(
    owner: &Owner<'_>,
    bytes: &mut [u8],
    plan: &IndexPlan,
    replacement: &Rhs<'_>,
) -> Result<(), RuntimeError> {
    for (ordinal, &index) in plan.indices.iter().enumerate() {
        if ordinal % 4096 == 0 {
            owner.check()?;
        }
        bytes[index as usize] = replacement.at(ordinal);
    }
    Ok(())
}
