//! Dense numeric-RHS policy used by existing VM/Native logical operations.
//! Reuse native RHS interpretation, relocation and final storage; no f64
//! source/return mirror. Runtime's narrower primitive grammar stays separate.
use super::{growth::Relocation, host_rhs::HostRhs, IndexPlan};
use crate::{
    context::RuntimeContext,
    indexing::{plan::admission::Owner, read_slice::payload::output, value::primitive_output},
    runtime_error::semantic_error,
    RuntimeError,
};
use runmat_value::{LogicalArray, NumericDType, NumericScalar, Value};

pub async fn assign_logical_with_plan_in_context(
    context: Option<&RuntimeContext>,
    value: LogicalArray,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    if !matches!(rhs, Value::GpuTensor(_)) || plan.indices.is_empty() {
        return assign_logical_with_plan_host_in_context(context, value, plan, rhs);
    }
    let rhs = owner
        .await_stage(super::gather_assignment_rhs_in_context(context, rhs, false))
        .await??;
    assign_logical_with_plan_host_in_context(context, value, plan, &rhs)
}

pub fn assign_logical_with_plan_host_in_context(
    context: Option<&RuntimeContext>,
    mut value: LogicalArray,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    if plan.indices.is_empty() {
        super::empty_rhs::validate_real_or_host_complex_rhs(rhs, plan)?;
        owner.check()?;
        return Ok(Value::LogicalArray(value));
    }
    let complex = matches!(rhs, Value::Complex(_, _) | Value::ComplexTensor(_));
    let count = plan
        .base_shape
        .iter()
        .try_fold(1usize, |length, extent| length.checked_mul(*extent))
        .ok_or_else(|| {
            semantic_error(
                "ShapeOverflow",
                "slice assignment target shape overflows usize",
            )
        })?;
    if !complex && count < value.len() {
        return Err(semantic_error(
            "InvalidAssignmentPlan",
            "slice assignment plan cannot shrink numeric storage",
        ));
    }
    let view = if complex {
        HostRhs::complex(rhs, &plan.selection_lengths, false)?
    } else {
        HostRhs::for_plan(rhs, plan)?
    };
    if view.len() != plan.indices.len() {
        return Err(semantic_error(
            "ShapeMismatch",
            "shape mismatch for slice assign",
        ));
    }
    for &index in &plan.indices {
        owner.check()?;
        if index as usize >= count {
            return Err(semantic_error(
                "ShapeMismatch",
                "slice assign: index out of bounds",
            ));
        }
    }
    if complex {
        let relocation = if value.shape == plan.base_shape {
            Relocation::Linear
        } else {
            Relocation::new(context, &value.shape, value.len(), &plan.base_shape)?
        };
        return output::complex(context, NumericDType::F64, &plan.base_shape, |output| {
            for (index, &byte) in value.data.iter().enumerate() {
                owner.check()?;
                output.set(
                    relocation.position(index),
                    NumericScalar::F64(if byte != 0 { 1. } else { 0. }),
                    NumericScalar::F64(0.),
                )?;
            }
            for (ordinal, &index) in plan.indices.iter().enumerate() {
                owner.check()?;
                let (real, imag) = view.numeric_components_at(ordinal);
                output.set(index as usize, real, imag)?;
            }
            owner.check()
        })
        .map(Value::ComplexTensor);
    }
    if !plan.base_shape.is_empty() && value.shape == plan.base_shape {
        if let Some(bytes) = value.data.try_as_mut_slice_unique() {
            for (ordinal, &index) in plan.indices.iter().enumerate() {
                owner.check()?;
                bytes[index as usize] = u8::from(view.numeric_at(ordinal).materialize_f64() != 0.);
            }
            owner.check()?;
            return Ok(Value::LogicalArray(value));
        }
    }
    let relocation = Relocation::new(context, &value.shape, value.len(), &plan.base_shape)?;
    primitive_output::logical(context, &plan.base_shape, |bytes| {
        for (index, &byte) in value.data.iter().enumerate() {
            owner.check()?;
            bytes[relocation.position(index)] = u8::from(byte != 0);
        }
        for (ordinal, &index) in plan.indices.iter().enumerate() {
            owner.check()?;
            bytes[index as usize] = u8::from(view.numeric_at(ordinal).materialize_f64() != 0.);
        }
        owner.check()
    })
    .map(Value::LogicalArray)
}
