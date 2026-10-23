//! Dense assignment copies/restrides directly into its final admitted owner.
use super::{
    empty_rhs, growth::Relocation, host_rhs::HostRhs, preflight_tensor_assignment, IndexPlan,
};
use crate::context::RuntimeContext;
use crate::indexing::{integer_assignment, plan::admission::Owner, read_slice::payload::output};
use crate::RuntimeError;
use runmat_value::{ComplexTensor, NumericScalar, Tensor, Value};

fn assignment_scalar(value: integer_assignment::IntegerAssignmentValue) -> NumericScalar {
    match value {
        integer_assignment::IntegerAssignmentValue::Exact(value) => NumericScalar::from(value),
        integer_assignment::IntegerAssignmentValue::Float(value) => NumericScalar::F64(value),
    }
}

fn update_unique_real(
    tensor: &mut Tensor,
    owner: &Owner<'_>,
    view: &HostRhs<'_>,
    plan: &IndexPlan,
) -> Result<Option<()>, RuntimeError> {
    let integer = tensor.integer_storage().is_some();
    let Some(mut output) = tensor.try_unique_numeric_mutation() else {
        return Ok(None);
    };
    for (ordinal, &index) in plan.indices.iter().enumerate() {
        owner.check()?;
        let value = if integer {
            assignment_scalar(view.integer_at(ordinal))
        } else {
            view.numeric_at(ordinal)
        };
        output.set_value(index as usize, value).map_err(|error| {
            if integer {
                super::mex("Assignment", error)
            } else {
                super::map_slice_shape_error("slice assign", error)
            }
        })?;
    }
    owner.check()?;
    Ok(Some(()))
}

fn update_unique_complex(
    tensor: &mut ComplexTensor,
    owner: &Owner<'_>,
    view: &HostRhs<'_>,
    plan: &IndexPlan,
) -> Result<Option<()>, RuntimeError> {
    let integer = tensor.integer_storage().is_some();
    let Some(mut output) = tensor.try_unique_numeric_mutation() else {
        return Ok(None);
    };
    for (ordinal, &index) in plan.indices.iter().enumerate() {
        owner.check()?;
        let (real, imag) = if integer {
            let value = view.complex_integer_at(ordinal);
            (assignment_scalar(value.real), assignment_scalar(value.imag))
        } else {
            view.numeric_components_at(ordinal)
        };
        output
            .set_value(index as usize, real, imag)
            .map_err(|error| {
                if integer {
                    super::mex("Assignment", error)
                } else {
                    super::map_slice_shape_error("complex slice assign", error)
                }
            })?;
    }
    owner.check()?;
    Ok(Some(()))
}

fn checked_view<'a>(view: HostRhs<'a>, plan: &IndexPlan) -> Result<HostRhs<'a>, RuntimeError> {
    if view.len() != plan.indices.len() {
        return Err(super::mex(
            "ShapeMismatch",
            "shape mismatch for slice assign",
        ));
    }
    Ok(view)
}

fn validate_integer_destinations(owner: &Owner<'_>, plan: &IndexPlan) -> Result<(), RuntimeError> {
    let len = super::checked_assignment_element_count(&plan.base_shape)?;
    for &index in &plan.indices {
        owner.check()?;
        if index as usize >= len {
            return Err(super::mex(
                "Assignment",
                format!("integer storage index {index} is out of bounds"),
            ));
        }
    }
    Ok(())
}

pub fn assign_tensor_with_plan_host_in_context(
    context: Option<&RuntimeContext>,
    mut tensor: Tensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    preflight_tensor_assignment(&tensor, plan, rhs)?;
    if plan.indices.is_empty() {
        owner.check()?;
        return Ok(Value::Tensor(tensor));
    }
    if matches!(rhs, Value::Complex(_, _) | Value::ComplexTensor(_)) {
        let view = if tensor.integer_storage().is_some() {
            HostRhs::complex_for_plan(rhs, plan)?
        } else {
            HostRhs::complex(rhs, &plan.selection_lengths, false)?
        };
        let view = checked_view(view, plan)?;
        if tensor.integer_storage().is_some() {
            validate_integer_destinations(&owner, plan)?;
        }
        let relocation = if tensor.shape() == plan.base_shape.as_slice() {
            Relocation::Linear
        } else {
            Relocation::new(context, tensor.shape(), tensor.len(), &plan.base_shape)?
        };
        let result = output::complex(
            context,
            tensor.numeric_dtype(),
            &plan.base_shape,
            |output| {
                let zero = tensor
                    .integer_storage()
                    .map_or(NumericScalar::F64(0.0), |prototype| {
                        NumericScalar::from(prototype.cast_f64_assignment(0.0))
                    });
                for index in 0..tensor.len() {
                    owner.check()?;
                    output.set(
                        relocation.position(index),
                        tensor
                            .numeric_value_at(index)
                            .expect("validated original tensor"),
                        zero,
                    )?;
                }
                for (ordinal, &index) in plan.indices.iter().enumerate() {
                    owner.check()?;
                    let (real, imag) = if let Some(prototype) = tensor.integer_storage() {
                        let value = view.complex_integer_at(ordinal);
                        (
                            NumericScalar::from(integer_assignment::scalar_value(
                                prototype,
                                &value.real,
                            )),
                            NumericScalar::from(integer_assignment::scalar_value(
                                prototype,
                                &value.imag,
                            )),
                        )
                    } else {
                        view.numeric_components_at(ordinal)
                    };
                    output.set(index as usize, real, imag)?;
                }
                owner.check()
            },
        )?;
        owner.check()?;
        return Ok(Value::ComplexTensor(result));
    }
    let view = checked_view(HostRhs::for_plan(rhs, plan)?, plan)?;
    if tensor.integer_storage().is_some() {
        validate_integer_destinations(&owner, plan)?;
    }
    if !plan.base_shape.is_empty()
        && tensor.shape() == plan.base_shape.as_slice()
        && update_unique_real(&mut tensor, &owner, &view, plan)?.is_some()
    {
        owner.check()?;
        return Ok(Value::Tensor(tensor));
    }
    let relocation = Relocation::new(context, tensor.shape(), tensor.len(), &plan.base_shape)?;
    let result = output::real(
        context,
        tensor.numeric_dtype(),
        plan.base_shape.clone(),
        |output| {
            for index in 0..tensor.len() {
                owner.check()?;
                output.set(
                    relocation.position(index),
                    tensor
                        .numeric_value_at(index)
                        .expect("validated original tensor"),
                )?;
            }
            for (ordinal, &index) in plan.indices.iter().enumerate() {
                owner.check()?;
                let value = if let Some(prototype) = tensor.integer_storage() {
                    NumericScalar::from(integer_assignment::scalar_value(
                        prototype,
                        &view.integer_at(ordinal),
                    ))
                } else {
                    view.numeric_at(ordinal)
                };
                output.set(index as usize, value)?;
            }
            owner.check()
        },
    )?;
    owner.check()?;
    Ok(Value::Tensor(result))
}

pub fn assign_complex_with_plan_host_in_context(
    context: Option<&RuntimeContext>,
    mut tensor: ComplexTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    if plan.indices.is_empty() {
        empty_rhs::validate_complex_rhs(rhs, plan, tensor.integer_storage().is_some())?;
        owner.check()?;
        return Ok(Value::ComplexTensor(tensor));
    }
    let view = if tensor.integer_storage().is_some() {
        HostRhs::complex_for_plan(rhs, plan)?
    } else {
        HostRhs::complex(rhs, &plan.selection_lengths, false)?
    };
    let view = checked_view(view, plan)?;
    if tensor.integer_storage().is_some() {
        validate_integer_destinations(&owner, plan)?;
    }
    if tensor.shape() == plan.base_shape.as_slice()
        && update_unique_complex(&mut tensor, &owner, &view, plan)?.is_some()
    {
        owner.check()?;
        return Ok(Value::ComplexTensor(tensor));
    }
    // Match the existing complex same-shape boundary, including rank zero.
    let relocation = if tensor.shape() == plan.base_shape.as_slice() {
        Relocation::Linear
    } else {
        Relocation::new(context, tensor.shape(), tensor.len(), &plan.base_shape)?
    };
    let result = output::complex(
        context,
        tensor.numeric_dtype(),
        &plan.base_shape,
        |output| {
            for index in 0..tensor.len() {
                owner.check()?;
                let (real, imag) = tensor
                    .numeric_value_at(index)
                    .expect("validated original complex tensor");
                output.set(relocation.position(index), real, imag)?;
            }
            for (ordinal, &index) in plan.indices.iter().enumerate() {
                owner.check()?;
                let (real, imag) = if let Some(prototype) = tensor.integer_storage() {
                    let value = view.complex_integer_at(ordinal);
                    (
                        NumericScalar::from(integer_assignment::scalar_value(
                            prototype.real(),
                            &value.real,
                        )),
                        NumericScalar::from(integer_assignment::scalar_value(
                            prototype.imag(),
                            &value.imag,
                        )),
                    )
                } else {
                    view.numeric_components_at(ordinal)
                };
                output.set(index as usize, real, imag)?;
            }
            owner.check()
        },
    )?;
    owner.check()?;
    Ok(Value::ComplexTensor(result))
}
