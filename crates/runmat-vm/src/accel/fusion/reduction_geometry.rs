//! Request-local reduction geometry. Cached graph shapes are never runtime evidence.
use crate::interpreter::errors::mex;
use runmat_accelerate::fusion_exec::FusionExecutionRequest;
use runmat_accelerate::{FusionGroupPlan, ReductionAxes};
use runmat_runtime::builtins::common::tensor::scalar_integer_value;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

#[path = "reduction_geometry/data_shape.rs"]
mod data_shape;

#[derive(Debug, PartialEq, Eq)]
pub struct ReductionGeometry {
    pub axis: usize,
    pub reduce_len: usize,
    pub num_slices: usize,
    pub output_shape: Vec<usize>,
}

pub(super) fn bind_axis(
    plan: &FusionGroupPlan,
    geometry: &ReductionGeometry,
) -> Result<FusionGroupPlan, RuntimeError> {
    use runmat_accelerate::fusion::FusionOp;
    let mut highest = plan
        .inputs
        .iter()
        .chain(plan.const_values.keys())
        .copied()
        .max()
        .unwrap_or(0);
    for operation in &plan.operations {
        let (inputs, output) = match operation {
            FusionOp::Primitive { inputs, output, .. }
            | FusionOp::Builtin { inputs, output, .. } => (inputs, output),
        };
        for id in inputs.iter().chain(output.iter()) {
            highest = highest.max(*id);
        }
    }
    let fresh = highest.checked_add(1).ok_or_else(|| {
        mex(
            "FusionReductionControlOverflow",
            "fusion: no unused reduction control identity",
        )
    })?;
    let mut resolved = plan.clone();
    if matches!(plan.reduction_axes, Some(ReductionAxes::All)) {
        resolved
            .const_values
            .insert(fresh, Value::String("all".into()));
    } else {
        resolved.reduction_dim = Some(fresh);
        resolved
            .const_values
            .insert(fresh, Value::Num((geometry.axis + 1) as f64));
    }
    Ok(resolved)
}

pub(super) fn cardinality(shape: &[usize]) -> Option<usize> {
    if shape.contains(&0) {
        Some(0)
    } else {
        shape
            .iter()
            .try_fold(1usize, |size, dim| size.checked_mul(*dim))
    }
}

pub(super) fn validate_fused_reduction_domain(
    axes: Option<&ReductionAxes>,
    _runtime_shape: Option<&[usize]>,
    _reduce_all: bool,
) -> Result<(), RuntimeError> {
    if matches!(axes, Some(ReductionAxes::Explicit(dims)) if dims.len() != 1) {
        return Err(mex(
            "FusionReductionAxesUnsupported",
            "fusion: multi-axis reduction requires the runtime N-D path",
        ));
    }
    Ok(())
}

fn canonical_reduction_shape(
    runtime_shape: Option<Vec<usize>>,
    rows: usize,
    columns: usize,
) -> Vec<usize> {
    let mut shape = runtime_shape.unwrap_or_else(|| vec![rows, columns]);
    if shape.len() < 2 {
        shape.resize(2, 1);
    }
    shape
}

fn reduced_output_shape(
    mut shape: Vec<usize>,
    axis: usize,
    reduce_all: bool,
) -> Result<Vec<usize>, RuntimeError> {
    if reduce_all {
        return Ok(vec![1, 1]);
    }
    let dim = shape.get_mut(axis).ok_or_else(|| {
        mex(
            "FusionReductionAxisUnsupported",
            "fusion: reduction dimension exceeds supported runtime rank",
        )
    })?;
    *dim = 1;
    Ok(shape)
}

#[cfg(test)]
pub(super) fn fused_reduction_output_shape(
    runtime_shape: Option<Vec<usize>>,
    rows: usize,
    columns: usize,
    axis: usize,
    reduce_all: bool,
) -> Result<Vec<usize>, RuntimeError> {
    reduced_output_shape(
        canonical_reduction_shape(runtime_shape, rows, columns),
        axis,
        reduce_all,
    )
}

fn axis_from_value(value: &Value) -> Result<usize, RuntimeError> {
    let dim = if let Some(integer) = scalar_integer_value(value) {
        integer.try_to_usize()
    } else if let Value::Num(number) = value {
        // usize::MAX rounds upward on 64-bit hosts; use a strict upper bound.
        (number.is_finite()
            && number.fract() == 0.0
            && *number >= 1.0
            && *number < usize::MAX as f64)
            .then_some(*number as usize)
    } else {
        None
    };
    dim.and_then(|dim| dim.checked_sub(1)).ok_or_else(|| {
        mex(
            "FusionReductionAxisUnsupported",
            "fusion: reduction dimension is not an exact positive platform index",
        )
    })
}

pub fn resolve_reduction_geometry(
    plan: &FusionGroupPlan,
    request: &FusionExecutionRequest<'_>,
) -> Result<ReductionGeometry, RuntimeError> {
    if std::env::var("RUNMAT_DISABLE_FUSED_REDUCTION").is_ok() {
        return Err(mex(
            "FusionReductionDisabled",
            "fusion: fused reductions disabled",
        ));
    }
    let shape = data_shape::resolve(plan, &request.inputs)?;
    let total = cardinality(&shape).ok_or_else(|| {
        mex(
            "FusionReductionExtentOverflow",
            "fusion: runtime reduction extent overflows",
        )
    })?;
    if total == 0 {
        // The current fused kernel has no empty semantics. The interpreter owns
        // sum identities, mean NaNs, singleton axes, and any option errors.
        return Err(mex(
            "FusionReductionEmpty",
            "fusion: empty reduction requires runtime semantics",
        ));
    }
    let reduce_all = matches!(plan.reduction_axes, Some(ReductionAxes::All));
    validate_fused_reduction_domain(plan.reduction_axes.as_ref(), Some(&shape), reduce_all)?;
    let rows = shape.first().copied().unwrap_or(1);
    let columns = shape.get(1).copied().unwrap_or(1);
    let explicit = match &plan.reduction_axes {
        Some(ReductionAxes::Explicit(dims)) => Some(dims[0].checked_sub(1).ok_or_else(|| {
            mex(
                "FusionReductionAxisUnsupported",
                "fusion: zero reduction dimension",
            )
        })?),
        _ => None,
    };
    let runtime_axis = plan
        .reduction_dim
        .map(|id| {
            let value = plan
                .inputs
                .iter()
                .position(|input| *input == id)
                .and_then(|index| request.inputs.get(index))
                .or_else(|| plan.const_values.get(&id))
                .ok_or_else(|| {
                    mex(
                        "FusionReductionAxisUnknown",
                        "fusion: unresolved reduction dimension",
                    )
                })?;
            axis_from_value(value)
        })
        .transpose()?;
    if explicit
        .zip(runtime_axis)
        .is_some_and(|(left, right)| left != right)
    {
        return Err(mex(
            "FusionReductionAxisConflict",
            "fusion: cached reduction axis differs from actual control",
        ));
    }
    let axis = if reduce_all {
        0
    } else {
        runtime_axis
            .or(explicit)
            .unwrap_or_else(|| shape.iter().position(|dim| *dim != 1).unwrap_or(0))
    };
    let execution_shape = canonical_reduction_shape(Some(shape), rows, columns);
    let reduce_len = if reduce_all {
        total
    } else {
        *execution_shape.get(axis).ok_or_else(|| {
            mex(
                "FusionReductionAxisUnsupported",
                "fusion: reduction dimension exceeds supported runtime rank",
            )
        })?
    };
    let output_shape = reduced_output_shape(execution_shape, axis, reduce_all)?;
    let num_slices = total.checked_div(reduce_len).ok_or_else(|| {
        mex(
            "FusionReductionGeometryMismatch",
            "fusion: reduction geometry does not cover actual data",
        )
    })?;
    if reduce_len.checked_mul(num_slices) != Some(total)
        || cardinality(&output_shape) != Some(num_slices)
    {
        return Err(mex(
            "FusionReductionGeometryMismatch",
            "fusion: reduction geometry does not cover actual data",
        ));
    }
    Ok(ReductionGeometry {
        axis,
        reduce_len,
        num_slices,
        output_shape,
    })
}

#[cfg(test)]
#[path = "reduction_geometry/tests.rs"]
mod tests;
