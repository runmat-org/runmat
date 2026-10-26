//! Prepare selected model controls before any starting-point conversion.
use super::*;
pub(in crate::builtins::math::optim) async fn prepare(
    options: Option<&Value>,
    nonlinear: bool,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<PreparedOptions> {
    crate::context::host_allocation::require_unbounded_workspace(
        cooperation.runtime(),
        super::super::NAME,
        "modeled nonlinear option and solver workspace require unbounded host memory",
    )?;
    let default = super::super::empty();
    let options = options.unwrap_or(&default);
    // The modeled consumer has no selected derivative callback. Retained
    // declarations that would claim one are refused, not silently ignored.
    let structure = match options {
        Value::Struct(value) => Some(value),
        Value::Tensor(value) if value.is_empty() => None,
        _ => {
            return Err(super::super::invalid(
                "options must be [] or a scalar structure",
            ))
        }
    };
    let fields = ScalarOptionFields::from_struct(structure);
    for field in [
        Field::HessianFcn,
        Field::HessianMultiplyFcn,
        Field::HessPattern,
    ] {
        if fields.exact(field).is_some() {
            return Err(super::super::invalid(format!(
                "{} is not available for modeled numerical derivatives",
                field.name()
            )));
        }
    }
    let controls = Controls::parse(options.clone(), nonlinear, cooperation).await?;
    if controls.objective_gradient || controls.constraint_gradient {
        return Err(super::super::invalid(
            "modeled analytic derivatives are not available for this solver consumer",
        ));
    }
    if controls.algorithm == OptimAlgorithm::TrustRegionReflective {
        return Err(super::super::invalid(
            "modeled trust-region-reflective requires analytic derivatives",
        ));
    }

    Ok(PreparedOptions { controls })
}
