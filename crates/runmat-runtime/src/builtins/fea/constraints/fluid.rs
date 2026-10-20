use super::*;
use crate::builtins::fea::values::{PRESSURE_CLASS, VELOCITY_CLASS};
use runmat_analysis_core::{PhysicalQuantityKind, SignalShape};

fn vector_velocity(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<PrescribedValue<[Velocity; 3]>> {
    if let Value::Cell(cell) = value {
        if cell.data().len() != 3 || (cell.rows != 1 && cell.cols != 1) {
            return Err(builtin_error(builtin, &ERROR_INPUT, "velocity requires exactly three typed fea.Velocity components in global coordinates"));
        }
        let mut components = [Velocity::new(0.0); 3];
        for (component, value) in components.iter_mut().zip(cell.data()) {
            *component = Velocity::new(quantity_scalar(
                builtin,
                value,
                VELOCITY_CLASS,
                "meters_per_second",
            )?);
        }
        return Ok(PrescribedValue::Constant(components));
    }
    let signal = crate::builtins::fea::signals::physical_signal_from_value(builtin, value)?;
    let reference = PrescribedValue::<[Velocity; 3]>::Signal(signal.signal_id().clone());
    if signal.quantity() != PhysicalQuantityKind::Velocity
        || signal.shape() != &SignalShape::Vector3
        || signal.frame().is_none()
        || !reference.is_well_formed(std::slice::from_ref(&signal))
    {
        return Err(builtin_error(builtin, &ERROR_INPUT, "velocity requires a framed three-component fea.VelocitySignal; scalar speed is not a vector boundary condition"));
    }
    // The model retains this exact signal separately. Do not sample, transform,
    // or collapse it to a magnitude while constructing a boundary constraint.
    Ok(reference)
}

macro_rules! define_vector_velocity_constraint {
    ($function:ident, $descriptor:ident, $name:literal, $build:expr) => {
        #[runtime_builtin(
                                    name = $name,
                                    binding_variant = "default",
                                    builtin_path = "crate::builtins::fea::constraints::fluid"
                                )]
        pub async fn $function(id: String, region: Value, value: Value) -> BuiltinResult<Value> {
            fixed_outputs(&runmat_builtins::$descriptor, $name)?;
            let (id, geometry_id, geometry_revision, target) = bound_region($name, id, &region)?;
            let velocity = vector_velocity($name, &value)?;
            let constraint = ($build)(id, target, velocity);
            bound_constraint_to_object(
                $name,
                FEA_FLUID_CONSTRAINT_CLASS,
                geometry_id,
                geometry_revision,
                constraint,
            )
        }
    };
}

define_simple_constraint!(
    fluid_no_slip_wall_builtin,
    FEA_FLUID_CONSTRAINT_NO_SLIP_WALL_DESCRIPTOR,
    "fea.FluidConstraint.noSlipWall",
    FEA_FLUID_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::fluid",
    |constraint_id, target| FluidConstraint {
        constraint_id,
        kind: FluidConstraintKind::NoSlipWall { target },
    }
);

#[cfg(test)]
mod vector_tests;

define_simple_constraint!(
    fluid_slip_wall_builtin,
    FEA_FLUID_CONSTRAINT_SLIP_WALL_DESCRIPTOR,
    "fea.FluidConstraint.slipWall",
    FEA_FLUID_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::fluid",
    |constraint_id, target| FluidConstraint {
        constraint_id,
        kind: FluidConstraintKind::SlipWall { target },
    }
);

define_simple_constraint!(
    fluid_symmetry_builtin,
    FEA_FLUID_CONSTRAINT_SYMMETRY_DESCRIPTOR,
    "fea.FluidConstraint.symmetry",
    FEA_FLUID_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::fluid",
    |constraint_id, target| FluidConstraint {
        constraint_id,
        kind: FluidConstraintKind::Symmetry { target },
    }
);

define_vector_velocity_constraint!(
    fluid_prescribed_velocity_builtin,
    FEA_FLUID_CONSTRAINT_PRESCRIBED_VELOCITY_DESCRIPTOR,
    "fea.FluidConstraint.prescribedVelocity",
    |constraint_id, target, velocity| FluidConstraint {
        constraint_id,
        kind: FluidConstraintKind::PrescribedVelocity {
            target,
            value: velocity,
        },
    }
);

define_vector_velocity_constraint!(
    fluid_moving_wall_builtin,
    FEA_FLUID_CONSTRAINT_MOVING_WALL_DESCRIPTOR,
    "fea.FluidConstraint.movingWall",
    |constraint_id, target, velocity| FluidConstraint {
        constraint_id,
        kind: FluidConstraintKind::MovingWall { target, velocity },
    }
);

define_quantity_constraint!(
    fluid_pressure_reference_builtin,
    FEA_FLUID_CONSTRAINT_PRESSURE_REFERENCE_DESCRIPTOR,
    "fea.FluidConstraint.pressureReference",
    FEA_FLUID_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::fluid",
    PRESSURE_CLASS,
    "pascals",
    |constraint_id, target, pascals| FluidConstraint {
        constraint_id,
        kind: FluidConstraintKind::PressureReference {
            target,
            value: PrescribedValue::Constant(Pressure::new(pascals)),
        },
    }
);
