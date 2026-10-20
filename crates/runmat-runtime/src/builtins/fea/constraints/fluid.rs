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
    ($function:ident, $descriptor:ident, $signatures:ident, $inputs:ident,
     $name:literal, $label:literal, $summary:literal, $build:expr) => {
        const $inputs: [BuiltinParamDescriptor; 3] = [
            BuiltinParamDescriptor { name: "id", ty: BuiltinParamType::StringScalar, arity: BuiltinParamArity::Required, default: None, description: "Constraint identifier." },
            BuiltinParamDescriptor { name: "region", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "Exact geometry.Region boundary target." },
            BuiltinParamDescriptor { name: "velocity", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "Three typed fea.Velocity components in a cell vector (global), or a framed vector fea.VelocitySignal retained by the model; no scalar speed." },
        ];
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor { label: $label, inputs: &$inputs, outputs: &OUT_ANY }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor { signatures: &$signatures, output_mode: BuiltinOutputMode::Fixed, completion_policy: BuiltinCompletionPolicy::Public, errors: &ERRORS };
        #[runtime_builtin(name = $name, category = "fea", summary = $summary,
            keywords = "fea,constraint,region,velocity,vector,typed",
            descriptor($descriptor), integer_audit(TYPED_CONSTRAINT_INTEGER_AUDIT),
            builtin_path = "crate::builtins::fea::constraints::fluid")]
        pub async fn $function(id: String, region: Value, value: Value) -> BuiltinResult<Value> {
            let (id, geometry_id, geometry_revision, target) = bound_region($name, id, &region)?;
            let velocity = vector_velocity($name, &value)?;
            let constraint = ($build)(id, target, velocity);
            bound_constraint_to_object($name, FEA_FLUID_CONSTRAINT_CLASS, geometry_id, geometry_revision, constraint)
        }
    };
}

define_simple_constraint!(
    fluid_no_slip_wall_builtin,
    FLUID_NO_SLIP_WALL_DESCRIPTOR,
    FLUID_NO_SLIP_WALL_SIGNATURES,
    "fea.FluidConstraint.noSlipWall",
    "constraint = fea.FluidConstraint.noSlipWall(id, region)",
    "Create a fluid no-slip wall constraint.",
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
    FLUID_SLIP_WALL_DESCRIPTOR,
    FLUID_SLIP_WALL_SIGNATURES,
    "fea.FluidConstraint.slipWall",
    "constraint = fea.FluidConstraint.slipWall(id, region)",
    "Create a fluid slip-wall constraint.",
    FEA_FLUID_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::fluid",
    |constraint_id, target| FluidConstraint {
        constraint_id,
        kind: FluidConstraintKind::SlipWall { target },
    }
);

define_simple_constraint!(
    fluid_symmetry_builtin,
    FLUID_SYMMETRY_DESCRIPTOR,
    FLUID_SYMMETRY_SIGNATURES,
    "fea.FluidConstraint.symmetry",
    "constraint = fea.FluidConstraint.symmetry(id, region)",
    "Create a fluid symmetry constraint.",
    FEA_FLUID_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::fluid",
    |constraint_id, target| FluidConstraint {
        constraint_id,
        kind: FluidConstraintKind::Symmetry { target },
    }
);

define_vector_velocity_constraint!(
    fluid_prescribed_velocity_builtin,
    FLUID_PRESCRIBED_VELOCITY_DESCRIPTOR,
    FLUID_PRESCRIBED_VELOCITY_SIGNATURES,
    FLUID_PRESCRIBED_VELOCITY_INPUTS,
    "fea.FluidConstraint.prescribedVelocity",
    "constraint = fea.FluidConstraint.prescribedVelocity(id, region, velocity)",
    "Create an exact three-component prescribed fluid-velocity constraint.",
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
    FLUID_MOVING_WALL_DESCRIPTOR,
    FLUID_MOVING_WALL_SIGNATURES,
    FLUID_MOVING_WALL_INPUTS,
    "fea.FluidConstraint.movingWall",
    "constraint = fea.FluidConstraint.movingWall(id, region, velocity)",
    "Create an exact three-component moving-wall velocity constraint.",
    |constraint_id, target, velocity| FluidConstraint {
        constraint_id,
        kind: FluidConstraintKind::MovingWall { target, velocity },
    }
);

define_quantity_constraint!(
    fluid_pressure_reference_builtin,
    FLUID_PRESSURE_REFERENCE_DESCRIPTOR,
    FLUID_PRESSURE_REFERENCE_SIGNATURES,
    FLUID_PRESSURE_REFERENCE_INPUTS,
    "fea.FluidConstraint.pressureReference",
    "constraint = fea.FluidConstraint.pressureReference(id, region, pressure)",
    "Create a fluid pressure-reference constraint.",
    FEA_FLUID_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::fluid",
    "pressure",
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
