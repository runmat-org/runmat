use super::*;
use crate::builtins::fea::values::{PRESSURE_CLASS, VELOCITY_CLASS};

define_simple_constraint!(
    fluid_no_slip_wall_builtin,
    FLUID_NO_SLIP_WALL_DESCRIPTOR,
    FLUID_NO_SLIP_WALL_SIGNATURES,
    "fea.FluidConstraint.noSlipWall",
    "constraint = fea.FluidConstraint.noSlipWall(id, region)",
    "Create a fluid no-slip wall constraint.",
    FEA_FLUID_CONSTRAINT_CLASS,
    |constraint_id, target| FluidConstraint {
        constraint_id,
        kind: FluidConstraintKind::NoSlipWall { target },
    }
);

define_simple_constraint!(
    fluid_slip_wall_builtin,
    FLUID_SLIP_WALL_DESCRIPTOR,
    FLUID_SLIP_WALL_SIGNATURES,
    "fea.FluidConstraint.slipWall",
    "constraint = fea.FluidConstraint.slipWall(id, region)",
    "Create a fluid slip-wall constraint.",
    FEA_FLUID_CONSTRAINT_CLASS,
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
    |constraint_id, target| FluidConstraint {
        constraint_id,
        kind: FluidConstraintKind::Symmetry { target },
    }
);

define_quantity_constraint!(
    fluid_prescribed_velocity_builtin,
    FLUID_PRESCRIBED_VELOCITY_DESCRIPTOR,
    FLUID_PRESCRIBED_VELOCITY_SIGNATURES,
    FLUID_PRESCRIBED_VELOCITY_INPUTS,
    "fea.FluidConstraint.prescribedVelocity",
    "constraint = fea.FluidConstraint.prescribedVelocity(id, region, velocity)",
    "Create a prescribed fluid-velocity constraint.",
    FEA_FLUID_CONSTRAINT_CLASS,
    "velocity",
    VELOCITY_CLASS,
    "meters_per_second",
    |constraint_id, target, meters_per_second| FluidConstraint {
        constraint_id,
        kind: FluidConstraintKind::PrescribedVelocity {
            target,
            value: PrescribedValue::Constant(Velocity::new(meters_per_second)),
        },
    }
);

define_quantity_constraint!(
    fluid_moving_wall_builtin,
    FLUID_MOVING_WALL_DESCRIPTOR,
    FLUID_MOVING_WALL_SIGNATURES,
    FLUID_MOVING_WALL_INPUTS,
    "fea.FluidConstraint.movingWall",
    "constraint = fea.FluidConstraint.movingWall(id, region, velocity)",
    "Create a moving-wall velocity constraint.",
    FEA_FLUID_CONSTRAINT_CLASS,
    "velocity",
    VELOCITY_CLASS,
    "meters_per_second",
    |constraint_id, target, meters_per_second| FluidConstraint {
        constraint_id,
        kind: FluidConstraintKind::MovingWall {
            target,
            velocity: PrescribedValue::Constant(Velocity::new(meters_per_second)),
        },
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
