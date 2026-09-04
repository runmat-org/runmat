use serde::{Deserialize, Serialize};

use super::{visit_periodic, visit_periodic_ref, PeriodicConstraint};
use crate::problem::{
    prescribed_value::PrescribedValue,
    quantity::{Pressure, Velocity},
    target::RegionTarget,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidConstraint {
    pub constraint_id: String,
    pub kind: FluidConstraintKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FluidConstraintKind {
    PrescribedVelocity {
        target: RegionTarget,
        value: PrescribedValue<Velocity>,
    },
    MovingWall {
        target: RegionTarget,
        velocity: PrescribedValue<Velocity>,
    },
    NoSlipWall {
        target: RegionTarget,
    },
    SlipWall {
        target: RegionTarget,
    },
    Symmetry {
        target: RegionTarget,
    },
    Periodic(PeriodicConstraint),
    PressureReference {
        target: RegionTarget,
        value: PrescribedValue<Pressure>,
    },
}

impl FluidConstraintKind {
    pub(super) fn target(&self) -> Option<&RegionTarget> {
        match self {
            Self::PrescribedVelocity { target, .. }
            | Self::MovingWall { target, .. }
            | Self::NoSlipWall { target }
            | Self::SlipWall { target }
            | Self::Symmetry { target }
            | Self::PressureReference { target, .. } => Some(target),
            Self::Periodic(periodic) => Some(&periodic.pair.primary),
        }
    }

    pub(super) fn visit_targets_mut(&mut self, visit: &mut impl FnMut(&mut RegionTarget)) {
        match self {
            Self::PrescribedVelocity { target, .. }
            | Self::MovingWall { target, .. }
            | Self::NoSlipWall { target }
            | Self::SlipWall { target }
            | Self::Symmetry { target }
            | Self::PressureReference { target, .. } => visit(target),
            Self::Periodic(periodic) => visit_periodic(periodic, visit),
        }
    }

    pub(super) fn visit_targets(&self, visit: &mut impl FnMut(&RegionTarget)) {
        match self {
            Self::PrescribedVelocity { target, .. }
            | Self::MovingWall { target, .. }
            | Self::NoSlipWall { target }
            | Self::SlipWall { target }
            | Self::Symmetry { target }
            | Self::PressureReference { target, .. } => visit(target),
            Self::Periodic(periodic) => visit_periodic_ref(periodic, visit),
        }
    }
}
