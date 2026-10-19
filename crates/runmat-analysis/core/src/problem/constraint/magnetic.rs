use serde::{Deserialize, Serialize};

use super::{
    visit_periodic, visit_periodic_ref, PeriodicConstraint, PhaseShiftedPeriodicConstraint,
};
use crate::problem::{
    prescribed_value::PrescribedValue, quantity::MagneticVectorPotential, target::RegionTarget,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagneticConstraint {
    pub constraint_id: String,
    pub kind: MagneticConstraintKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MagneticConstraintKind {
    PrescribedVectorPotential {
        target: RegionTarget,
        value: PrescribedValue<MagneticVectorPotential>,
    },
    Periodic(PeriodicConstraint),
    PhaseShiftedPeriodic(PhaseShiftedPeriodicConstraint),
    VectorPotentialGround {
        target: RegionTarget,
    },
    CoulombGauge {
        target: RegionTarget,
    },
}

impl MagneticConstraintKind {
    pub(super) fn target(&self) -> Option<&RegionTarget> {
        match self {
            Self::PrescribedVectorPotential { target, .. }
            | Self::VectorPotentialGround { target }
            | Self::CoulombGauge { target } => Some(target),
            Self::Periodic(periodic) => Some(&periodic.pair.primary),
            Self::PhaseShiftedPeriodic(periodic) => Some(&periodic.periodic.pair.primary),
        }
    }

    pub(super) fn visit_targets_mut(&mut self, visit: &mut impl FnMut(&mut RegionTarget)) {
        match self {
            Self::PrescribedVectorPotential { target, .. }
            | Self::VectorPotentialGround { target }
            | Self::CoulombGauge { target } => visit(target),
            Self::Periodic(periodic) => visit_periodic(periodic, visit),
            Self::PhaseShiftedPeriodic(periodic) => visit_periodic(&mut periodic.periodic, visit),
        }
    }

    pub(super) fn visit_targets(&self, visit: &mut impl FnMut(&RegionTarget)) {
        match self {
            Self::PrescribedVectorPotential { target, .. }
            | Self::VectorPotentialGround { target }
            | Self::CoulombGauge { target } => visit(target),
            Self::Periodic(periodic) => visit_periodic_ref(periodic, visit),
            Self::PhaseShiftedPeriodic(periodic) => visit_periodic_ref(&periodic.periodic, visit),
        }
    }
}
