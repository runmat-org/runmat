use serde::{Deserialize, Serialize};

use super::{visit_pair, visit_pair_ref, PairedConstraint};
use crate::problem::{
    quantity::MagneticVectorPotential, schedule::ValueSchedule, target::RegionTarget,
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
        value: ValueSchedule<MagneticVectorPotential>,
    },
    Periodic(PairedConstraint),
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
            Self::Periodic(pair) => Some(&pair.primary),
        }
    }

    pub(super) fn visit_targets_mut(&mut self, visit: &mut impl FnMut(&mut RegionTarget)) {
        match self {
            Self::PrescribedVectorPotential { target, .. }
            | Self::VectorPotentialGround { target }
            | Self::CoulombGauge { target } => visit(target),
            Self::Periodic(pair) => visit_pair(pair, visit),
        }
    }

    pub(super) fn visit_targets(&self, visit: &mut impl FnMut(&RegionTarget)) {
        match self {
            Self::PrescribedVectorPotential { target, .. }
            | Self::VectorPotentialGround { target }
            | Self::CoulombGauge { target } => visit(target),
            Self::Periodic(pair) => visit_pair_ref(pair, visit),
        }
    }
}
