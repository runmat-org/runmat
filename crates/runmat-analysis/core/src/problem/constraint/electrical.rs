use serde::{Deserialize, Serialize};

use super::{visit_pair, visit_pair_ref, PairedConstraint};
use crate::problem::{
    prescribed_value::PrescribedValue,
    quantity::{ElectricCurrent, ElectricPotential},
    target::RegionTarget,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElectricalConstraint {
    pub constraint_id: String,
    pub kind: ElectricalConstraintKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElectricalConstraintKind {
    PrescribedPotential {
        target: RegionTarget,
        value: PrescribedValue<ElectricPotential>,
    },
    Equipotential {
        target: RegionTarget,
    },
    FloatingConductor {
        target: RegionTarget,
    },
    IntegralCurrent {
        target: RegionTarget,
        value: PrescribedValue<ElectricCurrent>,
    },
    Periodic(PairedConstraint),
    PotentialReference {
        target: RegionTarget,
    },
}

impl ElectricalConstraintKind {
    pub(super) fn target(&self) -> Option<&RegionTarget> {
        match self {
            Self::PrescribedPotential { target, .. }
            | Self::Equipotential { target }
            | Self::FloatingConductor { target }
            | Self::IntegralCurrent { target, .. }
            | Self::PotentialReference { target } => Some(target),
            Self::Periodic(pair) => Some(&pair.primary),
        }
    }

    pub(super) fn visit_targets_mut(&mut self, visit: &mut impl FnMut(&mut RegionTarget)) {
        match self {
            Self::PrescribedPotential { target, .. }
            | Self::Equipotential { target }
            | Self::FloatingConductor { target }
            | Self::IntegralCurrent { target, .. }
            | Self::PotentialReference { target } => visit(target),
            Self::Periodic(pair) => visit_pair(pair, visit),
        }
    }

    pub(super) fn visit_targets(&self, visit: &mut impl FnMut(&RegionTarget)) {
        match self {
            Self::PrescribedPotential { target, .. }
            | Self::Equipotential { target }
            | Self::FloatingConductor { target }
            | Self::IntegralCurrent { target, .. }
            | Self::PotentialReference { target } => visit(target),
            Self::Periodic(pair) => visit_pair_ref(pair, visit),
        }
    }
}
