use serde::{Deserialize, Serialize};

use super::{
    visit_periodic, visit_periodic_ref, PeriodicConstraint, PhaseShiftedPeriodicConstraint,
};
use crate::problem::{prescribed_value::PrescribedValue, quantity::Pressure, target::RegionTarget};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticConstraint {
    pub constraint_id: String,
    pub kind: AcousticConstraintKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcousticConstraintKind {
    PrescribedPressure {
        target: RegionTarget,
        value: PrescribedValue<Pressure>,
    },
    Periodic(PeriodicConstraint),
    PhaseShiftedPeriodic(PhaseShiftedPeriodicConstraint),
}

impl AcousticConstraintKind {
    pub(super) fn target(&self) -> Option<&RegionTarget> {
        match self {
            Self::PrescribedPressure { target, .. } => Some(target),
            Self::Periodic(periodic) => Some(&periodic.pair.primary),
            Self::PhaseShiftedPeriodic(periodic) => Some(&periodic.periodic.pair.primary),
        }
    }

    pub(super) fn visit_targets_mut(&mut self, visit: &mut impl FnMut(&mut RegionTarget)) {
        match self {
            Self::PrescribedPressure { target, .. } => visit(target),
            Self::Periodic(periodic) => visit_periodic(periodic, visit),
            Self::PhaseShiftedPeriodic(periodic) => visit_periodic(&mut periodic.periodic, visit),
        }
    }

    pub(super) fn visit_targets(&self, visit: &mut impl FnMut(&RegionTarget)) {
        match self {
            Self::PrescribedPressure { target, .. } => visit(target),
            Self::Periodic(periodic) => visit_periodic_ref(periodic, visit),
            Self::PhaseShiftedPeriodic(periodic) => visit_periodic_ref(&periodic.periodic, visit),
        }
    }
}
