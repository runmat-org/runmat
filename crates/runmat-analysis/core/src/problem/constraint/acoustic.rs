use serde::{Deserialize, Serialize};

use super::{visit_pair, visit_pair_ref, PairedConstraint};
use crate::problem::{quantity::Pressure, schedule::ValueSchedule, target::RegionTarget};

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
        value: ValueSchedule<Pressure>,
    },
    Periodic(PairedConstraint),
}

impl AcousticConstraintKind {
    pub(super) fn target(&self) -> Option<&RegionTarget> {
        match self {
            Self::PrescribedPressure { target, .. } => Some(target),
            Self::Periodic(pair) => Some(&pair.primary),
        }
    }

    pub(super) fn visit_targets_mut(&mut self, visit: &mut impl FnMut(&mut RegionTarget)) {
        match self {
            Self::PrescribedPressure { target, .. } => visit(target),
            Self::Periodic(pair) => visit_pair(pair, visit),
        }
    }

    pub(super) fn visit_targets(&self, visit: &mut impl FnMut(&RegionTarget)) {
        match self {
            Self::PrescribedPressure { target, .. } => visit(target),
            Self::Periodic(pair) => visit_pair_ref(pair, visit),
        }
    }
}
