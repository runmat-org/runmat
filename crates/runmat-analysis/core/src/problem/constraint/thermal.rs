use serde::{Deserialize, Serialize};

use super::{visit_pair, visit_pair_ref, PairedConstraint};
use crate::problem::{quantity::Temperature, schedule::ValueSchedule, target::RegionTarget};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalConstraint {
    pub constraint_id: String,
    pub kind: ThermalConstraintKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThermalConstraintKind {
    Prescribed {
        target: RegionTarget,
        value: ValueSchedule<Temperature>,
    },
    Periodic(PairedConstraint),
    Tie(PairedConstraint),
    AffineJump {
        pair: PairedConstraint,
        temperature_jump: ValueSchedule<Temperature>,
    },
}

impl ThermalConstraintKind {
    pub(super) fn target(&self) -> Option<&RegionTarget> {
        match self {
            Self::Prescribed { target, .. } => Some(target),
            Self::Periodic(pair) | Self::Tie(pair) | Self::AffineJump { pair, .. } => {
                Some(&pair.primary)
            }
        }
    }

    pub(super) fn visit_targets_mut(&mut self, visit: &mut impl FnMut(&mut RegionTarget)) {
        match self {
            Self::Prescribed { target, .. } => visit(target),
            Self::Periodic(pair) | Self::Tie(pair) | Self::AffineJump { pair, .. } => {
                visit_pair(pair, visit)
            }
        }
    }

    pub(super) fn visit_targets(&self, visit: &mut impl FnMut(&RegionTarget)) {
        match self {
            Self::Prescribed { target, .. } => visit(target),
            Self::Periodic(pair) | Self::Tie(pair) | Self::AffineJump { pair, .. } => {
                visit_pair_ref(pair, visit)
            }
        }
    }
}
