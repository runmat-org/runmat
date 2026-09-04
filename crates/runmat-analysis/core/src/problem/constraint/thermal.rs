use serde::{Deserialize, Serialize};

use super::{
    visit_pair, visit_pair_ref, visit_periodic, visit_periodic_ref, PairedConstraint,
    PeriodicConstraint,
};
use crate::problem::{
    prescribed_value::PrescribedValue,
    quantity::{Temperature, TemperatureDifference},
    target::RegionTarget,
};

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
        value: PrescribedValue<Temperature>,
    },
    Periodic(PeriodicConstraint),
    Tie(PairedConstraint),
    AffineJump {
        periodic: PeriodicConstraint,
        temperature_jump: PrescribedValue<TemperatureDifference>,
    },
}

impl ThermalConstraintKind {
    pub(super) fn target(&self) -> Option<&RegionTarget> {
        match self {
            Self::Prescribed { target, .. } => Some(target),
            Self::Periodic(periodic) | Self::AffineJump { periodic, .. } => {
                Some(&periodic.pair.primary)
            }
            Self::Tie(pair) => Some(&pair.primary),
        }
    }

    pub(super) fn visit_targets_mut(&mut self, visit: &mut impl FnMut(&mut RegionTarget)) {
        match self {
            Self::Prescribed { target, .. } => visit(target),
            Self::Periodic(periodic) | Self::AffineJump { periodic, .. } => {
                visit_periodic(periodic, visit)
            }
            Self::Tie(pair) => visit_pair(pair, visit),
        }
    }

    pub(super) fn visit_targets(&self, visit: &mut impl FnMut(&RegionTarget)) {
        match self {
            Self::Prescribed { target, .. } => visit(target),
            Self::Periodic(periodic) | Self::AffineJump { periodic, .. } => {
                visit_periodic_ref(periodic, visit)
            }
            Self::Tie(pair) => visit_pair_ref(pair, visit),
        }
    }
}
