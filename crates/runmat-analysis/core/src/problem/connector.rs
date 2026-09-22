use serde::{Deserialize, Serialize};

use super::{
    quantity::{Damping, FoundationModulus, Stiffness},
    target::RegionTarget,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisConnector {
    pub connector_id: String,
    pub kind: ConnectorKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectorKind {
    AxialSpring {
        primary: RegionTarget,
        secondary: RegionTarget,
        stiffness: Stiffness,
    },
    AxialDashpot {
        primary: RegionTarget,
        secondary: RegionTarget,
        damping: Damping,
    },
    IsotropicGroundSpring {
        target: RegionTarget,
        stiffness: Stiffness,
    },
    SurfaceNormalFoundation {
        target: RegionTarget,
        modulus: FoundationModulus,
    },
}

impl ConnectorKind {
    pub fn visit_targets(&self, mut visit: impl FnMut(&RegionTarget)) {
        match self {
            Self::AxialSpring {
                primary, secondary, ..
            }
            | Self::AxialDashpot {
                primary, secondary, ..
            } => {
                visit(primary);
                visit(secondary);
            }
            Self::IsotropicGroundSpring { target, .. }
            | Self::SurfaceNormalFoundation { target, .. } => visit(target),
        }
    }

    pub fn visit_targets_mut(&mut self, mut visit: impl FnMut(&mut RegionTarget)) {
        match self {
            Self::AxialSpring {
                primary, secondary, ..
            }
            | Self::AxialDashpot {
                primary, secondary, ..
            } => {
                visit(primary);
                visit(secondary);
            }
            Self::IsotropicGroundSpring { target, .. }
            | Self::SurfaceNormalFoundation { target, .. } => visit(target),
        }
    }
}
