use serde::{Deserialize, Serialize};

use super::{
    quantity::{ComplexPhase, Length, Position, Rotation, Translation},
    target::RegionTarget,
};

mod acoustic;
mod electrical;
mod fluid;
mod magnetic;
mod structural;
mod thermal;

pub use acoustic::{AcousticConstraint, AcousticConstraintKind};
pub use electrical::{ElectricalConstraint, ElectricalConstraintKind};
pub use fluid::{FluidConstraint, FluidConstraintKind};
pub use magnetic::{MagneticConstraint, MagneticConstraintKind};
pub use structural::{
    AccelerationSignalId, PrescribedStructuralDofs, PrescribedStructuralMotion,
    ReferencePointCoupling, RotationDofValue, StructuralConstraint, StructuralConstraintEquation,
    StructuralConstraintEquationTerm, StructuralConstraintKind, StructuralKinematicHistory,
    StructuralMotionReconstruction, StructuralRotationDof, StructuralTranslationDof,
    TranslationDofMotion, TranslationDofValue,
};
pub use thermal::{ThermalConstraint, ThermalConstraintKind};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisConstraint {
    Structural(StructuralConstraint),
    Thermal(ThermalConstraint),
    Electrical(ElectricalConstraint),
    Magnetic(MagneticConstraint),
    Acoustic(AcousticConstraint),
    Fluid(FluidConstraint),
}

impl AnalysisConstraint {
    pub fn constraint_id(&self) -> &str {
        match self {
            Self::Structural(value) => &value.constraint_id,
            Self::Thermal(value) => &value.constraint_id,
            Self::Electrical(value) => &value.constraint_id,
            Self::Magnetic(value) => &value.constraint_id,
            Self::Acoustic(value) => &value.constraint_id,
            Self::Fluid(value) => &value.constraint_id,
        }
    }

    pub fn target(&self) -> Option<&RegionTarget> {
        match self {
            Self::Structural(value) => value.kind.target(),
            Self::Thermal(value) => value.kind.target(),
            Self::Electrical(value) => value.kind.target(),
            Self::Magnetic(value) => value.kind.target(),
            Self::Acoustic(value) => value.kind.target(),
            Self::Fluid(value) => value.kind.target(),
        }
    }

    pub fn structural(&self) -> Option<&StructuralConstraint> {
        match self {
            Self::Structural(value) => Some(value),
            _ => None,
        }
    }

    pub fn visit_targets(&self, mut visit: impl FnMut(&RegionTarget)) {
        match self {
            Self::Structural(value) => value.kind.visit_targets(&mut visit),
            Self::Thermal(value) => value.kind.visit_targets(&mut visit),
            Self::Electrical(value) => value.kind.visit_targets(&mut visit),
            Self::Magnetic(value) => value.kind.visit_targets(&mut visit),
            Self::Acoustic(value) => value.kind.visit_targets(&mut visit),
            Self::Fluid(value) => value.kind.visit_targets(&mut visit),
        }
    }

    pub fn visit_targets_mut(&mut self, mut visit: impl FnMut(&mut RegionTarget)) {
        match self {
            Self::Structural(value) => value.kind.visit_targets_mut(&mut visit),
            Self::Thermal(value) => value.kind.visit_targets_mut(&mut visit),
            Self::Electrical(value) => value.kind.visit_targets_mut(&mut visit),
            Self::Magnetic(value) => value.kind.visit_targets_mut(&mut visit),
            Self::Acoustic(value) => value.kind.visit_targets_mut(&mut visit),
            Self::Fluid(value) => value.kind.visit_targets_mut(&mut visit),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedConstraint {
    pub primary: RegionTarget,
    pub secondary: RegionTarget,
    pub pairing: PairingPolicy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairingPolicy {
    Matching,
    GeometricTolerance(Length),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeriodicConstraint {
    pub pair: PairedConstraint,
    pub transform: PeriodicTransform,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PeriodicTransform {
    Identity,
    Translation(Translation),
    Rotation {
        origin: Position,
        axis: [f64; 3],
        angle: Rotation,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseShiftedPeriodicConstraint {
    pub periodic: PeriodicConstraint,
    pub phase: ComplexPhase,
}

pub(super) fn visit_pair_ref(pair: &PairedConstraint, visit: &mut impl FnMut(&RegionTarget)) {
    visit(&pair.primary);
    visit(&pair.secondary);
}

pub(super) fn visit_pair(pair: &mut PairedConstraint, visit: &mut impl FnMut(&mut RegionTarget)) {
    visit(&mut pair.primary);
    visit(&mut pair.secondary);
}

pub(super) fn visit_periodic_ref(
    periodic: &PeriodicConstraint,
    visit: &mut impl FnMut(&RegionTarget),
) {
    visit_pair_ref(&periodic.pair, visit);
}

pub(super) fn visit_periodic(
    periodic: &mut PeriodicConstraint,
    visit: &mut impl FnMut(&mut RegionTarget),
) {
    visit_pair(&mut periodic.pair, visit);
}
