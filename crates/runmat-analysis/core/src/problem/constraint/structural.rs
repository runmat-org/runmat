use serde::{Deserialize, Serialize};

use super::{
    visit_pair, visit_pair_ref, visit_periodic, visit_periodic_ref, PairedConstraint,
    PeriodicConstraint,
};
use crate::problem::{
    coordinate_frame::CoordinateFrame,
    prescribed_value::PrescribedValue,
    quantity::{Displacement, Rotation},
    reference_point::ReferencePoint,
    signal::SignalId,
    target::RegionTarget,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralConstraint {
    pub constraint_id: String,
    pub frame: CoordinateFrame,
    pub kind: StructuralConstraintKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralConstraintKind {
    Fixed {
        target: RegionTarget,
    },
    Prescribed(PrescribedStructuralDofs),
    PrescribedMotion(PrescribedStructuralMotion),
    NormalDisplacement {
        target: RegionTarget,
        value: PrescribedValue<Displacement>,
    },
    Symmetry {
        target: RegionTarget,
    },
    Antisymmetry {
        target: RegionTarget,
    },
    RigidCoupling(ReferencePointCoupling),
    DistributingCoupling(ReferencePointCoupling),
    Tie(PairedConstraint),
    Periodic(PeriodicConstraint),
    Equation(StructuralConstraintEquation),
}

impl StructuralConstraintKind {
    pub fn target(&self) -> Option<&RegionTarget> {
        match self {
            Self::Fixed { target }
            | Self::NormalDisplacement { target, .. }
            | Self::Symmetry { target }
            | Self::Antisymmetry { target } => Some(target),
            Self::Prescribed(value) => Some(&value.target),
            Self::PrescribedMotion(value) => Some(&value.target),
            Self::RigidCoupling(value) | Self::DistributingCoupling(value) => Some(&value.target),
            Self::Tie(value) => Some(&value.primary),
            Self::Periodic(value) => Some(&value.pair.primary),
            Self::Equation(_) => None,
        }
    }

    pub(super) fn visit_targets_mut(&mut self, visit: &mut impl FnMut(&mut RegionTarget)) {
        match self {
            Self::Fixed { target }
            | Self::NormalDisplacement { target, .. }
            | Self::Symmetry { target }
            | Self::Antisymmetry { target } => visit(target),
            Self::Prescribed(value) => visit(&mut value.target),
            Self::PrescribedMotion(value) => visit(&mut value.target),
            Self::RigidCoupling(value) | Self::DistributingCoupling(value) => {
                visit(&mut value.target)
            }
            Self::Tie(pair) => visit_pair(pair, visit),
            Self::Periodic(periodic) => visit_periodic(periodic, visit),
            Self::Equation(equation) => {
                for term in &mut equation.terms {
                    visit(&mut term.target);
                }
            }
        }
    }

    pub(super) fn visit_targets(&self, visit: &mut impl FnMut(&RegionTarget)) {
        match self {
            Self::Fixed { target }
            | Self::NormalDisplacement { target, .. }
            | Self::Symmetry { target }
            | Self::Antisymmetry { target } => visit(target),
            Self::Prescribed(value) => visit(&value.target),
            Self::PrescribedMotion(value) => visit(&value.target),
            Self::RigidCoupling(value) | Self::DistributingCoupling(value) => visit(&value.target),
            Self::Tie(pair) => visit_pair_ref(pair, visit),
            Self::Periodic(periodic) => visit_periodic_ref(periodic, visit),
            Self::Equation(equation) => equation.terms.iter().for_each(|term| visit(&term.target)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralTranslationDof {
    Ux,
    Uy,
    Uz,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralRotationDof {
    Rx,
    Ry,
    Rz,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranslationDofValue {
    pub dof: StructuralTranslationDof,
    pub value: PrescribedValue<Displacement>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotationDofValue {
    pub dof: StructuralRotationDof,
    pub value: PrescribedValue<Rotation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrescribedStructuralDofs {
    pub target: RegionTarget,
    pub translations: Vec<TranslationDofValue>,
    pub rotations: Vec<RotationDofValue>,
}

/// A translational support trajectory reconstructed into compatible displacement, velocity, and
/// acceleration from one quantity-typed source. The solver consumes the resulting canonical
/// kinematics bundle rather than integrating an independent load path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrescribedStructuralMotion {
    pub target: RegionTarget,
    pub translations: Vec<TranslationDofMotion>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranslationDofMotion {
    pub dof: StructuralTranslationDof,
    pub history: StructuralKinematicHistory,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralKinematicHistory {
    AccelerationDriven {
        acceleration: AccelerationSignalId,
        initial_displacement: Displacement,
        initial_velocity: crate::problem::quantity::Velocity,
        reconstruction: StructuralMotionReconstruction,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AccelerationSignalId(pub SignalId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralMotionReconstruction {
    PiecewiseLinearAccelerationExact,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferencePointCoupling {
    pub target: RegionTarget,
    pub reference_point: ReferencePoint,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralConstraintEquation {
    pub terms: Vec<StructuralConstraintEquationTerm>,
    pub right_hand_side: Displacement,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralConstraintEquationTerm {
    pub target: RegionTarget,
    pub dof: StructuralTranslationDof,
    pub coefficient: f64,
}
