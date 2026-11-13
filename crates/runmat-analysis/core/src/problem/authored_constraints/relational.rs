use serde::{Deserialize, Serialize};

use super::{
    AuthoredConstraint, AuthoredConstraintError, AuthoredConstraintKind, AuthoredConstraintTarget,
};
use crate::problem::{
    constraint::{StructuralRotationDof, StructuralTranslationDof},
    coordinate_frame::CoordinateFrame,
    prescribed_value::PrescribedValue,
    quantity::{
        ComplexPhase, Displacement, Length, Position, Rotation, TemperatureDifference, Translation,
    },
    reference_point::ReferencePoint,
};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredPairedConstraint {
    pub primary: AuthoredConstraintTarget,
    pub secondary: AuthoredConstraintTarget,
    pub pairing: AuthoredPairingPolicy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredPairingPolicy {
    Matching,
    GeometricTolerance(Length),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredPeriodicConstraint {
    pub pair: AuthoredPairedConstraint,
    pub transform: AuthoredPeriodicTransform,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredPeriodicTransform {
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
pub struct AuthoredPhaseShiftedPeriodicConstraint {
    pub periodic: AuthoredPeriodicConstraint,
    pub phase: ComplexPhase,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredStructuralDofs {
    pub target: AuthoredConstraintTarget,
    pub frame: CoordinateFrame,
    pub translations: Vec<AuthoredTranslationDofValue>,
    pub rotations: Vec<AuthoredRotationDofValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredTranslationDofValue {
    pub dof: StructuralTranslationDof,
    pub value: PrescribedValue<Displacement>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredRotationDofValue {
    pub dof: StructuralRotationDof,
    pub value: PrescribedValue<Rotation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredReferencePointCoupling {
    pub target: AuthoredConstraintTarget,
    pub reference_point: ReferencePoint,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredStructuralConstraintEquation {
    pub terms: Vec<AuthoredStructuralConstraintEquationTerm>,
    pub right_hand_side: Displacement,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredStructuralConstraintEquationTerm {
    pub target: AuthoredConstraintTarget,
    pub dof: StructuralTranslationDof,
    pub coefficient: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredThermalJump {
    pub periodic: AuthoredPeriodicConstraint,
    pub temperature_jump: PrescribedValue<TemperatureDifference>,
}

pub(super) fn validate(value: &AuthoredConstraint) -> Result<(), AuthoredConstraintError> {
    match &value.kind {
        AuthoredConstraintKind::StructuralPrescribed(dofs) => validate_dofs(dofs),
        AuthoredConstraintKind::StructuralPrescribedMotion(motion) => {
            super::motion::validate(motion)
        }
        AuthoredConstraintKind::StructuralRigidCoupling(coupling)
        | AuthoredConstraintKind::StructuralDistributingCoupling(coupling) => {
            if coupling
                .reference_point
                .reference_point_id
                .trim()
                .is_empty()
                || !coupling.reference_point.position.is_finite()
            {
                Err(AuthoredConstraintError::InvalidValue)
            } else {
                Ok(())
            }
        }
        AuthoredConstraintKind::StructuralTie(pair) | AuthoredConstraintKind::ThermalTie(pair) => {
            validate_pair(pair)
        }
        AuthoredConstraintKind::StructuralPeriodic(periodic)
        | AuthoredConstraintKind::ThermalPeriodic(periodic)
        | AuthoredConstraintKind::ElectricalPeriodic(periodic)
        | AuthoredConstraintKind::MagneticPeriodic(periodic)
        | AuthoredConstraintKind::AcousticPeriodic(periodic)
        | AuthoredConstraintKind::FluidPeriodic(periodic) => validate_periodic(periodic),
        AuthoredConstraintKind::ThermalAffineJump(jump) => validate_periodic(&jump.periodic),
        AuthoredConstraintKind::MagneticPhaseShiftedPeriodic(periodic)
        | AuthoredConstraintKind::AcousticPhaseShiftedPeriodic(periodic) => {
            validate_periodic(&periodic.periodic)?;
            finite(periodic.phase.canonical_value())
        }
        AuthoredConstraintKind::StructuralEquation(equation) => validate_equation(equation),
        _ => Ok(()),
    }
}

fn validate_dofs(value: &AuthoredStructuralDofs) -> Result<(), AuthoredConstraintError> {
    crate::validate_coordinate_frame(&value.frame)
        .map_err(|_| AuthoredConstraintError::InvalidFrame)?;
    if value.translations.is_empty() && value.rotations.is_empty() {
        return Err(AuthoredConstraintError::MissingDofs);
    }
    let mut translations = BTreeSet::new();
    for entry in &value.translations {
        if !translations.insert(entry.dof) {
            return Err(AuthoredConstraintError::DuplicateDof);
        }
        if let PrescribedValue::Constant(value) = entry.value {
            finite(value.canonical_value())?;
        }
    }
    let mut rotations = BTreeSet::new();
    for entry in &value.rotations {
        if !rotations.insert(entry.dof) {
            return Err(AuthoredConstraintError::DuplicateDof);
        }
        if let PrescribedValue::Constant(value) = entry.value {
            finite(value.canonical_value())?;
        }
    }
    Ok(())
}

fn validate_pair(value: &AuthoredPairedConstraint) -> Result<(), AuthoredConstraintError> {
    if value.primary.region == value.secondary.region {
        return Err(AuthoredConstraintError::IdenticalTargets);
    }
    match value.pairing {
        AuthoredPairingPolicy::Matching => Ok(()),
        AuthoredPairingPolicy::GeometricTolerance(value) => nonnegative(value.canonical_value()),
    }
}

fn validate_periodic(value: &AuthoredPeriodicConstraint) -> Result<(), AuthoredConstraintError> {
    validate_pair(&value.pair)?;
    match value.transform {
        AuthoredPeriodicTransform::Identity => Ok(()),
        AuthoredPeriodicTransform::Translation(value) => {
            value.canonical_value().into_iter().try_for_each(finite)
        }
        AuthoredPeriodicTransform::Rotation {
            origin,
            axis,
            angle,
        } => {
            origin.canonical_value().into_iter().try_for_each(finite)?;
            axis.into_iter().try_for_each(finite)?;
            if axis.into_iter().all(|value| value == 0.0) {
                return Err(AuthoredConstraintError::InvalidTransform);
            }
            finite(angle.canonical_value())
        }
    }
}

fn validate_equation(
    value: &AuthoredStructuralConstraintEquation,
) -> Result<(), AuthoredConstraintError> {
    if value.terms.is_empty() {
        return Err(AuthoredConstraintError::MissingTerms);
    }
    finite(value.right_hand_side.canonical_value())?;
    for term in &value.terms {
        finite(term.coefficient)?;
        if term.coefficient == 0.0 {
            return Err(AuthoredConstraintError::ZeroCoefficient);
        }
    }
    Ok(())
}

fn nonnegative(value: f64) -> Result<(), AuthoredConstraintError> {
    finite(value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(AuthoredConstraintError::InvalidValue)
    }
}

fn finite(value: f64) -> Result<(), AuthoredConstraintError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(AuthoredConstraintError::InvalidValue)
    }
}
