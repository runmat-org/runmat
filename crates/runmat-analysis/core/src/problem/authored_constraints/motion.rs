use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use super::{AuthoredConstraintError, AuthoredConstraintTarget};
use crate::problem::{
    constraint::{StructuralKinematicHistory, TranslationDofMotion},
    coordinate_frame::CoordinateFrame,
    signal::{
        GeneratorDomain, PhysicalSignal, SignalExtrapolation, SignalInterpolation, SignalPrecision,
        SignalRepresentation, SignalShape, TabulatedSignal, TabulatedSignalDomain,
    },
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredPrescribedStructuralMotion {
    pub target: AuthoredConstraintTarget,
    pub translations: Vec<TranslationDofMotion>,
}

pub(super) fn validate(
    motion: &AuthoredPrescribedStructuralMotion,
) -> Result<(), AuthoredConstraintError> {
    if motion.translations.is_empty() {
        return Err(AuthoredConstraintError::MissingDofs);
    }
    let mut dofs = BTreeSet::new();
    for translation in &motion.translations {
        if !dofs.insert(translation.dof) {
            return Err(AuthoredConstraintError::DuplicateDof);
        }
        let StructuralKinematicHistory::AccelerationDriven {
            initial_displacement,
            initial_velocity,
            ..
        } = &translation.history;
        if !initial_displacement.is_finite() || !initial_velocity.is_finite() {
            return Err(AuthoredConstraintError::InvalidValue);
        }
    }
    Ok(())
}

pub fn authored_motion_acceleration_is_compatible(signal: &PhysicalSignal) -> bool {
    let PhysicalSignal::Acceleration(signal) = signal else {
        return false;
    };
    base_contract(signal)
        && match &signal.representation {
            SignalRepresentation::Tabulated(table) => table_contract(table),
            SignalRepresentation::Generator(generator) => {
                generator.domain == GeneratorDomain::TimeSeconds
                    && generator.interpolation == SignalInterpolation::Linear
                    && generator.extrapolation == SignalExtrapolation::Reject
            }
            SignalRepresentation::Artifact(_)
            | SignalRepresentation::Constant(_)
            | SignalRepresentation::Waveform(_) => false,
        }
}

pub fn materialized_motion_acceleration_is_compatible(signal: &PhysicalSignal) -> bool {
    let PhysicalSignal::Acceleration(signal) = signal else {
        return false;
    };
    base_contract(signal)
        && matches!(
            &signal.representation,
            SignalRepresentation::Tabulated(table) if table_contract(table)
        )
}

fn base_contract(
    signal: &crate::problem::signal::Signal<crate::problem::quantity::Acceleration>,
) -> bool {
    signal.shape == SignalShape::Vector3
        && signal.frame == Some(CoordinateFrame::Global)
        && signal.precision == SignalPrecision::Real64
}

fn table_contract(table: &TabulatedSignal<crate::problem::quantity::Acceleration>) -> bool {
    table.domain == TabulatedSignalDomain::TimeSeconds
        && table.interpolation == SignalInterpolation::Linear
        && table.extrapolation == SignalExtrapolation::Reject
        && table.knots.len() >= 2
}
