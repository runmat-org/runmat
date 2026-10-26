use serde::{Deserialize, Serialize};

use super::{coordinate_frame::CoordinateFrame, quantity::FiniteQuantity};
use super::{
    quantity::{
        Displacement, ElectricCurrent, ElectricPotential, HeatFlux, MagneticVectorPotential,
        Pressure, Rotation, Temperature, TemperatureDifference, Velocity,
    },
    signal::{PhysicalQuantityKind, PhysicalSignal, SignalId, SignalRepresentation, SignalValue},
};

/// A prescribed scalar or vector quantity authored inline or sourced from a canonical signal.
/// Inline vectors are components in the global frame; signal references retain their own frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrescribedValue<T> {
    Constant(T),
    Signal(SignalId),
}

impl<T: PrescribedQuantity> PrescribedValue<T> {
    pub fn is_well_formed(&self, signals: &[PhysicalSignal]) -> bool {
        match self {
            Self::Constant(value) => value.is_finite(),
            Self::Signal(_) => self.signal(signals).is_some(),
        }
    }

    /// Returns the original quantity/shape-matched signal, without sampling, transforming,
    /// or discarding its frame, table, or provenance.
    pub fn signal<'a>(&self, signals: &'a [PhysicalSignal]) -> Option<&'a PhysicalSignal> {
        let Self::Signal(signal_id) = self else {
            return None;
        };
        let mut matching = signals
            .iter()
            .filter(|signal| signal.signal_id() == signal_id);
        let signal = matching.next()?;
        (matching.next().is_none() && signal.quantity() == T::KIND && signal.shape() == &T::SHAPE)
            .then_some(signal)
    }

    /// Resolves values that are constant for the entire study.
    ///
    /// Time-, frequency-, and parameter-dependent signals are realized by the procedure's signal
    /// evaluator rather than being sampled implicitly at assembly time.
    pub fn study_constant<'a>(&'a self, signals: &'a [PhysicalSignal]) -> Option<&'a T> {
        match self {
            Self::Constant(value) => Some(value),
            Self::Signal(_) => self.signal(signals).and_then(T::constant_from_signal),
        }
    }
}

pub trait PrescribedQuantity: FiniteQuantity {
    const KIND: PhysicalQuantityKind;
    const SHAPE: super::signal::SignalShape;

    fn constant_from_signal(signal: &PhysicalSignal) -> Option<&Self>;
}

macro_rules! prescribed_quantity {
    ($quantity:ty, $kind:ident, $variant:ident) => {
        impl PrescribedQuantity for $quantity {
            const KIND: PhysicalQuantityKind = PhysicalQuantityKind::$kind;
            const SHAPE: super::signal::SignalShape = super::signal::SignalShape::Scalar;

            fn constant_from_signal(signal: &PhysicalSignal) -> Option<&Self> {
                let PhysicalSignal::$variant(signal) = signal else {
                    return None;
                };
                let SignalRepresentation::Constant(SignalValue::Scalar(value)) =
                    &signal.representation
                else {
                    return None;
                };
                Some(value)
            }
        }
    };
}

prescribed_quantity!(Displacement, Displacement, Displacement);
prescribed_quantity!(Rotation, Rotation, Rotation);
prescribed_quantity!(Temperature, Temperature, Temperature);
prescribed_quantity!(
    TemperatureDifference,
    TemperatureDifference,
    TemperatureDifference
);
prescribed_quantity!(Pressure, Pressure, Pressure);
prescribed_quantity!(ElectricPotential, ElectricPotential, ElectricPotential);
prescribed_quantity!(ElectricCurrent, ElectricCurrent, ElectricCurrent);
prescribed_quantity!(
    MagneticVectorPotential,
    MagneticVectorPotential,
    MagneticVectorPotential
);
prescribed_quantity!(Velocity, Velocity, Velocity);
prescribed_quantity!(HeatFlux, HeatFlux, HeatFlux);

// Only the fluid velocity boundary currently owns full-vector prescribed semantics.
// Scalar thermodynamic quantities do not acquire vector boundary semantics merely
// because the general shape/finite-array infrastructure supports vectors.
impl PrescribedQuantity for [Velocity; 3] {
    const KIND: PhysicalQuantityKind = PhysicalQuantityKind::Velocity;
    const SHAPE: super::signal::SignalShape = super::signal::SignalShape::Vector3;

    fn constant_from_signal(signal: &PhysicalSignal) -> Option<&Self> {
        let PhysicalSignal::Velocity(signal) = signal else {
            return None;
        };
        // Borrowing components alone cannot carry a frame transformation.
        if signal.frame.as_ref() != Some(&CoordinateFrame::Global) {
            return None;
        }
        let SignalRepresentation::Constant(SignalValue::Vector3(value)) = &signal.representation
        else {
            return None;
        };
        Some(value)
    }
}

#[cfg(test)]
mod tests;
