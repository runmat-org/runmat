use serde::{Deserialize, Serialize};

use super::quantity::FiniteQuantity;
use super::{
    quantity::{
        Displacement, ElectricCurrent, ElectricPotential, HeatFlux, MagneticVectorPotential,
        Pressure, Rotation, Temperature, TemperatureDifference, Velocity,
    },
    signal::{PhysicalQuantityKind, PhysicalSignal, SignalId, SignalRepresentation, SignalValue},
};

/// A scalar prescribed quantity authored inline or sourced from the canonical signal model.
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
            Self::Signal(signal_id) => signals.iter().any(|signal| {
                signal.signal_id() == signal_id
                    && signal.quantity() == T::KIND
                    && signal.shape() == &super::signal::SignalShape::Scalar
            }),
        }
    }

    /// Resolves values that are constant for the entire study.
    ///
    /// Time-, frequency-, and parameter-dependent signals are realized by the procedure's signal
    /// evaluator rather than being sampled implicitly at assembly time.
    pub fn study_constant<'a>(&'a self, signals: &'a [PhysicalSignal]) -> Option<&'a T> {
        match self {
            Self::Constant(value) => Some(value),
            Self::Signal(signal_id) => signals
                .iter()
                .find(|signal| signal.signal_id() == signal_id)
                .and_then(T::constant_from_signal),
        }
    }
}

pub trait PrescribedQuantity: FiniteQuantity {
    const KIND: PhysicalQuantityKind;

    fn constant_from_signal(signal: &PhysicalSignal) -> Option<&Self>;
}

macro_rules! prescribed_quantity {
    ($quantity:ty, $kind:ident, $variant:ident) => {
        impl PrescribedQuantity for $quantity {
            const KIND: PhysicalQuantityKind = PhysicalQuantityKind::$kind;

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
