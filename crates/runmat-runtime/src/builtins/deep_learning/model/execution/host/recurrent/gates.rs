use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use super::geometry::{Activation, GateActivation};

pub(super) trait Real:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(value: f64) -> Self;
    fn exp(self) -> Self;
    fn tanh(self) -> Self;
    fn abs(self) -> Self;
}

macro_rules! impl_real {
    ($type:ty) => {
        impl Real for $type {
            fn zero() -> Self {
                0.0
            }
            fn one() -> Self {
                1.0
            }
            fn from_f64(value: f64) -> Self {
                value as Self
            }
            fn exp(self) -> Self {
                self.exp()
            }
            fn tanh(self) -> Self {
                self.tanh()
            }
            fn abs(self) -> Self {
                self.abs()
            }
        }
    };
}

impl_real!(f32);
impl_real!(f64);

#[derive(Clone, Copy)]
pub(super) enum Gate {
    Input,
    Forget,
    Candidate,
    Output,
}

impl Gate {
    pub fn index(self) -> usize {
        match self {
            Self::Input => 0,
            Self::Forget => 1,
            Self::Candidate => 2,
            Self::Output => 3,
        }
    }
}

pub(super) fn gate_row(direction: usize, gate: Gate, unit: usize, hidden_size: usize) -> usize {
    direction * 4 * hidden_size + gate.index() * hidden_size + unit
}

pub(super) fn state<T: Real>(kind: Activation, value: T) -> T {
    match kind {
        Activation::Tanh => value.tanh(),
        Activation::Softsign => value / (T::one() + value.abs()),
        Activation::Relu => {
            if value > T::zero() {
                value
            } else {
                T::zero()
            }
        }
    }
}

pub(super) fn state_derivative<T: Real>(kind: Activation, value: T, activated: T) -> T {
    match kind {
        Activation::Tanh => T::one() - activated * activated,
        Activation::Softsign => {
            let denom = T::one() + value.abs();
            T::one() / (denom * denom)
        }
        Activation::Relu => {
            if value > T::zero() {
                T::one()
            } else {
                T::zero()
            }
        }
    }
}

pub(super) fn gate<T: Real>(kind: GateActivation, value: T) -> T {
    match kind {
        GateActivation::Sigmoid => {
            // Avoid exp overflow for large negative preactivations.
            if value >= T::zero() {
                T::one() / (T::one() + (-value).exp())
            } else {
                let e = value.exp();
                e / (T::one() + e)
            }
        }
        GateActivation::HardSigmoid => {
            let result = T::from_f64(0.2) * value + T::from_f64(0.5);
            if result < T::zero() {
                T::zero()
            } else if result > T::one() {
                T::one()
            } else {
                result
            }
        }
    }
}

pub(super) fn gate_derivative<T: Real>(kind: GateActivation, preactivation: T, activated: T) -> T {
    match kind {
        GateActivation::Sigmoid => activated * (T::one() - activated),
        GateActivation::HardSigmoid => {
            if preactivation > T::from_f64(-2.5) && preactivation < T::from_f64(2.5) {
                T::from_f64(0.2)
            } else {
                T::zero()
            }
        }
    }
}
