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
    fn is_nan(self) -> bool;
    fn is_infinite(self) -> bool;
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
            fn is_nan(self) -> bool {
                self.is_nan()
            }
            fn is_infinite(self) -> bool {
                self.is_infinite()
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

/// The row ordering shared by recurrent kernels and per-gate training factors.
/// Each direction contains four contiguous hidden-unit blocks in gate order.
#[derive(Clone, Copy, Debug)]
pub(in crate::builtins::deep_learning) struct RecurrentGateRows {
    rows: usize,
    hidden_size: usize,
}

impl RecurrentGateRows {
    pub(in crate::builtins::deep_learning) fn for_layer(
        rows: usize,
        kind: runmat_builtins::DeepLearningLayerKind,
    ) -> Option<Self> {
        let blocks = kind.recurrent_gate_count()?;
        let hidden_size = rows.checked_div(blocks)?;
        (hidden_size > 0 && hidden_size.checked_mul(blocks) == Some(rows))
            .then_some(Self { rows, hidden_size })
    }

    /// `index` is a column-major element index in any recurrent parameter.
    pub(in crate::builtins::deep_learning) fn factor_index(self, index: usize) -> usize {
        (index % self.rows) / self.hidden_size
    }

    pub(in crate::builtins::deep_learning) fn factor_count(self) -> usize {
        self.rows / self.hidden_size
    }
}

pub(super) fn state<T: Real>(kind: Activation, value: T) -> T {
    match kind {
        Activation::Tanh => value.tanh(),
        Activation::Softsign => {
            if value.is_infinite() {
                if value > T::zero() {
                    T::one()
                } else {
                    -T::one()
                }
            } else {
                value / (T::one() + value.abs())
            }
        }
        Activation::Relu => {
            if value.is_nan() || value > T::zero() {
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
            let reciprocal = T::one() / (T::one() + value.abs());
            reciprocal * reciprocal
        }
        Activation::Relu => {
            if value.is_nan() {
                value
            } else if value > T::zero() {
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
            if preactivation.is_nan() {
                preactivation
            } else if preactivation > T::from_f64(-2.5) && preactivation < T::from_f64(2.5) {
                T::from_f64(0.2)
            } else {
                T::zero()
            }
        }
    }
}
