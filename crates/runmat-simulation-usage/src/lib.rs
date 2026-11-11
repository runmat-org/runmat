//! Exact numerical-operation units shared by simulation kernels and usage adapters.
//!
//! Kernel owners supply counts from defined reference algorithms. This crate has
//! no account policy, wall-time conversion, implicit counters or allocation hooks.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

mod ledger;
mod usage;
mod decimal;
mod total;
mod measurement;
mod context;
mod async_capacity;
pub mod outage;
pub use async_capacity::AsyncCapacityOwner;
pub use context::{UsageContext, UsageContextGuard};
pub use ledger::{
    grant_target, DelegatedReservation, GrantRenewal, LedgerError, LedgerSnapshot, UsageLedger,
    WorkReservation,
};
pub use measurement::{InvalidMeasurementVersion, MeasurementVersion};
pub use total::OperationTotal;
pub use usage::{CompletedUsage, UsageCategory};

/// An exact nonnegative number of reference floating-point operations.
///
/// The wire representation is a canonical decimal string, including for small
/// counts. JavaScript clients can use `BigInt` without rounding values above 2^53.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct OperationCount(u64);

impl OperationCount {
    pub const ZERO: Self = Self(0);
    pub const PER_GFLOP: u64 = 1_000_000_000;

    pub const fn from_operations(operations: u64) -> Self {
        Self(operations)
    }

    pub const fn operations(self) -> u64 {
        self.0
    }

    pub fn checked_add(self, other: Self) -> Result<Self, CountOverflow> {
        self.0.checked_add(other.0).map(Self).ok_or(CountOverflow)
    }

    pub fn checked_mul(self, repetitions: u64) -> Result<Self, CountOverflow> {
        self.0
            .checked_mul(repetitions)
            .map(Self)
            .ok_or(CountOverflow)
    }

    /// Direct real dot-product schedule: n products and n−1 additions.
    ///
    /// This formula describes that reference algorithm, not compensated sums,
    /// complex products or a physical instruction counter.
    pub fn direct_real_dot(length: u64) -> Result<Self, CountOverflow> {
        if length == 0 {
            return Ok(Self::ZERO);
        }
        length
            .checked_add(length - 1)
            .map(Self)
            .ok_or(CountOverflow)
    }
}

impl fmt::Display for OperationCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Serialize for OperationCount {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for OperationCount {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        decimal::deserialize(deserializer, 20).map(Self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CountOverflow;

impl fmt::Display for CountOverflow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("simulation operation count exceeds its integer range")
    }
}

impl std::error::Error for CountOverflow {}

/// Basic real arithmetic under the simulation counting convention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RealArithmetic {
    Add,
    Subtract,
    Multiply,
    Divide,
    FusedMultiplyAdd,
}

impl RealArithmetic {
    /// Count active numerical elements, independently of storage precision,
    /// vector width, workgroup padding or the backend's instruction selection.
    pub fn count(self, active_elements: u64) -> Result<OperationCount, CountOverflow> {
        let per_element = match self {
            Self::Add | Self::Subtract | Self::Multiply | Self::Divide => 1,
            Self::FusedMultiplyAdd => 2,
        };
        OperationCount::from_operations(per_element).checked_mul(active_elements)
    }
}

#[cfg(test)]
mod tests;
