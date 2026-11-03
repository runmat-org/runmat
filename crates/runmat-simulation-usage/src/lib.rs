//! Exact numerical-operation units shared by simulation kernels and usage adapters.
//!
//! Kernel owners supply counts from defined reference algorithms. This crate has
//! no account policy, wall-time conversion, implicit counters or allocation hooks.

use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

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
        struct CountVisitor;
        impl de::Visitor<'_> for CountVisitor {
            type Value = OperationCount;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a canonical decimal operation-count string fitting u64")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                if value.is_empty()
                    || value.len() > 20
                    || (value.len() > 1 && value.starts_with('0'))
                    || !value.bytes().all(|byte| byte.is_ascii_digit())
                {
                    return Err(E::custom("invalid decimal operation count"));
                }
                value.parse::<u64>().map(OperationCount).map_err(E::custom)
            }
        }
        deserializer.deserialize_str(CountVisitor)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CountOverflow;

impl fmt::Display for CountOverflow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("simulation operation count exceeds u64")
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
