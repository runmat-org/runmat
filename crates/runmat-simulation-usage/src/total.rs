use crate::{CountOverflow, OperationCount};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// Exact aggregate across categories or runs. Each category/grant is a u64
/// operation count; aggregates use u128 and retain decimal-string transport.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct OperationTotal(u128);

impl OperationTotal {
    pub const ZERO: Self = Self(0);
    pub const fn from_operations(value: u128) -> Self {
        Self(value)
    }
    pub const fn operations(self) -> u128 {
        self.0
    }
    pub fn checked_add(self, other: Self) -> Result<Self, CountOverflow> {
        self.0.checked_add(other.0).map(Self).ok_or(CountOverflow)
    }
}

impl From<OperationCount> for OperationTotal {
    fn from(count: OperationCount) -> Self {
        Self(u128::from(count.operations()))
    }
}
impl fmt::Display for OperationTotal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl Serialize for OperationTotal {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}
impl<'de> Deserialize<'de> for OperationTotal {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        crate::decimal::deserialize(deserializer, 39).map(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_range_is_distinct_from_one_count_and_never_rounds() {
        let wide = OperationTotal::from(OperationCount::from_operations(u64::MAX))
            .checked_add(OperationTotal::from(OperationCount::from_operations(1)))
            .unwrap();
        assert_eq!(wide.operations(), u128::from(u64::MAX) + 1);
        for total in [wide, OperationTotal::from_operations(u128::MAX)] {
            let wire = serde_json::to_string(&total).unwrap();
            assert_eq!(
                serde_json::from_str::<OperationTotal>(&wire).unwrap(),
                total
            );
            assert!(serde_json::from_str::<OperationCount>(&wire).is_err());
        }
        for invalid in [
            "1",
            "\"01\"",
            "\"+1\"",
            "\"340282366920938463463374607431768211456\"",
        ] {
            assert!(serde_json::from_str::<OperationTotal>(invalid).is_err());
        }
        assert_eq!(
            OperationTotal::from_operations(u128::MAX).checked_add(wide),
            Err(CountOverflow)
        );
    }
}
