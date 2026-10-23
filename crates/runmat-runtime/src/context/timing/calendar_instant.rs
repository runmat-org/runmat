//! Absolute calendar identity, deliberately separate from elapsed durations.
use chrono::{DateTime, FixedOffset, NaiveDateTime, Timelike, Utc};
use std::time::Duration;

/// Signed Unix seconds plus Chrono's validated leap-second representation.
/// This is a calendar instant, not a monotonic interval or a sleep duration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeCalendarInstant(DateTime<Utc>);

impl RuntimeCalendarInstant {
    pub fn from_unix_timestamp(seconds: i64, nanosecond: u32) -> Option<Self> {
        DateTime::from_timestamp(seconds, nanosecond).map(Self)
    }

    pub fn from_epoch_duration(epoch: Duration) -> Option<Self> {
        Self::from_unix_timestamp(i64::try_from(epoch.as_secs()).ok()?, epoch.subsec_nanos())
    }

    pub fn from_naive_utc(utc: NaiveDateTime) -> Self {
        Self(utc.and_utc())
    }

    /// Scheduling's nonnegative epoch domain cannot encode a leap second.
    pub fn to_epoch_duration(self) -> Option<Duration> {
        if self.nanosecond() >= 1_000_000_000 {
            return None;
        }
        Some(Duration::new(
            u64::try_from(self.unix_seconds()).ok()?,
            self.nanosecond(),
        ))
    }

    pub fn unix_seconds(self) -> i64 {
        self.0.timestamp()
    }
    pub fn nanosecond(self) -> u32 {
        self.0.nanosecond()
    }

    pub fn at_fixed_offset(self, seconds: i32) -> Option<NaiveDateTime> {
        self.0
            .naive_utc()
            .checked_add_offset(FixedOffset::east_opt(seconds)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calendar_identity_does_not_normalize_signed_or_leap_instants_into_durations() {
        assert!(RuntimeCalendarInstant::from_unix_timestamp(-1, 0)
            .unwrap()
            .to_epoch_duration()
            .is_none());
        let leap =
            RuntimeCalendarInstant::from_unix_timestamp(1_483_228_799, 1_500_000_000).unwrap();
        assert_eq!(leap.nanosecond(), 1_500_000_000);
        assert!(leap.to_epoch_duration().is_none());
        assert!(RuntimeCalendarInstant::from_unix_timestamp(0, 1_500_000_000).is_none());
    }
}
