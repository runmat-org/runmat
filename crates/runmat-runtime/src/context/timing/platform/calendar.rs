use chrono::{Datelike, NaiveDate, NaiveDateTime, Timelike};
use jiff::{civil, tz::TimeZone, Timestamp};
use std::time::Duration;

/// One immutable set of host-local calendar rules captured for a timing
/// service. The zone is never rediscovered while the service is alive.
pub(super) struct CapturedCalendar {
    zone: TimeZone,
}

impl CapturedCalendar {
    pub(super) fn try_system() -> Result<Self, jiff::Error> {
        Self::capture_with(TimeZone::try_system)
    }

    fn capture_with<E>(capture: impl FnOnce() -> Result<TimeZone, E>) -> Result<Self, E> {
        capture().map(|zone| Self { zone })
    }

    pub(super) fn local_to_epoch(&self, local: NaiveDateTime) -> Option<Duration> {
        let local = civil::DateTime::new(
            i16::try_from(local.year()).ok()?,
            i8::try_from(local.month()).ok()?,
            i8::try_from(local.day()).ok()?,
            i8::try_from(local.hour()).ok()?,
            i8::try_from(local.minute()).ok()?,
            i8::try_from(local.second()).ok()?,
            i32::try_from(local.nanosecond()).ok()?,
        )
        .ok()?;
        let timestamp = self.zone.to_ambiguous_timestamp(local).unambiguous().ok()?;
        Some(Duration::new(
            u64::try_from(timestamp.as_second()).ok()?,
            u32::try_from(timestamp.subsec_nanosecond()).ok()?,
        ))
    }

    pub(super) fn local_at_epoch(&self, epoch: Duration) -> Option<NaiveDateTime> {
        let timestamp = Timestamp::new(
            i64::try_from(epoch.as_secs()).ok()?,
            i32::try_from(epoch.subsec_nanos()).ok()?,
        )
        .ok()?;
        let local = self.zone.to_datetime(timestamp);
        NaiveDate::from_ymd_opt(
            i32::from(local.year()),
            u32::try_from(local.month()).ok()?,
            u32::try_from(local.day()).ok()?,
        )?
        .and_hms_nano_opt(
            u32::try_from(local.hour()).ok()?,
            u32::try_from(local.minute()).ok()?,
            u32::try_from(local.second()).ok()?,
            u32::try_from(local.subsec_nanosecond()).ok()?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn named(name: &str) -> CapturedCalendar {
        CapturedCalendar::capture_with(|| TimeZone::get(name)).expect("test time zone")
    }

    fn local(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(year, month, day)
            .expect("test date")
            .and_hms_opt(hour, minute, 0)
            .expect("test time")
    }

    #[test]
    fn capture_keeps_the_selected_rules_after_the_selector_changes() {
        let selection = RefCell::new("America/New_York");
        let calendar = CapturedCalendar::capture_with(|| TimeZone::get(*selection.borrow()))
            .expect("initial time zone");
        *selection.borrow_mut() = "America/Los_Angeles";

        assert_eq!(
            calendar.local_at_epoch(Duration::ZERO),
            Some(local(1969, 12, 31, 19, 0))
        );
        assert_eq!(
            TimeZone::get(*selection.borrow())
                .expect("updated time zone")
                .to_datetime(Timestamp::UNIX_EPOCH)
                .hour(),
            16
        );
    }

    #[test]
    fn epoch_conversion_follows_rules_across_a_dst_transition() {
        let calendar = named("America/New_York");
        assert_eq!(
            calendar.local_at_epoch(Duration::from_secs(1_710_053_940)),
            Some(local(2024, 3, 10, 1, 59))
        );
        assert_eq!(
            calendar.local_at_epoch(Duration::from_secs(1_710_054_000)),
            Some(local(2024, 3, 10, 3, 0))
        );
    }

    #[test]
    fn local_conversion_rejects_dst_gaps_and_folds() {
        let calendar = named("America/New_York");
        assert_eq!(calendar.local_to_epoch(local(2024, 3, 10, 2, 30)), None);
        assert_eq!(calendar.local_to_epoch(local(2024, 11, 3, 1, 30)), None);
        assert_eq!(
            calendar.local_to_epoch(local(2024, 3, 10, 3, 0)),
            Some(Duration::from_secs(1_710_054_000))
        );
    }

    #[test]
    fn conversion_rejects_pre_epoch_and_out_of_range_values() {
        let calendar = named("UTC");
        assert_eq!(calendar.local_to_epoch(local(1969, 12, 31, 23, 59)), None);
        assert_eq!(calendar.local_at_epoch(Duration::from_secs(u64::MAX)), None);
        assert_eq!(calendar.local_to_epoch(local(10_000, 1, 1, 0, 0)), None);
    }

    #[test]
    fn checked_boundary_preserves_subsecond_precision() {
        let calendar = named("UTC");
        let local = NaiveDate::from_ymd_opt(2024, 1, 1)
            .expect("test date")
            .and_hms_nano_opt(0, 0, 0, 987_654_321)
            .expect("test time");
        let epoch = calendar.local_to_epoch(local).expect("representable epoch");
        assert_eq!(epoch, Duration::new(1_704_067_200, 987_654_321));
        assert_eq!(calendar.local_at_epoch(epoch), Some(local));
    }
}
