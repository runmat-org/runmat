use super::super::RuntimeCalendarInstant;
#[cfg(test)]
use chrono::NaiveDate;
use chrono::{Datelike, NaiveDateTime, Timelike};
use jiff::{civil, tz::TimeZone, Timestamp};
#[cfg(test)]
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

    pub(super) fn local_to_instant(&self, local: NaiveDateTime) -> Option<RuntimeCalendarInstant> {
        // Resolve the offset using the ordinary second. Reapply it with
        // Chrono's leap-preserving offset operation, never Duration::new.
        let leap_nanos = local.nanosecond();
        let local = civil::DateTime::new(
            i16::try_from(local.year()).ok()?,
            i8::try_from(local.month()).ok()?,
            i8::try_from(local.day()).ok()?,
            i8::try_from(local.hour()).ok()?,
            i8::try_from(local.minute()).ok()?,
            i8::try_from(local.second()).ok()?,
            i32::try_from(leap_nanos % 1_000_000_000).ok()?,
        )
        .ok()?;
        let timestamp = self.zone.to_ambiguous_timestamp(local).unambiguous().ok()?;
        // Jiff's whole-second accessor truncates toward zero. Chrono pairs
        // floor epoch seconds with positive subsecond (or leap) nanoseconds.
        let seconds = i64::try_from(timestamp.as_nanosecond().div_euclid(1_000_000_000)).ok()?;
        RuntimeCalendarInstant::from_unix_timestamp(seconds, leap_nanos)
    }

    pub(super) fn local_at_instant(
        &self,
        instant: RuntimeCalendarInstant,
    ) -> Option<NaiveDateTime> {
        let timestamp = Timestamp::new(
            instant.unix_seconds(),
            i32::try_from(instant.nanosecond() % 1_000_000_000).ok()?,
        )
        .ok()?;
        instant.at_fixed_offset(self.zone.to_offset(timestamp).seconds())
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn capture_keeps_the_selected_rules_after_the_selector_changes() {
        let selection = RefCell::new("America/New_York");
        let calendar = CapturedCalendar::capture_with(|| TimeZone::get(*selection.borrow()))
            .expect("initial time zone");
        *selection.borrow_mut() = "America/Los_Angeles";

        assert_eq!(
            calendar.local_at_instant(
                RuntimeCalendarInstant::from_epoch_duration(Duration::ZERO).unwrap()
            ),
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn epoch_conversion_follows_rules_across_a_dst_transition() {
        let calendar = named("America/New_York");
        assert_eq!(
            calendar.local_at_instant(
                RuntimeCalendarInstant::from_unix_timestamp(1_710_053_940, 0).unwrap()
            ),
            Some(local(2024, 3, 10, 1, 59))
        );
        assert_eq!(
            calendar.local_at_instant(
                RuntimeCalendarInstant::from_unix_timestamp(1_710_054_000, 0).unwrap()
            ),
            Some(local(2024, 3, 10, 3, 0))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn local_conversion_rejects_dst_gaps_and_folds() {
        let calendar = named("America/New_York");
        assert_eq!(calendar.local_to_instant(local(2024, 3, 10, 2, 30)), None);
        assert_eq!(calendar.local_to_instant(local(2024, 11, 3, 1, 30)), None);
        assert_eq!(
            calendar.local_to_instant(local(2024, 3, 10, 3, 0)),
            Some(RuntimeCalendarInstant::from_unix_timestamp(1_710_054_000, 0).unwrap())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn conversion_preserves_signed_instants_and_rejects_out_of_range_values() {
        let calendar = named("UTC");
        let before = RuntimeCalendarInstant::from_unix_timestamp(-60, 0).unwrap();
        assert_eq!(
            calendar.local_to_instant(local(1969, 12, 31, 23, 59)),
            Some(before)
        );
        assert_eq!(
            calendar.local_at_instant(before),
            Some(local(1969, 12, 31, 23, 59))
        );
        assert!(
            RuntimeCalendarInstant::from_epoch_duration(Duration::from_secs(u64::MAX)).is_none()
        );
        assert_eq!(calendar.local_to_instant(local(10_000, 1, 1, 0, 0)), None);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn checked_boundary_preserves_subsecond_precision() {
        let calendar = named("UTC");
        let local = NaiveDate::from_ymd_opt(2024, 1, 1)
            .expect("test date")
            .and_hms_nano_opt(0, 0, 0, 987_654_321)
            .expect("test time");
        let epoch = calendar
            .local_to_instant(local)
            .expect("representable epoch");
        assert_eq!(
            epoch,
            RuntimeCalendarInstant::from_unix_timestamp(1_704_067_200, 987_654_321).unwrap()
        );
        assert_eq!(calendar.local_at_instant(epoch), Some(local));
    }

    #[test]
    fn captured_rules_preserve_negative_epochs_and_valid_leap_representation() {
        for zone in ["UTC", "America/New_York"] {
            let calendar = named(zone);
            for instant in [
                RuntimeCalendarInstant::from_unix_timestamp(-60, 123).unwrap(),
                RuntimeCalendarInstant::from_unix_timestamp(1_483_228_799, 1_500_000_000).unwrap(),
            ] {
                let local = calendar.local_at_instant(instant).unwrap();
                assert_eq!(local.nanosecond(), instant.nanosecond());
                assert_eq!(calendar.local_to_instant(local), Some(instant));
            }
        }
    }
}
