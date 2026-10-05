//! Bounded ordered time union shared by physical transient procedures.
use runmat_analysis_core::SignalSampleApproach;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ProcedureTimeSchedule {
    points_s: Vec<f64>,
    events_s: Vec<f64>,
}
#[derive(Debug)]
pub(crate) enum TimeScheduleError {
    Invalid(&'static str),
    Ceiling { required: usize, maximum: usize },
}
impl ProcedureTimeSchedule {
    pub(crate) fn merge(
        grid: &[f64],
        events: &[f64],
        required: &[f64],
        maximum: usize,
    ) -> Result<Self, TimeScheduleError> {
        if grid.len() < 2
            || grid.iter().any(|t| !t.is_finite())
            || grid.windows(2).any(|p| p[0] >= p[1])
        {
            return Err(TimeScheduleError::Invalid(
                "time grid must be finite and strictly increasing",
            ));
        }
        check_count(grid.len(), maximum)?;
        let start = grid[0];
        let end = grid[grid.len() - 1];
        let canonical = |v: f64| if v == 0.0 { 0.0 } else { v };
        let mut points = grid.iter().copied().map(canonical).collect::<Vec<_>>();
        for source in [events, required] {
            if source
                .iter()
                .any(|t| !t.is_finite() || *t < start || *t > end)
                || source.windows(2).any(|p| p[0] > p[1])
            {
                return Err(TimeScheduleError::Invalid(
                    "time schedule source must be ordered inside the authored interval",
                ));
            }
            let mut left = points.into_iter().peekable();
            let mut right = source.iter().copied().map(canonical).peekable();
            let mut merged = Vec::new();
            while left.peek().is_some() || right.peek().is_some() {
                let value = match (left.peek(), right.peek()) {
                    (Some(a), Some(b)) if a <= b => left.next().unwrap(),
                    (_, Some(_)) => right.next().unwrap(),
                    _ => left.next().unwrap(),
                };
                if merged.last().is_none_or(|last| *last != value) {
                    check_count(
                        merged
                            .len()
                            .checked_add(1)
                            .ok_or(TimeScheduleError::Invalid("time schedule count overflow"))?,
                        maximum,
                    )?;
                    merged.push(value);
                }
            }
            points = merged;
        }
        let mut events_s = events.iter().copied().map(canonical).collect::<Vec<_>>();
        events_s.dedup();
        Ok(Self {
            points_s: points,
            events_s,
        })
    }
    pub(crate) fn points_s(&self) -> &[f64] {
        &self.points_s
    }
    pub(crate) fn arrival(&self, ordinal: usize) -> SignalSampleApproach {
        if self
            .events_s
            .binary_search_by(|e| e.total_cmp(&self.points_s[ordinal]))
            .is_ok()
        {
            SignalSampleApproach::LeftLimit
        } else {
            SignalSampleApproach::At
        }
    }
    pub(crate) fn departure(&self, ordinal: usize) -> SignalSampleApproach {
        if self.arrival(ordinal) == SignalSampleApproach::LeftLimit {
            SignalSampleApproach::RightLimit
        } else {
            SignalSampleApproach::At
        }
    }
}
fn check_count(required: usize, maximum: usize) -> Result<(), TimeScheduleError> {
    if required > maximum {
        Err(TimeScheduleError::Ceiling { required, maximum })
    } else {
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordered_union_preserves_arrival_departure_and_unique_ceiling() {
        let s =
            ProcedureTimeSchedule::merge(&[0., 0.5, 1.], &[0.2, 0.5], &[0.2, 0.65, 1.], 5).unwrap();
        assert_eq!(s.points_s(), &[0., 0.2, 0.5, 0.65, 1.]);
        assert_eq!(s.arrival(1), SignalSampleApproach::LeftLimit);
        assert_eq!(s.departure(1), SignalSampleApproach::RightLimit);
        assert_eq!(s.departure(3), SignalSampleApproach::At);
        assert!(matches!(
            ProcedureTimeSchedule::merge(&[0., 1.], &[], &[0.2, 0.3], 3),
            Err(TimeScheduleError::Ceiling {
                required: 4,
                maximum: 3
            })
        ));
    }
    #[test]
    fn invalid_or_unsorted_sources_cannot_create_negative_intervals() {
        for source in [
            &[0.7, 0.2][..],
            &[f64::NAN],
            &[f64::INFINITY],
            &[-0.1],
            &[1.1],
        ] {
            assert!(ProcedureTimeSchedule::merge(&[0., 1.], source, &[], 9).is_err());
            assert!(ProcedureTimeSchedule::merge(&[0., 1.], &[], source, 9).is_err());
        }
        for grid in [&[][..], &[0.], &[1., 0.], &[0., 0.]] {
            assert!(ProcedureTimeSchedule::merge(grid, &[], &[], 9).is_err());
        }
        let s = ProcedureTimeSchedule::merge(&[-0., 1.], &[0., 0.], &[-0.], 2).unwrap();
        assert_eq!(s.points_s()[0].to_bits(), 0_f64.to_bits());
    }
}
