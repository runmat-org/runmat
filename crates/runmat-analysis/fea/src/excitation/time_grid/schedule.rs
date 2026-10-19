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
        Self::merge_admitted(grid, events, required, maximum, u64::MAX)
    }
    /// Additional requested heap; original source arrays are borrowed and excluded.
    pub(crate) fn construction_resident_bytes(
        grid: &[f64],
        events: &[f64],
        required: &[f64],
        maximum: usize,
    ) -> Result<u64, TimeScheduleError> {
        let (point_count, event_count) = union::counts(grid, events, required, maximum)?;
        union::resident_bytes(point_count, event_count)
    }
    pub(crate) fn merge_admitted(
        grid: &[f64],
        events: &[f64],
        required: &[f64],
        maximum: usize,
        maximum_resident_bytes: u64,
    ) -> Result<Self, TimeScheduleError> {
        let requested = Self::construction_resident_bytes(grid, events, required, maximum)?;
        if requested > maximum_resident_bytes {
            return Err(TimeScheduleError::Invalid(
                "time schedule exceeds resident byte ceiling",
            ));
        }
        let (point_count, event_count) = union::counts(grid, events, required, maximum)?;
        let mut points_s = union::reserve(point_count)?;
        points_s.extend(union::OrderedUnion::new([grid, events, required]));
        let mut events_s = union::reserve(event_count)?;
        events_s.extend(union::OrderedUnion::new([events, &[], &[]]));
        Ok(Self { points_s, events_s })
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
#[cfg(test)]
mod tests;
mod union;
