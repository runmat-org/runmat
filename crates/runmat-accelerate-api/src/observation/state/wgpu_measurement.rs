//! Optional physical WGPU evidence owned by one operation's completed tokens.
//! A missing/overflowed metric stays unknown without failing required work.

use super::{ProviderObservationStatus, WgpuQueryAllocationObservation, WgpuSubmissionMeasurement};
use std::time::Duration;

#[derive(Default)]
pub(super) struct WgpuMeasurementSummary {
    pub(super) compute_elapsed: Option<Duration>,
    pub(super) dispatches: Option<u64>,
    pub(super) query_allocations: Option<WgpuQueryAllocationObservation>,
}

pub(super) struct WgpuMeasurementAggregate {
    settled: Option<u64>,
    compute_elapsed: Option<Duration>,
    dispatches: Option<u64>,
    query_allocations: Option<WgpuQueryAllocationObservation>,
}

impl Default for WgpuMeasurementAggregate {
    fn default() -> Self {
        Self {
            settled: Some(0),
            compute_elapsed: Some(Duration::ZERO),
            dispatches: Some(0),
            query_allocations: Some(WgpuQueryAllocationObservation::default()),
        }
    }
}

impl WgpuMeasurementAggregate {
    pub(super) fn record(&mut self, measurement: Option<WgpuSubmissionMeasurement>) {
        self.settled = self.settled.and_then(|count| count.checked_add(1));
        self.compute_elapsed = self
            .compute_elapsed
            .zip(measurement.and_then(|sample| sample.compute_elapsed))
            .and_then(|(sum, elapsed)| sum.checked_add(elapsed));
        self.dispatches = self
            .dispatches
            .zip(measurement.and_then(|sample| sample.dispatches))
            .and_then(|(sum, count)| sum.checked_add(count));
        self.query_allocations = self
            .query_allocations
            .zip(measurement.and_then(|sample| sample.query_allocation))
            .and_then(|(sum, disposition)| sum.checked_record(disposition));
    }

    pub(super) fn snapshot(
        &self,
        registered: u64,
        status: ProviderObservationStatus,
    ) -> WgpuMeasurementSummary {
        if registered == 0
            || self.settled != Some(registered)
            || !matches!(status, ProviderObservationStatus::Completed { .. })
        {
            return WgpuMeasurementSummary::default();
        }
        WgpuMeasurementSummary {
            compute_elapsed: self.compute_elapsed,
            dispatches: self.dispatches,
            query_allocations: self.query_allocations,
        }
    }
}

#[cfg(test)]
#[path = "wgpu_measurement/tests.rs"]
mod tests;
