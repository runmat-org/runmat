//! One encoder's compute-pass evidence, attached transactionally to submission.
//! No device queries or accounting state are allocated without an active owner.

use runmat_accelerate_api::observation::{
    ProviderObservationBudget, WgpuQueryAllocationDisposition, WgpuQueryPermit,
};
#[cfg(not(target_arch = "wasm32"))]
use runmat_accelerate_api::observation::{
    WgpuObservationProgress, WgpuObservationProgressReservation,
};
use runmat_accelerate_api::{WgpuBufferOwner, WgpuEmission};
use runmat_execution::profiling::CommandProfilingReceipt;
use std::sync::Arc;
use std::time::Duration;

use super::admission::ProviderAdmission;

const MAX_QUERY_PAIRS_PER_ENCODER: u32 = 8;
pub(crate) const MAX_TIMESTAMP_SLOTS_PER_ENCODER: u32 = 2 * (MAX_QUERY_PAIRS_PER_ENCODER + 1);
const BYTES_PER_PAIR: u64 = 2 * std::mem::size_of::<u64>() as u64;
/// Maximum resolve plus staging request held by one admitted query permit.
pub(crate) const MAX_QUERY_REQUESTED_BYTES: u64 =
    2 * (MAX_QUERY_PAIRS_PER_ENCODER as u64 + 1) * BYTES_PER_PAIR;
pub(crate) const MAX_COMPUTE_QUERY_REQUESTED_BYTES: u64 =
    2 * MAX_QUERY_PAIRS_PER_ENCODER as u64 * BYTES_PER_PAIR;

#[derive(Clone, Copy)]
pub(crate) struct TimestampCapability {
    period_ns: f32,
    inside_encoders: bool,
}

impl TimestampCapability {
    pub(super) fn period_ns(self) -> f32 {
        self.period_ns
    }
    pub(crate) fn new(period_ns: f32, inside_encoders: bool) -> Option<Self> {
        (period_ns.is_finite() && period_ns > 0.0).then_some(Self {
            period_ns,
            inside_encoders,
        })
    }

    pub(crate) fn inside_encoders(self) -> bool {
        self.inside_encoders
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn without_encoder_timestamps_for_test(self) -> Self {
        Self {
            inside_encoders: false,
            ..self
        }
    }
}

pub(crate) enum ObservationBuffer {
    Ordinary(WgpuBufferOwner),
    Readback(runmat_accelerate_api::WgpuNumericReadbackBuffer),
}
impl std::ops::Deref for ObservationBuffer {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Ordinary(buffer) => buffer,
            Self::Readback(buffer) => std::borrow::Borrow::borrow(buffer),
        }
    }
}
impl ObservationBuffer {
    pub(super) fn map_async(
        &self,
        callback: impl FnOnce(Result<(), wgpu::BufferAsyncError>) + wgpu::WasmNotSend + 'static,
    ) {
        match self {
            Self::Ordinary(buffer) => buffer.slice(..).map_async(wgpu::MapMode::Read, callback),
            Self::Readback(buffer) => {
                // Rejection drops the status-only sender; the shared completion
                // owner still retires exact submitted work as failed/missing.
                if buffer.map_async(callback).is_err() {
                    log::warn!("numeric readback completion map authorization failed");
                }
            }
        }
    }
}

pub(crate) struct TimestampReadback {
    pub(crate) staging: ObservationBuffer,
    // Keep the complete device resource ownership until map/progress cleanup.
    _resolve: ObservationBuffer,
    _query_set: wgpu::QuerySet,
    _permit: WgpuQueryPermit,
    query_pairs: u32,
    outer_pair: bool,
    compute_complete: bool,
    period_ns: f32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DecodedTimestamps {
    pub(crate) device_elapsed: Option<Duration>,
    pub(crate) compute_elapsed: Option<Duration>,
}

impl TimestampReadback {
    pub(super) fn numeric(
        resources: runmat_accelerate_api::WgpuNumericReadbackTimestampResources,
        permit: WgpuQueryPermit,
        timestamp: TimestampCapability,
    ) -> Self {
        Self {
            staging: ObservationBuffer::Readback(resources.staging),
            _resolve: ObservationBuffer::Readback(resources.resolve),
            _query_set: resources.query_set,
            _permit: permit,
            query_pairs: 0,
            outer_pair: true,
            compute_complete: true,
            period_ns: timestamp.period_ns(),
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn poison_query_budget(&self) {
        self._permit.poison_budget();
    }

    /// Called under the existing device cleanup authority only after mapping.
    pub(crate) fn decode_mapped(&self) -> DecodedTimestamps {
        let bytes = self.staging.slice(..).get_mapped_range();
        decode_timestamp_intervals(
            &bytes,
            self.outer_pair,
            self.query_pairs,
            self.compute_complete,
            self.period_ns,
        )
    }
}

fn decode_timestamp_intervals(
    bytes: &[u8],
    outer_pair: bool,
    compute_pairs: u32,
    compute_complete: bool,
    period_ns: f32,
) -> DecodedTimestamps {
    let pair_bytes = BYTES_PER_PAIR as usize;
    let expected = usize::try_from(compute_pairs)
        .ok()
        .and_then(|pairs| pairs.checked_add(usize::from(outer_pair)))
        .and_then(|pairs| pairs.checked_mul(pair_bytes));
    if expected != Some(bytes.len()) {
        return DecodedTimestamps::default();
    }
    let device_elapsed = outer_pair
        .then(|| decode_timestamp_pairs(&bytes[..pair_bytes], 1, period_ns))
        .flatten();
    let compute_bytes = if outer_pair {
        &bytes[pair_bytes..]
    } else {
        bytes
    };
    let compute_elapsed = compute_complete
        .then(|| decode_timestamp_pairs(compute_bytes, compute_pairs, period_ns))
        .flatten();
    DecodedTimestamps {
        device_elapsed,
        compute_elapsed,
    }
}

fn decode_timestamp_pairs(bytes: &[u8], query_pairs: u32, period_ns: f32) -> Option<Duration> {
    if !period_ns.is_finite() || period_ns <= 0.0 {
        return None;
    }
    let expected_bytes = usize::try_from(query_pairs)
        .ok()?
        .checked_mul(BYTES_PER_PAIR as usize)?;
    if bytes.len() != expected_bytes {
        return None;
    }
    let mut elapsed = Duration::ZERO;
    let mut pairs = bytes.chunks_exact(BYTES_PER_PAIR as usize);
    for pair in pairs.by_ref() {
        let begin = u64::from_le_bytes(pair[0..8].try_into().ok()?);
        let end = u64::from_le_bytes(pair[8..16].try_into().ok()?);
        let ticks = end.checked_sub(begin)?;
        let nanos = (ticks as f64) * f64::from(period_ns);
        // f64 rounds u64::MAX upward, so the exclusive 2^64 bound is
        // required before integer conversion (which otherwise saturates).
        if !nanos.is_finite() || nanos < 0.0 || nanos.round() >= 18_446_744_073_709_551_616.0 {
            return None;
        }
        elapsed = elapsed.checked_add(Duration::from_nanos(nanos.round() as u64))?;
    }
    pairs.remainder().is_empty().then_some(elapsed)
}

pub(crate) struct ProfiledEncoder<'a> {
    device: &'a wgpu::Device,
    encoder: wgpu::CommandEncoder,
    dispatches: Option<u64>,
    pass_count: u32,
    query_pairs: u32,
    outer_requested: bool,
    outer_pair: bool,
    query_set: Option<wgpu::QuerySet>,
    query_permit: Option<WgpuQueryPermit>,
    timestamp: Option<TimestampCapability>,
    budget: Option<Arc<ProviderObservationBudget>>,
    #[cfg(not(target_arch = "wasm32"))]
    progress_port: Option<Arc<dyn WgpuObservationProgress>>,
    #[cfg(not(target_arch = "wasm32"))]
    provider_progress: bool,
    #[cfg(not(target_arch = "wasm32"))]
    progress: Option<Box<dyn WgpuObservationProgressReservation>>,
    timing_complete: bool,
    caller_timestamp: bool,
    admission: ProviderAdmission,
    command: Option<CommandProfilingReceipt>,
}

pub(crate) struct EncodedMeasurement<E = wgpu::CommandEncoder> {
    pub(crate) encoder: E,
    pub(crate) dispatches: Option<u64>,
    pub(crate) empty_compute_elapsed: Option<Duration>,
    pub(crate) empty_device_elapsed: Option<Duration>,
    pub(crate) readback: Option<TimestampReadback>,
    pub(crate) query_allocation: Option<WgpuQueryAllocationDisposition>,
    pub(super) admission: ProviderAdmission,
    pub(super) command: Option<CommandProfilingReceipt>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) progress: Option<Box<dyn WgpuObservationProgressReservation>>,
}

/// The only submission ingress. Requiring this carrier at compile time keeps
/// future raw compute passes from silently escaping the physical-pass census.
pub(crate) trait SubmissionEncoder {
    fn into_measurement(self) -> EncodedMeasurement;
}

impl SubmissionEncoder for ProfiledEncoder<'_> {
    fn into_measurement(self) -> EncodedMeasurement {
        self.into_submission()
    }
}

/// Captured observation ownership and optional instrumentation for one encoder.
pub(super) struct EncoderObservation {
    pub(super) command: Option<CommandProfilingReceipt>,
    pub(super) admission: ProviderAdmission,
    pub(super) timestamp: Option<TimestampCapability>,
    pub(super) budget: Option<Arc<ProviderObservationBudget>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) progress_port: Option<Arc<dyn WgpuObservationProgress>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) completion_prepared: bool,
}

impl<'a> ProfiledEncoder<'a> {
    pub(super) fn new(
        emission: &'a WgpuEmission<'_>,
        descriptor: &wgpu::CommandEncoderDescriptor<'_>,
        observation: EncoderObservation,
    ) -> Self {
        let EncoderObservation {
            command,
            admission,
            timestamp,
            budget,
            #[cfg(not(target_arch = "wasm32"))]
            progress_port,
            #[cfg(not(target_arch = "wasm32"))]
            completion_prepared,
        } = observation;
        let active = !matches!(&admission, ProviderAdmission::Absent) || command.is_some();
        #[cfg(not(target_arch = "wasm32"))]
        let provider_progress =
            matches!(&admission, ProviderAdmission::Prepared(_)) || completion_prepared;
        let mut profiled = Self {
            device: emission.device(),
            encoder: emission.device().create_command_encoder(descriptor),
            dispatches: active.then_some(0),
            pass_count: 0,
            query_pairs: 0,
            outer_requested: command.is_some()
                && timestamp.is_some_and(|capability| capability.inside_encoders),
            outer_pair: false,
            query_set: None,
            query_permit: None,
            timestamp,
            budget,
            #[cfg(not(target_arch = "wasm32"))]
            progress_port,
            #[cfg(not(target_arch = "wasm32"))]
            provider_progress,
            #[cfg(not(target_arch = "wasm32"))]
            progress: None,
            timing_complete: active,
            caller_timestamp: false,
            admission,
            command,
        };
        if profiled.outer_requested {
            profiled.admit_query();
            if let Some(query_set) = &profiled.query_set {
                profiled.encoder.write_timestamp(query_set, 0);
                profiled.outer_pair = true;
            }
        }
        profiled
    }

    fn admit_query(&mut self) {
        if self.query_set.is_some() || !self.timing_complete || self.timestamp.is_none() {
            return;
        }
        let requested_bytes = if self.outer_requested {
            MAX_QUERY_REQUESTED_BYTES
        } else {
            MAX_COMPUTE_QUERY_REQUESTED_BYTES
        };
        let Some(permit) = self
            .budget
            .as_ref()
            .and_then(|budget| budget.try_reserve_wgpu_query(requested_bytes))
        else {
            self.timing_complete = false;
            return;
        };
        #[cfg(not(target_arch = "wasm32"))]
        if !self.provider_progress {
            let Some(progress) = self
                .progress_port
                .as_ref()
                .and_then(|port| port.reserve().ok())
            else {
                self.timing_complete = false;
                return;
            };
            self.progress = Some(progress);
        }
        self.query_set = Some(self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("runmat-compute-pass-timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: if self.outer_requested {
                MAX_TIMESTAMP_SLOTS_PER_ENCODER
            } else {
                MAX_TIMESTAMP_SLOTS_PER_ENCODER - 2
            },
        }));
        self.query_permit = Some(permit);
    }

    pub(crate) fn begin_compute_pass(
        &mut self,
        descriptor: &wgpu::ComputePassDescriptor<'_>,
    ) -> ProfiledComputePass<'_> {
        // A caller-owned query is not our evidence. Preserve its descriptor
        // unchanged and leave the operation's device-time field incomplete.
        if descriptor.timestamp_writes.is_some() {
            self.timing_complete = false;
            self.caller_timestamp = true;
        }
        self.pass_count = match self.pass_count.checked_add(1) {
            Some(count) => count,
            None => {
                self.timing_complete = false;
                u32::MAX
            }
        };
        if self.query_pairs == MAX_QUERY_PAIRS_PER_ENCODER {
            self.timing_complete = false;
        }
        self.admit_query();
        let pair = if self.timing_complete && self.query_set.is_some() {
            let pair = self.query_pairs;
            self.query_pairs += 1;
            Some(pair)
        } else {
            None
        };
        let timestamp_writes = descriptor.timestamp_writes.clone().or_else(|| {
            pair.and_then(|pair| {
                self.query_set
                    .as_ref()
                    .map(|query_set| wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(
                            2 * (pair + u32::from(self.outer_pair)),
                        ),
                        end_of_pass_write_index: Some(2 * (pair + u32::from(self.outer_pair)) + 1),
                    })
            })
        });
        let descriptor = wgpu::ComputePassDescriptor {
            label: descriptor.label,
            timestamp_writes,
        };
        let pass = self.encoder.begin_compute_pass(&descriptor);
        ProfiledComputePass {
            pass,
            dispatches: &mut self.dispatches,
        }
    }

    pub(crate) fn copy_buffer_to_buffer(
        &mut self,
        source: &wgpu::Buffer,
        source_offset: u64,
        destination: &wgpu::Buffer,
        destination_offset: u64,
        size: u64,
    ) {
        self.encoder.copy_buffer_to_buffer(
            source,
            source_offset,
            destination,
            destination_offset,
            size,
        );
    }

    pub(crate) fn clear_buffer(&mut self, buffer: &wgpu::Buffer, offset: u64, size: Option<u64>) {
        self.encoder.clear_buffer(buffer, offset, size);
    }

    pub(crate) fn into_submission(mut self) -> EncodedMeasurement {
        if self.outer_pair {
            if let Some(query_set) = &self.query_set {
                // The outer end intentionally precedes instrumentation resolve,
                // readback copies, and the optional completion fence.
                self.encoder.write_timestamp(query_set, 1);
            }
        }
        let readback = match (self.query_set.take(), self.query_permit.take()) {
            (Some(query_set), Some(permit))
                if (self.query_pairs != 0 || self.outer_pair)
                    && u64::from(self.query_pairs + u32::from(self.outer_pair))
                        * BYTES_PER_PAIR
                        <= MAX_QUERY_REQUESTED_BYTES / 2 =>
            {
                let total_pairs = self.query_pairs + u32::from(self.outer_pair);
                let bytes = u64::from(total_pairs) * BYTES_PER_PAIR;
                let resolve =
                    WgpuBufferOwner::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("runmat-compute-timestamp-resolve"),
                        size: bytes,
                        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }));
                let staging =
                    WgpuBufferOwner::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("runmat-compute-timestamp-readback"),
                        size: bytes,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
                self.encoder
                    .resolve_query_set(&query_set, 0..(2 * total_pairs), &resolve, 0);
                self.encoder
                    .copy_buffer_to_buffer(&resolve, 0, &staging, 0, bytes);
                Some(TimestampReadback {
                    staging: ObservationBuffer::Ordinary(staging),
                    _resolve: ObservationBuffer::Ordinary(resolve),
                    _query_set: query_set,
                    _permit: permit,
                    query_pairs: self.query_pairs,
                    outer_pair: self.outer_pair,
                    compute_complete: self.timing_complete && self.query_pairs == self.pass_count,
                    period_ns: self
                        .timestamp
                        .expect("query admission has a timestamp period")
                        .period_ns,
                })
            }
            _ => None,
        };
        let query_allocation = if self.dispatches.is_none() {
            None
        } else if self.pass_count == 0 && !self.outer_pair {
            Some(WgpuQueryAllocationDisposition::NoComputePass)
        } else if self.timestamp.is_none() {
            Some(WgpuQueryAllocationDisposition::Unsupported)
        } else if let Some(readback) = &readback {
            let bytes =
                u64::from(readback.query_pairs + u32::from(readback.outer_pair)) * BYTES_PER_PAIR;
            Some(WgpuQueryAllocationDisposition::Allocated {
                timestamp_slots: if readback.outer_pair {
                    MAX_TIMESTAMP_SLOTS_PER_ENCODER
                } else {
                    MAX_TIMESTAMP_SLOTS_PER_ENCODER - 2
                },
                resolve_requested_bytes: bytes,
                staging_requested_bytes: bytes,
                map_callbacks: 1,
            })
        } else if self.caller_timestamp {
            Some(WgpuQueryAllocationDisposition::CallerOwned)
        } else {
            Some(WgpuQueryAllocationDisposition::Denied)
        };
        EncodedMeasurement {
            encoder: self.encoder,
            dispatches: self.dispatches,
            empty_compute_elapsed: (self.pass_count == 0 && self.dispatches.is_some())
                .then_some(Duration::ZERO),
            empty_device_elapsed: None,
            readback,
            query_allocation,
            admission: self.admission,
            command: self.command,
            #[cfg(not(target_arch = "wasm32"))]
            progress: self.progress,
        }
    }
}

pub(crate) struct ProfiledComputePass<'a> {
    pass: wgpu::ComputePass<'a>,
    dispatches: &'a mut Option<u64>,
}

impl ProfiledComputePass<'_> {
    pub(crate) fn set_pipeline(&mut self, pipeline: &wgpu::ComputePipeline) {
        self.pass.set_pipeline(pipeline);
    }

    pub(crate) fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &wgpu::BindGroup,
        offsets: &[wgpu::DynamicOffset],
    ) {
        self.pass.set_bind_group(index, bind_group, offsets);
    }

    pub(crate) fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        if let Some(count) = *self.dispatches {
            *self.dispatches = count.checked_add(1);
        }
        self.pass.dispatch_workgroups(x, y, z);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_pass_intervals_sum_without_counting_host_gaps() {
        let mut bytes = Vec::new();
        for tick in [10_u64, 15, 100, 107] {
            bytes.extend_from_slice(&tick.to_le_bytes());
        }
        assert_eq!(
            decode_timestamp_pairs(&bytes, 2, 2.0),
            Some(Duration::from_nanos(24))
        );
        assert_eq!(decode_timestamp_pairs(&bytes, 3, 2.0), None);
        assert_eq!(decode_timestamp_pairs(&bytes, 2, f32::NAN), None);
        bytes[24..32].copy_from_slice(&99_u64.to_le_bytes());
        assert_eq!(decode_timestamp_pairs(&bytes, 2, 2.0), None);
    }

    #[test]
    fn outer_encoder_interval_is_independent_of_compute_pass_sum() {
        let mut bytes = Vec::new();
        for tick in [5_u64, 120, 10, 15, 100, 107] {
            bytes.extend_from_slice(&tick.to_le_bytes());
        }
        let decoded = decode_timestamp_intervals(&bytes, true, 2, true, 2.0);
        assert_eq!(decoded.device_elapsed, Some(Duration::from_nanos(230)));
        assert_eq!(decoded.compute_elapsed, Some(Duration::from_nanos(24)));
        assert_eq!(
            decode_timestamp_intervals(&bytes[..16], true, 0, true, 2.0),
            DecodedTimestamps {
                device_elapsed: Some(Duration::from_nanos(230)),
                compute_elapsed: Some(Duration::ZERO),
            }
        );
        assert_eq!(
            decode_timestamp_intervals(&bytes, true, 2, false, 2.0).compute_elapsed,
            None
        );
        assert_eq!(
            decode_timestamp_intervals(&bytes[..32], true, 2, true, 2.0),
            DecodedTimestamps::default()
        );
    }
}
