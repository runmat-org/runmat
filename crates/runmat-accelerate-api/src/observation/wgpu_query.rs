//! Optional WGPU encoder-query demand and actual allocation facts.
//!
//! These facts never authorize a mandatory provider resource ticket. QuerySet
//! native bytes are opaque; buffer extents are requested bytes, not VRAM usage.

/// Checked upper bound from the provider's validated physical encoder route.
/// Requested bytes cover resolve and staging buffers, not opaque QuerySet
/// native allocation or driver VRAM granularity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WgpuEncoderQueryProspective {
    /// Both counts come from the same typed route. Noncompute encoders can
    /// carry an outer pair only when command profiling and the device feature
    /// are active; pass-only observation uses the compute subset.
    #[serde(default)]
    pub max_compute_encoders: u32,
    #[serde(default)]
    pub max_total_encoders: u32,
    pub max_query_sets: u32,
    pub max_timestamp_slots: u64,
    pub requested_peak_buffer_bytes: u64,
    pub max_map_callbacks: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WgpuQueryAllocationDisposition {
    NoComputePass,
    Unsupported,
    CallerOwned,
    Denied,
    Allocated {
        /// Exact QuerySet capacity; driver allocation bytes are not exposed.
        timestamp_slots: u32,
        resolve_requested_bytes: u64,
        staging_requested_bytes: u64,
        map_callbacks: u32,
    },
}

/// Checked cumulative actual query allocations from settled submissions.
/// Requested buffer extents are not in-use peak or native VRAM consumption.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WgpuQueryAllocationObservation {
    pub query_sets: u64,
    pub timestamp_slots: u64,
    pub resolve_requested_bytes: u64,
    pub staging_requested_bytes: u64,
    pub map_callbacks: u64,
    pub no_compute_submissions: u64,
    pub unsupported_submissions: u64,
    pub caller_owned_submissions: u64,
    pub denied_submissions: u64,
}

impl WgpuQueryAllocationObservation {
    pub(super) fn checked_record(
        mut self,
        disposition: WgpuQueryAllocationDisposition,
    ) -> Option<Self> {
        match disposition {
            WgpuQueryAllocationDisposition::NoComputePass => {
                self.no_compute_submissions = self.no_compute_submissions.checked_add(1)?;
            }
            WgpuQueryAllocationDisposition::Unsupported => {
                self.unsupported_submissions = self.unsupported_submissions.checked_add(1)?;
            }
            WgpuQueryAllocationDisposition::CallerOwned => {
                self.caller_owned_submissions = self.caller_owned_submissions.checked_add(1)?;
            }
            WgpuQueryAllocationDisposition::Denied => {
                self.denied_submissions = self.denied_submissions.checked_add(1)?;
            }
            WgpuQueryAllocationDisposition::Allocated {
                timestamp_slots,
                resolve_requested_bytes,
                staging_requested_bytes,
                map_callbacks,
            } => {
                self.query_sets = self.query_sets.checked_add(1)?;
                self.timestamp_slots = self
                    .timestamp_slots
                    .checked_add(u64::from(timestamp_slots))?;
                self.resolve_requested_bytes = self
                    .resolve_requested_bytes
                    .checked_add(resolve_requested_bytes)?;
                self.staging_requested_bytes = self
                    .staging_requested_bytes
                    .checked_add(staging_requested_bytes)?;
                self.map_callbacks = self.map_callbacks.checked_add(u64::from(map_callbacks))?;
            }
        }
        Some(self)
    }
}
