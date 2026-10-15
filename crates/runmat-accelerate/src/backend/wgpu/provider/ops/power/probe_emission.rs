//! Private pre-effect geometry and post-effect observations for the ordinary
//! POWER domain probe. No submission-completion, retirement or peak claim.
use super::super::io::PowerProbeReadbackObserved;
use super::super::reduction::{AnyStageObserved, FrozenAnySnapshot};
use super::*;
#[cfg(test)]
use crate::backend::wgpu::dispatch::observation::{ProbeFenceState, ProbeFenceTicket};
use crate::backend::wgpu::residency::BufferResidency;
use runmat_accelerate_api::allocation::TensorViewIdentity;
use runmat_accelerate_api::observation::{ProviderResourcePhasePlan, ProviderResourceScope};
use runmat_accelerate_api::ProviderRealElementwiseObserved;

/// POWER's phase names live with its physical owner, not the shared ledger.
#[derive(Clone, Copy)]
pub(crate) enum PowerProbeResourcePhase {
    Flag,
    AnyRows,
    AnyColumns,
    Readback,
}

impl PowerProbeResourcePhase {
    pub(crate) fn begin(self, scope: &ProviderResourceScope) -> Result<()> {
        let index = match self {
            Self::Flag => 0,
            Self::AnyRows => 1,
            Self::AnyColumns => 2,
            Self::Readback => 3,
        };
        let issued = scope.begin_next_phase()?;
        ensure!(
            issued.scope_id() == scope.id() && issued.index() == index,
            "POWER probe resource phase sequence mismatch"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProbeStep {
    FlattenAlias,
    AnyRows,
    AnyColumns,
    ScalarReadback,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProbeCompletionState {
    Absent,
    Pending,
    Completed,
    Failed,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SubmissionStep {
    FlagWarmup,
    FlagChunk,
    AnyRowsSnapshot,
    AnyRowsDispatch,
    AnyColumnsSnapshot,
    AnyColumnsDispatch,
    ScalarReadbackCopy,
}

#[cfg(test)]
pub(crate) struct ProbeSubmissionRecord {
    submissions: Vec<(SubmissionStep, ProbeFenceTicket)>,
}

#[cfg(test)]
impl ProbeSubmissionRecord {
    pub(crate) fn completion_state(&self) -> ProbeCompletionState {
        if self.submissions.is_empty()
            || self.submissions.iter().any(|(_, ticket)| {
                matches!(
                    ticket.state(),
                    ProbeFenceState::Armed | ProbeFenceState::Failed
                )
            })
        {
            return ProbeCompletionState::Failed;
        }
        if self
            .submissions
            .iter()
            .any(|(_, ticket)| ticket.state() == ProbeFenceState::Absent)
        {
            return ProbeCompletionState::Absent;
        }
        if self
            .submissions
            .iter()
            .any(|(_, ticket)| ticket.state() == ProbeFenceState::Pending)
        {
            return ProbeCompletionState::Pending;
        }
        ProbeCompletionState::Completed
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StageDemand {
    input_bytes: u64,
    output_bytes: u64,
    snapshot_copy_bytes: Option<u64>,
    uniform_bytes: u64,
    workgroups_x: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StageFacts {
    output_bytes: u64,
    snapshot_copy_bytes: Option<u64>,
    uniform_bytes: Option<u64>,
    workgroups_x: Option<u32>,
    snapshot_copy_submitted: bool,
    dispatch_submitted: bool,
    snapshot_fence_bytes: Option<u64>,
    dispatch_fence_bytes: Option<u64>,
}

impl From<&AnyStageObserved> for StageFacts {
    fn from(observed: &AnyStageObserved) -> Self {
        Self {
            output_bytes: observed.output_buffer_bytes,
            snapshot_copy_bytes: observed.snapshot_copy_bytes,
            uniform_bytes: observed.uniform_buffer_bytes,
            workgroups_x: observed.workgroups_x,
            snapshot_copy_submitted: observed.snapshot_copy_submitted,
            dispatch_submitted: observed.dispatch_submitted,
            snapshot_fence_bytes: observed.snapshot_fence_requested_bytes,
            dispatch_fence_bytes: observed.dispatch_fence_requested_bytes,
        }
    }
}

impl StageDemand {
    fn require_facts(
        self,
        step: ProbeStep,
        snapshot: FrozenAnySnapshot,
        facts: StageFacts,
    ) -> Result<()> {
        let fence_ok = |bytes: Option<u64>| bytes.is_none_or(|bytes| bytes == 8);
        ensure!(
            self.snapshot_copy_bytes == snapshot.enabled.then_some(self.input_bytes)
                && facts.output_bytes == self.output_bytes
                && facts.snapshot_copy_bytes == self.snapshot_copy_bytes
                && facts.uniform_bytes == Some(self.uniform_bytes)
                && facts.workgroups_x == Some(self.workgroups_x)
                && facts.snapshot_copy_submitted == self.snapshot_copy_bytes.is_some()
                && facts.dispatch_submitted
                && fence_ok(facts.snapshot_fence_bytes)
                && fence_ok(facts.dispatch_fence_bytes)
                && (snapshot.enabled || facts.snapshot_fence_bytes.is_none()),
            "{step:?} physical observation mismatch"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ProbeGeometry {
    flag_bytes: u64,
    scalar_bytes: u64,
    stages: [StageDemand; 2],
    readback_staging_bytes: u64,
}

impl ProbeGeometry {
    pub(super) fn plan(
        len: usize,
        precision: NumericPrecision,
        snapshot: FrozenAnySnapshot,
        limits: &wgpu::Limits,
        shader_workgroup: u32,
    ) -> Result<Self> {
        ensure!(
            len > 0 && len <= u32::MAX as usize,
            "POWER probe length exceeds u32"
        );
        let width = match precision {
            NumericPrecision::F32 => 4usize,
            NumericPrecision::F64 => 8usize,
        };
        let flag_bytes = BufferResidency::requested_bytes(len, width)
            .ok_or_else(|| anyhow!("POWER probe flag extent overflow"))?;
        let scalar_bytes = BufferResidency::requested_bytes(1, width)
            .ok_or_else(|| anyhow!("POWER probe scalar extent overflow"))?;
        let uniform_bytes =
            std::mem::size_of::<crate::backend::wgpu::params::ReduceDimParams>() as u64;
        let workgroups_x =
            crate::backend::wgpu::dispatch::common::dispatch_size(1, shader_workgroup);
        ensure!(
            workgroups_x == 1
                && workgroups_x <= limits.max_compute_workgroups_per_dimension
                && shader_workgroup > 0
                && shader_workgroup <= limits.max_compute_workgroup_size_x
                && shader_workgroup <= limits.max_compute_invocations_per_workgroup
                && flag_bytes <= limits.max_buffer_size
                && scalar_bytes <= limits.max_buffer_size
                && flag_bytes <= u64::from(limits.max_storage_buffer_binding_size)
                && scalar_bytes <= u64::from(limits.max_storage_buffer_binding_size)
                && uniform_bytes <= u64::from(limits.max_uniform_buffer_binding_size)
                && limits.max_storage_buffers_per_shader_stage >= 2
                && limits.max_uniform_buffers_per_shader_stage >= 1
                && limits.max_bind_groups >= 1,
            "POWER probe reduction/readback resource limits unavailable"
        );
        let stage = |input_bytes| StageDemand {
            input_bytes,
            output_bytes: scalar_bytes,
            snapshot_copy_bytes: snapshot.enabled.then_some(input_bytes),
            uniform_bytes,
            workgroups_x,
        };
        Ok(Self {
            flag_bytes,
            scalar_bytes,
            stages: [stage(flag_bytes), stage(scalar_bytes)],
            readback_staging_bytes: scalar_bytes,
        })
    }
}

pub(super) struct ProbePreflight {
    owner_device_id: u32,
    source_views: [TensorViewIdentity; 2],
    len: usize,
    snapshot: FrozenAnySnapshot,
    geometry: ProbeGeometry,
}

pub(super) struct ProbePlanRequest<'a> {
    pub(super) base: &'a GpuTensorHandle,
    pub(super) exponent: &'a GpuTensorHandle,
    pub(super) owner_device_id: u32,
    pub(super) len: usize,
    pub(super) precision: NumericPrecision,
    pub(super) snapshot: FrozenAnySnapshot,
    pub(super) limits: &'a wgpu::Limits,
    pub(super) shader_workgroup: u32,
}

impl ProbePreflight {
    pub(super) fn plan(request: ProbePlanRequest<'_>) -> Result<Self> {
        let ProbePlanRequest {
            base,
            exponent,
            owner_device_id,
            len,
            precision,
            snapshot,
            limits,
            shader_workgroup,
        } = request;
        base.view().validate(base.physical())?;
        exponent.view().validate(exponent.physical())?;
        base.allocation_lease().validate_live()?;
        exponent.allocation_lease().validate_live()?;
        ensure!(
            base.device_id() == owner_device_id && exponent.device_id() == owner_device_id,
            "POWER probe source/provider device mismatch"
        );
        Ok(Self {
            owner_device_id,
            source_views: [base.view_identity(), exponent.view_identity()],
            len,
            snapshot,
            geometry: ProbeGeometry::plan(len, precision, snapshot, limits, shader_workgroup)?,
        })
    }

    /// Conservative pre-effect ceiling for scoped active-owner *backing*
    /// bytes. Earlier phase allocations remain counted even if already
    /// retired; that makes each bound safe without assuming callback order.
    /// Borrowed sources and later published numerical output are separate.
    pub(super) fn resource_phase_limits(
        &self,
        flag: &ProviderRealElementwiseObserved,
    ) -> Result<ProviderResourcePhasePlan> {
        ProviderResourcePhasePlan::new(self.resource_limits(flag)?).map_err(Into::into)
    }

    /// The resource admission projection and the active-owner ledger consume
    /// the same checked physical ceiling, including native scalar staging and
    /// mandatory submission-fence backing. This does not price the probe or
    /// certify an unknown numerical domain.
    pub(super) fn resource_limits(
        &self,
        flag: &ProviderRealElementwiseObserved,
    ) -> Result<[u64; 4]> {
        self.resource_limits_with_fences(flag, true)
    }

    /// Admission adds optional observation fences/query storage at the
    /// canonical observation owner, exactly once. The active ledger retains
    /// its fence-inclusive ceilings above.
    pub(super) fn data_resource_limits(
        &self,
        flag: &ProviderRealElementwiseObserved,
    ) -> Result<[u64; 4]> {
        self.resource_limits_with_fences(flag, false)
    }

    fn resource_limits_with_fences(
        &self,
        flag: &ProviderRealElementwiseObserved,
        include_fences: bool,
    ) -> Result<[u64; 4]> {
        self.geometry
            .resource_limits_with_fences(flag, include_fences)
    }

    pub(super) fn require_sources(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
    ) -> Result<()> {
        base.allocation_lease().validate_live()?;
        exponent.allocation_lease().validate_live()?;
        ensure!(
            base.device_id() == self.owner_device_id
                && exponent.device_id() == self.owner_device_id
                && [base.view_identity(), exponent.view_identity()] == self.source_views,
            "POWER probe source view changed"
        );
        Ok(())
    }
}

impl ProbeGeometry {
    pub(super) fn resource_limits_with_fences(
        &self,
        flag: &ProviderRealElementwiseObserved,
        include_fences: bool,
    ) -> Result<[u64; 4]> {
        ensure!(
            flag.output_buffer_bytes == Some(self.flag_bytes)
                && flag.uniform_allocations == 1
                && flag.queue_submissions > 0,
            "POWER flag resource recipe cannot bound probe scope"
        );
        let add = |left: u64, right: u64| {
            left.checked_add(right)
                .ok_or_else(|| anyhow!("POWER probe phase resource ceiling overflow"))
        };
        let flag_fences = u64::from(flag.queue_submissions)
            .checked_mul(if include_fences { 8 } else { 0 })
            .ok_or_else(|| anyhow!("POWER flag fence ceiling overflow"))?;
        let flag_peak = add(
            add(self.flag_bytes, flag.uniform_buffer_bytes)?,
            flag_fences,
        )?;
        let stage_ceiling = |previous: u64, stage: StageDemand| -> Result<u64> {
            let with_output = add(previous, stage.output_bytes)?;
            let with_snapshot = add(with_output, stage.snapshot_copy_bytes.unwrap_or(0))?;
            let with_uniform = add(with_snapshot, stage.uniform_bytes)?;
            add(
                with_uniform,
                if !include_fences {
                    0
                } else if stage.snapshot_copy_bytes.is_some() {
                    16
                } else {
                    8
                },
            )
        };
        let rows_peak = stage_ceiling(flag_peak, self.stages[0])?;
        let columns_peak = stage_ceiling(rows_peak, self.stages[1])?;
        let readback_peak = add(
            add(columns_peak, self.readback_staging_bytes)?,
            if include_fences { 8 } else { 0 },
        )?;
        Ok([flag_peak, rows_peak, columns_peak, readback_peak])
    }
}

impl ProbePreflight {
    pub(super) fn require_flag(&self, flag: &GpuTensorHandle) -> Result<()> {
        flag.view().validate(flag.physical())?;
        flag.allocation_lease().validate_live()?;
        ensure!(
            flag.physical().requested_bytes() == self.geometry.flag_bytes
                && !self
                    .source_views
                    .iter()
                    .any(|view| view.allocation() == flag.allocation_identity()),
            "POWER probe flag extent/alias mismatch"
        );
        Ok(())
    }

    pub(super) fn require_flat(
        &self,
        flag: &GpuTensorHandle,
        flat: &GpuTensorHandle,
    ) -> Result<()> {
        flat.view().validate(flat.physical())?;
        flat.allocation_lease().validate_live()?;
        ensure!(
            flat.shape() == [self.len, 1]
                && flat.allocation_identity() == flag.allocation_identity()
                && flat.physical().requested_bytes() == flag.physical().requested_bytes()
                && flat.view().layout() == runmat_accelerate_api::GpuTensorLayout::Contiguous,
            "{:?} did not preserve the flag allocation",
            ProbeStep::FlattenAlias
        );
        Ok(())
    }

    fn require_stage(
        &self,
        step: ProbeStep,
        source: TensorViewIdentity,
        output: TensorViewIdentity,
        observed: &AnyStageObserved,
        demand: StageDemand,
    ) -> Result<()> {
        ensure!(
            observed.input_view == source && observed.output_view == output,
            "{step:?} view observation mismatch"
        );
        demand.require_facts(step, self.snapshot, StageFacts::from(observed))
    }

    pub(super) fn require_stages(
        &self,
        flat: &GpuTensorHandle,
        reduced: &GpuTensorHandle,
        stages: &[AnyStageObserved; 2],
    ) -> Result<()> {
        self.require_stage(
            ProbeStep::AnyRows,
            flat.view_identity(),
            stages[0].output_view,
            &stages[0],
            self.geometry.stages[0],
        )?;
        self.require_stage(
            ProbeStep::AnyColumns,
            stages[0].output_view,
            reduced.view_identity(),
            &stages[1],
            self.geometry.stages[1],
        )
    }

    pub(super) fn require_readback(
        &self,
        reduced: &GpuTensorHandle,
        observed: &PowerProbeReadbackObserved,
    ) -> Result<()> {
        observed.require_scalar(reduced.view_identity())?;
        ensure!(
            observed.copied_bytes == Some(self.geometry.scalar_bytes)
                && observed.staging_buffer_bytes == Some(self.geometry.readback_staging_bytes)
                && observed.mapped_bytes == Some(self.geometry.readback_staging_bytes)
                && observed
                    .copy_fence_requested_bytes
                    .is_none_or(|bytes| bytes == 8),
            "{:?} physical observation mismatch",
            ProbeStep::ScalarReadback
        );
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn collect_submissions(
        &self,
        flag_tickets: &[ProbeFenceTicket],
        stages: &[AnyStageObserved; 2],
        readback: &PowerProbeReadbackObserved,
    ) -> Result<ProbeSubmissionRecord> {
        ensure!(
            !flag_tickets.is_empty() && readback.copy_ticket.is_some(),
            "POWER probe submission tickets incomplete"
        );
        let mut submissions = Vec::new();
        for (index, ticket) in flag_tickets.iter().enumerate() {
            submissions.push((
                if index == 0 {
                    SubmissionStep::FlagWarmup
                } else {
                    SubmissionStep::FlagChunk
                },
                ticket.clone(),
            ));
        }
        let rows = &stages[0];
        let columns = &stages[1];
        for (snapshot_step, dispatch_step, stage) in [
            (
                SubmissionStep::AnyRowsSnapshot,
                SubmissionStep::AnyRowsDispatch,
                rows,
            ),
            (
                SubmissionStep::AnyColumnsSnapshot,
                SubmissionStep::AnyColumnsDispatch,
                columns,
            ),
        ] {
            if self.snapshot.enabled {
                let ticket = stage
                    .snapshot_ticket
                    .as_ref()
                    .ok_or_else(|| anyhow!("{snapshot_step:?} ticket missing"))?;
                ensure!(
                    ticket.fence_requested_bytes() == stage.snapshot_fence_requested_bytes,
                    "{snapshot_step:?} fence extent mismatch"
                );
                submissions.push((snapshot_step, ticket.clone()));
            } else {
                ensure!(
                    stage.snapshot_ticket.is_none(),
                    "{snapshot_step:?} unexpected"
                );
            }
            let ticket = stage
                .dispatch_ticket
                .as_ref()
                .ok_or_else(|| anyhow!("{dispatch_step:?} ticket missing"))?;
            ensure!(
                ticket.fence_requested_bytes() == stage.dispatch_fence_requested_bytes,
                "{dispatch_step:?} fence extent mismatch"
            );
            submissions.push((dispatch_step, ticket.clone()));
        }
        let copy = readback
            .copy_ticket
            .as_ref()
            .expect("checked readback ticket");
        ensure!(
            copy.fence_requested_bytes() == readback.copy_fence_requested_bytes,
            "POWER readback copy fence extent mismatch"
        );
        submissions.push((SubmissionStep::ScalarReadbackCopy, copy.clone()));
        Ok(ProbeSubmissionRecord { submissions })
    }
}

#[cfg(test)]
#[path = "probe_emission/tests.rs"]
mod tests;
