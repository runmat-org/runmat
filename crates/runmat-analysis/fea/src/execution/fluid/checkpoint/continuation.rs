//! Shared source-bound continuation law; no scheduler or alternate physical integrator.
use super::*;
use std::ops::Range;
const REQUEST_PREFIX: &[u8] = b"runmat-fluid-transient-block-workload/v1\0";
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidTransientBlockWorkload {
    pub workload: FluidTransientWorkload,
    pub predecessor: FluidTransientCheckpoint,
    pub end_interval: usize,
}
pub struct PreparedFluidTransientContinuation<'a> {
    request: &'a FluidTransientBlockWorkload,
    prepared: PreparedFluidTransient<'a>,
    recovery: RecoveryReservation,
}
impl FluidTransientBlockWorkload {
    pub fn encoding_profile(
        &self,
    ) -> AnalysisExecutionResult<runmat_canonical_codec::SerializedCanonicalProfile> {
        Ok(runmat_canonical_codec::profile_serialized(self, LIMITS)?)
    }
    pub fn producer_identity(&self) -> AnalysisExecutionResult<crate::execution::StepIdentity> {
        self.validate()?;
        crate::execution::StepIdentity::from_solver_contract(
            crate::execution::SequenceIdentity::from_solver_contract(
                &self.workload.model.sequence,
            )?,
            &self.workload.model.sequence.steps[0],
        )
    }
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if crate::is_fea_cancelled() {
            return Err(invalid("fluid continuation cancelled"));
        }
        let profile = runmat_canonical_codec::profile_serialized(self, LIMITS)?;
        super::super::source::require_bytes(
            profile.materialization_resident_bytes_estimate,
            self.workload.resident_ceiling()?,
        )?;
        self.workload.validate()?;
        if self.predecessor.next_interval() >= self.end_interval {
            return Err(invalid(
                "fluid continuation must advance original interval ordinals",
            ));
        }
        Ok(())
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(runmat_canonical_codec::encode_contract(
            REQUEST_PREFIX,
            "fluid-transient-block-workload",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let result: Self = runmat_canonical_codec::decode_contract(
            REQUEST_PREFIX,
            "fluid-transient-block-workload",
            bytes,
            LIMITS,
        )?;
        result.validate()?;
        Ok(result)
    }
    pub fn identity(&self) -> AnalysisExecutionResult<AnalysisWorkloadIdentity> {
        AnalysisWorkloadIdentity::derive(
            "fluid-transient-block-workload",
            &Digest::sha256(self.canonical_bytes()?),
        )
    }
    pub fn prepare(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<PreparedFluidTransientContinuation<'_>> {
        self.validate()?;
        let prepared = self.workload.prepare(mesh)?;
        let recovery = recover(&prepared, &self.predecessor)?;
        let range = self.predecessor.next_interval()..self.end_interval;
        let block = block(
            &prepared,
            &self.predecessor.data.scope,
            self.predecessor.terminal_fields(),
            range,
        )?;
        require_work(
            self.predecessor.charged_operator_work(),
            require_work(
                recovery.operator_work,
                block.reservation().operator_work_upper_bound,
                self.workload.policy.operator_work_budget(),
            )?,
            self.workload.policy.operator_work_budget(),
        )?;
        Ok(PreparedFluidTransientContinuation {
            request: self,
            prepared,
            recovery,
        })
    }
    /// Bind a cold output to the exact requested immutable predecessor and interval range.
    pub fn recover_output(
        &self,
        checkpoint: &FluidTransientCheckpoint,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.require_output_prefix(checkpoint)?;
        checkpoint.recover(&self.workload, mesh)
    }
    fn require_output_prefix(
        &self,
        checkpoint: &FluidTransientCheckpoint,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        let old = &self.predecessor.data;
        let new = &checkpoint.data;
        let [last] = new.blocks.get(old.blocks.len()..).unwrap_or(&[]) else {
            return Err(invalid(
                "fluid output does not append exactly one requested block",
            ));
        };
        if new.scope != old.scope
            || new.blocks[..old.blocks.len()] != old.blocks
            || last.start_interval != old.next_interval
            || last.end_interval != self.end_interval
            || old.next_output_ordinal != old.next_interval
            || new.next_interval != self.end_interval
            || old
                .charged_operator_work
                .checked_add(last.charged_operator_work)
                != Some(new.charged_operator_work)
        {
            return Err(invalid(
                "fluid output differs from original predecessor or requested ordinals",
            ));
        }
        Ok(())
    }
    pub fn execute(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<FluidTransientCheckpoint> {
        self.prepare(mesh)?.execute()
    }
}
fn require_work(previous: u64, added: u64, maximum: u64) -> AnalysisExecutionResult<u64> {
    previous
        .checked_add(added)
        .filter(|v| *v <= maximum)
        .ok_or_else(|| invalid("fluid continuation cumulative operator work ceiling"))
}
fn block<'a>(
    p: &'a PreparedFluidTransient<'_>,
    scope: &Scope,
    initial: CanonicalFluidFields<'a>,
    range: Range<usize>,
) -> AnalysisExecutionResult<super::super::FluidTransientBlock<'a>> {
    let mut policy = p.workload.policy;
    policy.set_resident_byte_budget(profile::require_complete_budget(p, scope)?);
    super::super::FluidTransientBlock::prepare(
        &p.workload.model,
        &p.workload.step_id,
        p.source.input(&p.workload.model, policy),
        initial,
        range,
    )
}
pub(super) struct RecoveryReservation {
    resident_bytes: u64,
    operator_work: u64,
}
pub(super) fn recover(
    p: &PreparedFluidTransient<'_>,
    checkpoint: &FluidTransientCheckpoint,
) -> AnalysisExecutionResult<RecoveryReservation> {
    if crate::is_fea_cancelled() {
        return Err(invalid("fluid checkpoint recovery cancelled"));
    }
    let actual = runmat_canonical_codec::profile_serialized(checkpoint, LIMITS)?;
    super::super::source::require_bytes(
        actual.materialization_resident_bytes_estimate,
        p.workload.resident_ceiling()?,
    )?;
    let scope = Scope::derive(p)?;
    let bound = profile::complete_profile(p, &scope)?;
    profile::require_complete_budget(p, &scope)?;
    let d = &checkpoint.data;
    if d.schema_version != 1 || d.scope != scope {
        return Err(invalid("fluid checkpoint original scope differs"));
    }
    if actual.encoded_bytes_upper_bound > bound.encoded_bytes_upper_bound
        || actual.materialization_resident_bytes_estimate
            > bound.materialization_resident_bytes_estimate
    {
        return Err(invalid(format!("fluid checkpoint complete payload exceeds admitted profile: encoded={}/{}, resident={}/{}",
            actual.encoded_bytes_upper_bound, bound.encoded_bytes_upper_bound,
            actual.materialization_resident_bytes_estimate, bound.materialization_resident_bytes_estimate)));
    }
    let mut recovery_resident_bytes = 0;
    let mut recovery_work = 0;
    let mut next = 0;
    let mut work = 0;
    let mut previous = scope.initial.original_fields();
    for entry in &d.blocks {
        if crate::is_fea_cancelled() {
            return Err(invalid("fluid checkpoint recovery cancelled"));
        }
        if entry.start_interval != next || entry.end_interval <= next {
            return Err(invalid(
                "fluid checkpoint block order differs from original schedule",
            ));
        }
        let prepared = block(p, &scope, previous, next..entry.end_interval)?;
        if entry.charged_operator_work
            != require_work(
                recovery_work,
                prepared.reservation().operator_work_upper_bound,
                p.workload.policy.operator_work_budget(),
            )?
        {
            return Err(invalid("fluid checkpoint original block work differs"));
        }
        work = require_work(
            work,
            entry.charged_operator_work,
            p.workload.policy.operator_work_budget(),
        )?;
        recovery_resident_bytes =
            recovery_resident_bytes.max(prepared.reservation().resident_bytes_upper_bound);
        recovery_work = require_work(
            recovery_work,
            prepared.reservation().operator_work_upper_bound,
            p.workload.policy.operator_work_budget(),
        )?;
        prepared.recover(&entry.original)?;
        previous = entry
            .original
            .terminal()
            .ok_or_else(|| invalid("fluid checkpoint lacks original terminal fields"))?
            .original_fields();
        next = entry.end_interval;
    }
    if d.next_interval != next || d.next_output_ordinal != next || d.charged_operator_work != work {
        return Err(invalid(
            "fluid checkpoint cumulative work/output ordinals differ",
        ));
    }
    Ok(RecoveryReservation {
        resident_bytes: recovery_resident_bytes,
        operator_work: recovery_work,
    })
}
impl PreparedFluidTransientContinuation<'_> {
    /// Verify a cold output with the already admitted original source. Preserve all
    /// equation/history checks without recompiling properties and initial fields.
    pub fn recover_output(
        &self,
        checkpoint: &FluidTransientCheckpoint,
    ) -> AnalysisExecutionResult<()> {
        self.request.require_output_prefix(checkpoint)?;
        recover(&self.prepared, checkpoint)?;
        Ok(())
    }

    pub fn phase_resident_bytes_upper_bound(&self) -> AnalysisExecutionResult<u64> {
        let numerical = self
            .reservation()?
            .resident_bytes_upper_bound
            .max(self.recovery.resident_bytes);
        let remaining =
            profile::require_complete_budget(&self.prepared, &self.request.predecessor.data.scope)?;
        self.request
            .workload
            .resident_ceiling()?
            .checked_sub(remaining)
            .and_then(|v| v.checked_add(numerical))
            .ok_or_else(|| invalid("fluid transient phase heap overflow"))
    }
    pub fn result_encoding_profile(
        &self,
    ) -> AnalysisExecutionResult<runmat_canonical_codec::SerializedCanonicalProfile> {
        profile::complete_profile(&self.prepared, &self.request.predecessor.data.scope)
    }
    pub fn reservation(
        &self,
    ) -> AnalysisExecutionResult<super::super::FluidTransientBlockReservation> {
        let mut reservation = block(
            &self.prepared,
            &self.request.predecessor.data.scope,
            self.request.predecessor.terminal_fields(),
            self.request.predecessor.next_interval()..self.request.end_interval,
        )?
        .reservation();
        reservation.resident_bytes_upper_bound = reservation
            .resident_bytes_upper_bound
            .max(self.recovery.resident_bytes);
        reservation.operator_work_upper_bound = require_work(
            self.recovery.operator_work,
            reservation.operator_work_upper_bound,
            self.request.workload.policy.operator_work_budget(),
        )?;
        Ok(reservation)
    }
    pub fn execute(&self) -> AnalysisExecutionResult<FluidTransientCheckpoint> {
        if crate::is_fea_cancelled() {
            return Err(invalid("fluid continuation cancelled"));
        }
        let old = &self.request.predecessor;
        let block = block(
            &self.prepared,
            &old.data.scope,
            old.terminal_fields(),
            old.next_interval()..self.request.end_interval,
        )?;
        let task_charge = require_work(
            self.recovery.operator_work,
            block.reservation().operator_work_upper_bound,
            self.request.workload.policy.operator_work_budget(),
        )?;
        let charged = require_work(
            old.charged_operator_work(),
            task_charge,
            self.request.workload.policy.operator_work_budget(),
        )?;
        let original = block.execute()?;
        let mut result = old.clone();
        result.data.blocks.push(Entry {
            start_interval: old.next_interval(),
            end_interval: self.request.end_interval,
            charged_operator_work: task_charge,
            original,
        });
        result.data.next_interval = self.request.end_interval;
        result.data.next_output_ordinal = self.request.end_interval;
        result.data.charged_operator_work = charged;
        Ok(result)
    }
}
