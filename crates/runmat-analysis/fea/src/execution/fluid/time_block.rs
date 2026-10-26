//! Local original-field temporal blocks. This is not a trusted checkpoint or native task.
mod resources;
const RESULT_PREFIX: &[u8] = b"runmat-fluid-temporal-block-result/v1\0";
use super::{invalid, AnalysisExecutionResult, FluidTransientSchedule};
use crate::{
    excitation::ProjectionSample,
    physics::cfd::canonical::{
        CanonicalFluidFields, CanonicalFluidInput, CanonicalFluidProblem, CanonicalFluidResult,
    },
};
use runmat_analysis_core::{AnalysisConstraint, AnalysisModel, ProcedureSettings, ProcedureStepId};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};
use std::{ops::Range, sync::Arc};

/// Frozen numerical intent borrowing original authority and initial U/p-reference/p/q.
/// The selected range indexes intervals; [a,b) ends at schedule point b.
pub struct FluidTransientBlock<'a> {
    input: CanonicalFluidInput<'a>,
    initial: CanonicalFluidFields<'a>,
    schedule: FluidTransientSchedule,
    range: Range<usize>,
    identity: Digest,
    reservation: FluidTransientBlockReservation,
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidTransientBlockReservation {
    pub resident_bytes_upper_bound: u64,
    pub operator_work_upper_bound: u64,
    pub result_encoded_bytes_upper_bound: u64,
}
/// Portable numerical claims. Decoding does not independently validate equations or lineage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidTransientBlockResult {
    identity: Digest,
    start_ordinal: usize,
    end_ordinal: usize,
    original: Arc<Vec<CanonicalFluidResult>>,
}
impl FluidTransientBlockResult {
    /// Structural transport only. Original source-bound recovery remains mandatory.
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        Ok(runmat_canonical_codec::encode_contract(
            RESULT_PREFIX,
            "fluid-temporal-block-result",
            self,
            super::LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(runmat_canonical_codec::decode_contract(
            RESULT_PREFIX,
            "fluid-temporal-block-result",
            bytes,
            super::LIMITS,
        )?)
    }
    pub fn original_results(&self) -> &[CanonicalFluidResult] {
        &self.original
    }
    pub fn terminal(&self) -> Option<&CanonicalFluidResult> {
        self.original.last()
    }
}
impl<'a> FluidTransientBlock<'a> {
    pub fn prepare(
        model: &'a AnalysisModel,
        step_id: &ProcedureStepId,
        input: CanonicalFluidInput<'a>,
        initial: CanonicalFluidFields<'a>,
        range: Range<usize>,
    ) -> AnalysisExecutionResult<Self> {
        if crate::is_fea_cancelled() {
            return Err(invalid("fluid temporal block cancelled"));
        }
        // Borrowed source profiling precedes model validation, time union and field traversal.
        let metadata = runmat_canonical_codec::profile_serialized(model, super::LIMITS)?;
        if metadata.materialization_resident_bytes_estimate > input.policy.maximum_resident_bytes {
            return Err(invalid(
                "fluid block source profile exceeds resident ceiling",
            ));
        }
        runmat_analysis_core::validate_model(model).map_err(invalid)?;
        let mut selected = model
            .sequence
            .steps
            .iter()
            .filter(|s| &s.step_id == step_id);
        let step = selected
            .next()
            .filter(|_| selected.next().is_none())
            .ok_or_else(|| invalid("fluid block needs exactly one selected step"))?;
        if step.settings != ProcedureSettings::None
            || model.cfd.as_ref() != Some(input.material)
            || input.signals != model.signals
            || input.boundary_operators != model.boundaries
            || model
                .constraints
                .iter()
                .filter_map(|c| match c {
                    AnalysisConstraint::Fluid(c) => Some(c),
                    _ => None,
                })
                .ne(input.constraints.iter())
            || model.constraints.len() != input.constraints.len()
            || !step.activations.is_empty()
        {
            return Err(invalid("fluid block numerical inputs differ from authored source or require unsupported activation"));
        }
        let selected_actions =
            crate::excitation::active_actions(model, Some(step)).map_err(invalid)?;
        if selected_actions
            .map(|a| a.action)
            .ne(input.excitations.iter())
        {
            return Err(invalid(
                "fluid block actions differ from shared selected cases",
            ));
        }
        let schedule = FluidTransientSchedule::derive_with_ceiling(
            model,
            step,
            input
                .policy
                .maximum_resident_bytes
                .min(step.accuracy.resources.maximum_resident_bytes),
        )?;
        if range.start >= range.end || range.end >= schedule.points_s().len() {
            return Err(invalid(
                "fluid block interval ordinals exceed authored schedule",
            ));
        }
        let reservation = resources::reserve(
            &input,
            initial,
            &schedule,
            &range,
            metadata.materialization_resident_bytes_estimate,
            step.accuracy.resources.maximum_resident_bytes,
        )?;
        let bytes = runmat_canonical_codec::encode_contract(
            b"runmat-fluid-temporal-block/v1\0",
            "fluid-temporal-block",
            &(
                model.solver_identity_contract(),
                step_id,
                &range,
                schedule.points_s(),
                initial.velocity_m_per_s.canonical_values(),
                initial.pressure_reference_pa.canonical_values(),
                initial.pressure_fluctuation_pa.canonical_values(),
                initial.face_volume_flux_m3_per_s.canonical_values(),
                input.policy,
                input.topology.mesh_digest,
                input.topology.projection_digest,
            ),
            super::LIMITS,
        )?;
        Ok(Self {
            input,
            initial,
            schedule,
            range,
            identity: Digest::sha256(bytes),
            reservation,
        })
    }
    pub fn schedule(&self) -> &FluidTransientSchedule {
        &self.schedule
    }
    pub fn reservation(&self) -> FluidTransientBlockReservation {
        self.reservation
    }
    fn problem(&self, end: usize) -> AnalysisExecutionResult<CanonicalFluidProblem<'_>> {
        CanonicalFluidProblem::prepare_at(
            CanonicalFluidInput {
                source: self.input.source,
                topology: self.input.topology,
                material: self.input.material,
                constraints: self.input.constraints,
                boundary_operators: self.input.boundary_operators,
                signals: self.input.signals,
                excitations: self.input.excitations,
                policy: self.input.policy,
            },
            ProjectionSample::TimeSeconds {
                value: self.schedule.points_s()[end],
                approach: self.schedule.arrival(end).unwrap(),
            },
        )
        .map_err(invalid)
    }
    pub fn execute(&self) -> AnalysisExecutionResult<FluidTransientBlockResult> {
        if crate::is_fea_cancelled() {
            return Err(invalid("fluid temporal block cancelled"));
        }
        let mut original: Vec<CanonicalFluidResult> = Vec::new();
        original
            .try_reserve_exact(self.range.len())
            .map_err(invalid)?;
        for ordinal in self.range.clone() {
            let previous = original
                .last()
                .map_or(self.initial, |r| r.original_fields());
            let p = self.problem(ordinal + 1)?;
            let interval = p
                .prepare_backward_euler(
                    self.schedule.points_s()[ordinal],
                    self.schedule.points_s()[ordinal + 1],
                    previous,
                )
                .map_err(invalid)?;
            original.push(interval.solve().map_err(invalid)?);
        }
        Ok(FluidTransientBlockResult {
            identity: self.identity,
            start_ordinal: self.range.start,
            end_ordinal: self.range.end,
            original: Arc::new(original),
        })
    }
    /// Re-evaluate every original interval, borrowing its retained original predecessor.
    /// This never solves, interpolates, repairs or reconstructs fields from absolute Pa.
    pub fn recover(&self, result: &FluidTransientBlockResult) -> AnalysisExecutionResult<()> {
        if result.identity != self.identity
            || result.start_ordinal != self.range.start
            || result.end_ordinal != self.range.end
            || result.original.len() != self.range.len()
        {
            return Err(invalid("fluid block scope/ordinal/field inventory differs"));
        }
        for (index, ordinal) in self.range.clone().enumerate() {
            let previous = if index == 0 {
                self.initial
            } else {
                result.original[index - 1].original_fields()
            };
            let p = self.problem(ordinal + 1)?;
            p.prepare_backward_euler(
                self.schedule.points_s()[ordinal],
                self.schedule.points_s()[ordinal + 1],
                previous,
            )
            .map_err(invalid)?
            .recover(&result.original[index])
            .map_err(invalid)?;
        }
        Ok(())
    }
}
