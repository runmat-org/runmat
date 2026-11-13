//! Cold decoding yields claims; only original source-bound recovery authorizes numerical use.
use super::*;
use crate::{
    admission::AnalysisRunIdentity, physics::cfd::canonical::CanonicalFluidInitialState,
    properties::CompiledPropertyMapIdentity,
};
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultData<I, B> {
    schema_version: u16,
    workload: AnalysisWorkloadIdentity,
    property_identity: CompiledPropertyMapIdentity,
    property_audit_digest: Digest,
    formulation: AnalysisRunIdentity,
    initial: I,
    original: B,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FluidTransientSolverResult {
    data: ResultData<CanonicalFluidInitialState, super::super::FluidTransientBlockResult>,
}
fn data<I, B>(
    p: &PreparedFluidTransient<'_>,
    initial: I,
    original: B,
) -> AnalysisExecutionResult<ResultData<I, B>> {
    Ok(ResultData {
        schema_version: 1,
        workload: p.workload.identity()?,
        property_identity: p.source.property_identity.clone(),
        property_audit_digest: p.source.property_audit_digest,
        formulation: p.formulation.clone(),
        initial,
        original,
    })
}
impl FluidTransientSolverResult {
    pub(super) fn from_original(
        p: &PreparedFluidTransient<'_>,
        original: super::super::FluidTransientBlockResult,
    ) -> AnalysisExecutionResult<Self> {
        Ok(Self {
            data: data(p, p.initial.clone(), original)?,
        })
    }
    /// Retained original intervals with their schedule interval ordinals.
    pub fn retained_results(
        &self,
    ) -> &[(usize, crate::physics::cfd::canonical::CanonicalFluidResult)] {
        self.data.original.retained_results()
    }
    /// Evidence of every accepted interval.
    pub fn intervals(&self) -> &[super::super::FluidIntervalEvidence] {
        self.data.original.intervals()
    }
    /// Terminal original fields of the march.
    pub fn terminal(&self) -> Option<&crate::physics::cfd::canonical::CanonicalFluidResult> {
        self.data.original.terminal()
    }
    pub fn initial_state(&self) -> &CanonicalFluidInitialState {
        &self.data.initial
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        Ok(runmat_canonical_codec::encode_contract(
            b"runmat-fluid-transient-result/v1\0",
            "fluid-transient-result",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(runmat_canonical_codec::decode_contract(
            b"runmat-fluid-transient-result/v1\0",
            "fluid-transient-result",
            bytes,
            LIMITS,
        )?)
    }
    pub fn recover(
        &self,
        workload: &FluidTransientWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        let p = workload.prepare(mesh)?;
        let d = &self.data;
        if d.schema_version != 1
            || d.workload != workload.identity()?
            || d.formulation != p.formulation
            || d.property_identity != p.source.property_identity
            || d.property_audit_digest != p.source.property_audit_digest
            || d.initial != p.initial
        {
            return Err(invalid(
                "transient original workload/property/formulation/initial state differs",
            ));
        }
        p.block()?.recover(&d.original)
    }
}
impl PreparedFluidTransient<'_> {
    /// Maximum-width serializable shape of the retained result, for wire profiles that
    /// embed it. Exact topology and schedule fix every cardinality; values are placeholders.
    pub(in crate::execution::fluid) fn result_shape(
        &self,
    ) -> AnalysisExecutionResult<impl Serialize + '_> {
        let source = &self.source;
        let model = &self.workload.model;
        let schedule =
            super::super::FluidTransientSchedule::derive(model, self.workload.selected_step()?)?;
        let authorities = crate::physics::cfd::canonical::potential_boundary_authorities(
            &source.constraints,
            &model.boundaries,
            &model.excitations,
        );
        let original = super::super::profile::original_block_profile(
            source.topology.cells.len(),
            source.topology.faces.len(),
            authorities,
            schedule.points_s().len() - 1,
            super::super::time_block::retained_ordinals(
                self.workload.selected_step()?,
                &schedule,
                &(0..schedule.points_s().len() - 1),
            )
            .len(),
        );
        data(self, &self.initial, original)
    }

    pub fn result_encoding_profile(
        &self,
    ) -> AnalysisExecutionResult<runmat_canonical_codec::SerializedCanonicalProfile> {
        let source = &self.source;
        let model = &self.workload.model;
        let schedule =
            super::super::FluidTransientSchedule::derive(model, self.workload.selected_step()?)?;
        let authorities = crate::physics::cfd::canonical::potential_boundary_authorities(
            &source.constraints,
            &model.boundaries,
            &model.excitations,
        );
        let original = super::super::profile::original_block_profile(
            source.topology.cells.len(),
            source.topology.faces.len(),
            authorities,
            schedule.points_s().len() - 1,
            super::super::time_block::retained_ordinals(
                self.workload.selected_step()?,
                &schedule,
                &(0..schedule.points_s().len() - 1),
            )
            .len(),
        );
        Ok(runmat_canonical_codec::profile_serialized(
            &data(self, &self.initial, original)?,
            LIMITS,
        )?)
    }
}
