//! Immutable original electrical closure and unbound thermal intent for retained heat input.
use super::*;
use runmat_canonical_codec::{
    profile_serialized, serialized_materialization_resident_bytes_estimate,
    SerializedCanonicalProfile,
};
use serde::{Deserialize, Serialize};

const PREFIX: &[u8] = b"runmat-retained-joule-source\0";
const LIMITS: CanonicalLimits = crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

/// The enclosing immutable task/root authenticates this source object's identity.
/// Its expected result is the original retained producer result, never a new solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetainedJouleHeatSource {
    electrical_workload: ElectricalStaticWorkload,
    electrical_result: ElectricalStaticSolverResult,
    expected_result: SolverResultIdentity,
    thermal_intent: ThermalWorkload,
    transfer: RetainedJouleHeatTransfer,
}

impl RetainedJouleHeatSource {
    /// Retained physical inputs cannot be downgraded to ordinary imported heat tensors.
    pub fn requires_binding(workload: &ThermalWorkload) -> bool {
        workload.model.signals.iter().any(|signal| {
            matches!(
                signal.provenance(),
                runmat_analysis_core::SignalProvenance::Retained { .. }
            )
        })
    }

    pub fn new(
        electrical: ElectricalStaticWorkload,
        result: ElectricalStaticSolverResult,
        expected_result: SolverResultIdentity,
        thermal_intent: ThermalWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Self> {
        let (transfer, _) =
            bind_retained_joule_heat(&electrical, &result, expected_result, &thermal_intent, mesh)?;
        Ok(Self {
            electrical_workload: electrical,
            electrical_result: result,
            expected_result,
            thermal_intent,
            transfer,
        })
    }

    fn validate(&self) -> AnalysisExecutionResult<()> {
        self.electrical_workload.validate()?;
        self.electrical_result.validate()?;
        self.thermal_intent.validate()?;
        self.transfer.validate()?;
        if self.electrical_result.source_workload() != self.electrical_workload.identity()?
            || self.electrical_result.producer() != self.electrical_workload.producer_identity()?
            || self.electrical_result.source_result_identity()? != self.expected_result
        {
            return Err(invalid(
                "retained Joule source differs from its original producer",
            ));
        }
        Ok(())
    }

    /// Reconstruct the target from original retained values, never from a trusted tensor alone.
    pub fn bind(&self, mesh: &SolverMeshArtifact) -> AnalysisExecutionResult<ThermalWorkload> {
        let (transfer, bound) = bind_retained_joule_heat(
            &self.electrical_workload,
            &self.electrical_result,
            self.expected_result,
            &self.thermal_intent,
            mesh,
        )?;
        if transfer != self.transfer {
            return Err(invalid(
                "retained Joule source transfer differs from original result",
            ));
        }
        Ok(bound)
    }

    pub fn validate_for(
        &self,
        bound: &ThermalWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        if &self.bind(mesh)? != bound {
            return Err(invalid(
                "thermal workload differs from its retained source binding",
            ));
        }
        Ok(())
    }

    pub fn thermal_intent(&self) -> &ThermalWorkload {
        &self.thermal_intent
    }
    pub fn electrical_workload(&self) -> &ElectricalStaticWorkload {
        &self.electrical_workload
    }
    pub fn electrical_result(&self) -> &ElectricalStaticSolverResult {
        &self.electrical_result
    }
    pub fn transfer(&self) -> &RetainedJouleHeatTransfer {
        &self.transfer
    }

    /// Streaming profile; this does not assemble, bind, solve, or allocate numerical arrays.
    pub fn encoding_profile(&self) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
        Ok(profile_serialized(self, LIMITS)?)
    }

    /// Conservative requested-heap bound for original-result validation, target copies and
    /// canonical identity scratch. Transport/decode input budgets are reserved separately.
    /// Counts and streaming profiles precede projection construction or target binding.
    pub fn validation_workspace_bytes(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<u64> {
        let source = serialized_materialization_resident_bytes_estimate(self, LIMITS)?;
        let mesh_bytes = serialized_materialization_resident_bytes_estimate(mesh, LIMITS)?;
        let projection =
            crate::topology::AnalysisProjection::derivation_resident_bytes_estimate(mesh)
                .map_err(invalid)?;
        let nodes = u64::try_from(mesh.topology.nodes.len()).map_err(invalid)?;
        let elements = u64::try_from(mesh.topology.volume_elements.len()).map_err(invalid)?;
        let terminals =
            u64::try_from(self.electrical_workload.model.constraints.len()).map_err(invalid)?;
        let required = source
            .checked_mul(4)
            .and_then(|n| n.checked_add(mesh_bytes.checked_mul(2)?))
            .and_then(|n| n.checked_add(projection.checked_mul(2)?))
            .and_then(|n| n.checked_add(nodes.checked_add(elements)?.checked_mul(256)?))
            .and_then(|n| n.checked_add(nodes.checked_mul(terminals)?.checked_mul(64)?))
            .ok_or_else(|| invalid("retained source validation workspace overflow"))?;
        Ok(required)
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "retained-joule-source",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let source: Self = decode_contract(PREFIX, "retained-joule-source", bytes, LIMITS)?;
        source.validate()?;
        Ok(source)
    }
    pub fn identity(&self) -> AnalysisExecutionResult<Digest> {
        Ok(Digest::sha256(self.canonical_bytes()?))
    }
}
