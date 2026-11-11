//! Original retained Joule values bound to a shared element-scalar signal.
//! Import authenticates the expected original result; it never executes electrical physics.
use super::*;
use crate::physics::electrical::ElectricalConstitutiveResponse;
use runmat_analysis_core::{
    CoordinateFrame, FieldLocation, HeatGenerationDensity, PhysicalSignal, Signal,
    SignalApproximationEvidence, SignalId, SignalPrecision, SignalProvenance, SignalRepresentation,
    SignalShape, SignalValue,
};
use serde::{Deserialize, Serialize};

const PREFIX: &[u8] = b"runmat-retained-joule-transfer\0";
const LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetainedJouleHeatTransfer {
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    producer: StepIdentity,
    mesh_digest: Digest,
    ordered_element_ids: Vec<u64>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    density_w_per_m3: runmat_types::physical::Quantity<
        runmat_types::physical::kind::HeatGenerationDensity,
        Vec<f64>,
    >,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    power_w: runmat_types::physical::Power,
}

impl RetainedJouleHeatTransfer {
    pub fn from_result(
        workload: &ElectricalStaticWorkload,
        result: &ElectricalStaticSolverResult,
        mesh: &SolverMeshArtifact,
        expected_result: SolverResultIdentity,
    ) -> AnalysisExecutionResult<Self> {
        result.validate_for(workload, mesh)?;
        if workload.equation()? != runmat_analysis_core::StaticElectricalEquation::SteadyConduction
            || result.source_result_identity()? != expected_result
        {
            return Err(invalid(
                "Joule transfer requires the authenticated original conduction result",
            ));
        }
        let ElectricalConstitutiveResponse::SteadyConduction {
            joule_density_w_per_m3,
            total_joule_power_w,
            ..
        } = &result.response().constitutive
        else {
            return Err(invalid(
                "Joule transfer cannot substitute electrostatic energy density",
            ));
        };
        let transfer = Self {
            source_workload: result.source_workload(),
            source_result: expected_result,
            producer: result.producer(),
            mesh_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            ordered_element_ids: result.response().ordered_element_ids.clone(),
            density_w_per_m3: joule_density_w_per_m3.clone(),
            power_w: *total_joule_power_w,
        };
        transfer.validate()?;
        Ok(transfer)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.ordered_element_ids.is_empty()
            || self.ordered_element_ids.len() != self.density_w_per_m3.canonical_values().len()
            || self
                .ordered_element_ids
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != self.ordered_element_ids.len()
            || self
                .density_w_per_m3
                .canonical_values()
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || !self.power_w.canonical_value().is_finite()
            || self.power_w.canonical_value() < 0.0
        {
            return Err(invalid(
                "Joule transfer has invalid canonical rows or physical values",
            ));
        }
        Ok(())
    }

    pub fn validate_for(
        &self,
        workload: &ElectricalStaticWorkload,
        result: &ElectricalStaticSolverResult,
        mesh: &SolverMeshArtifact,
        expected_result: SolverResultIdentity,
    ) -> AnalysisExecutionResult<()> {
        if self != &Self::from_result(workload, result, mesh, expected_result)? {
            return Err(invalid(
                "Joule transfer differs from the original source authority",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "retained-joule-transfer",
            self,
            LIMITS,
        )?)
    }

    pub fn from_canonical_bytes_for(
        bytes: &[u8],
        workload: &ElectricalStaticWorkload,
        result: &ElectricalStaticSolverResult,
        mesh: &SolverMeshArtifact,
        expected_result: SolverResultIdentity,
    ) -> AnalysisExecutionResult<Self> {
        let transfer: Self = decode_contract(PREFIX, "retained-joule-transfer", bytes, LIMITS)?;
        transfer.validate_for(workload, result, mesh, expected_result)?;
        Ok(transfer)
    }

    pub fn digest(&self) -> AnalysisExecutionResult<Digest> {
        Ok(Digest::sha256(self.canonical_bytes()?))
    }

    pub fn power_w(&self) -> f64 {
        self.power_w.canonical_value()
    }

    pub(super) fn signal(&self, signal_id: SignalId) -> AnalysisExecutionResult<PhysicalSignal> {
        Ok(PhysicalSignal::HeatGenerationDensity(Signal {
            signal_id,
            shape: SignalShape::TopologyField {
                topology_id: self.mesh_digest.to_string(),
                location: FieldLocation::Element,
                component_count: 1,
            },
            frame: Some(CoordinateFrame::Global),
            representation: SignalRepresentation::Constant(SignalValue::Tensor {
                shape: vec![self.density_w_per_m3.canonical_values().len(), 1],
                values: self
                    .density_w_per_m3
                    .canonical_values()
                    .iter()
                    .copied()
                    .map(HeatGenerationDensity::new)
                    .collect(),
            }),
            precision: SignalPrecision::Real64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: None,
                max_relative_error: None,
                derivative_max_absolute_error: None,
                sample_count: 1,
            },
            provenance: SignalProvenance::Retained {
                source_digest: self.digest()?.to_string(),
            },
        }))
    }
}
