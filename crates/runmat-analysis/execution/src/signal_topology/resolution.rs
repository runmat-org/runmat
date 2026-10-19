//! Explicit authored-selector to authenticated solver-mesh transition.
use runmat_analysis_core::{
    AuthoredSignalTopologySelector, FieldLocation, PhysicalSignal, SignalShape,
};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

use super::SignalTopologyBinding;
use crate::{AnalysisBridgeError, AnalysisBridgeResult};

const LIMITS: CanonicalLimits = CanonicalLimits::new(8 * 1024 * 1024, 65_536, 4096, 32);

/// Transported transitions must be rederived from the expected authored input and mesh.
/// This is not callable admission or permission to submit preprocessing/numerical work.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalTopologyResolution {
    authored_contract_digest: Digest,
    resolved: PhysicalSignal,
    binding: SignalTopologyBinding,
}

impl SignalTopologyResolution {
    pub fn derive(
        authored: &PhysicalSignal,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisBridgeResult<Self> {
        runmat_analysis_core::validate_physical_signals(std::slice::from_ref(authored))
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        let SignalShape::DeferredTopologyField {
            selector: AuthoredSignalTopologySelector::StudySolverMesh { location },
            component_count,
        } = authored.shape()
        else {
            return Err(AnalysisBridgeError::Invalid(
                "signal topology resolution requires a deferred authored selector".into(),
            ));
        };
        let location = FieldLocation::from(*location);
        let resolved = authored.with_shape(SignalShape::TopologyField {
            topology_id: Digest::from_bytes(*mesh.canonical_digest.bytes()).to_string(),
            location,
            component_count: *component_count,
        });
        let binding = SignalTopologyBinding::derive(&resolved, mesh)?.ok_or_else(|| {
            AnalysisBridgeError::Invalid("resolved field has no canonical mesh binding".into())
        })?;
        Ok(Self {
            authored_contract_digest: Digest::sha256(encode_contract(
                b"runmat-analysis-authored-field-topology\0",
                "authored-field-contract",
                authored,
                LIMITS,
            )?),
            resolved,
            binding,
        })
    }

    /// Expected authored intent is external authority; transported values alone are not.
    pub fn validate_for(
        &self,
        authored: &PhysicalSignal,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisBridgeResult<()> {
        if Self::derive(authored, mesh)? != *self {
            return Err(AnalysisBridgeError::Identity(
                "field topology transition differs from authored intent or mesh",
            ));
        }
        Ok(())
    }

    pub fn identity(&self) -> AnalysisBridgeResult<Digest> {
        Ok(Digest::sha256(encode_contract(
            b"runmat-analysis-resolved-field-topology\0",
            "resolved-field-transition",
            self,
            LIMITS,
        )?))
    }

    pub fn authored_contract_digest(&self) -> Digest {
        self.authored_contract_digest
    }
    pub fn resolved(&self) -> &PhysicalSignal {
        &self.resolved
    }
    pub fn binding(&self) -> &SignalTopologyBinding {
        &self.binding
    }
}

#[cfg(test)]
mod tests;
