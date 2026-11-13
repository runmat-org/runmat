use runmat_analysis_core::{DirectionalCombination, ResponseSpectrumId, SpectrumResponse};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::ModalBasisIdentity;

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, StateIdentity,
    StructuralModalResponseSpectrumScopeIdentity, StructuralModalResponseSpectrumWorkload,
    StructuralModalResponseSpectrumWorkloadIdentity,
};

const SCOPE_PREFIX: &[u8] = b"runmat-structural-modal-response-spectrum-scope/v1\0";
const SCOPE_LIMITS: CanonicalLimits = CanonicalLimits::new(4 * 1024 * 1024, 65_536, 65_536, 32);

/// Immutable physical and semantic boundary shared by every direction task in one study step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralModalResponseSpectrumScopeWire"
)]
pub struct StructuralModalResponseSpectrumScope {
    schema_version: u16,
    identity: StructuralModalResponseSpectrumScopeIdentity,
    workload: StructuralModalResponseSpectrumWorkloadIdentity,
    topology_digest: Digest,
    source_state: StateIdentity,
    basis: ModalBasisIdentity,
    operator_digest: Digest,
    spectrum_ids: Vec<ResponseSpectrumId>,
    response: SpectrumResponse,
    directional_combination: DirectionalCombination,
    physical_dof_count: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralModalResponseSpectrumScopeWire {
    schema_version: u16,
    identity: StructuralModalResponseSpectrumScopeIdentity,
    workload: StructuralModalResponseSpectrumWorkloadIdentity,
    topology_digest: Digest,
    source_state: StateIdentity,
    basis: ModalBasisIdentity,
    operator_digest: Digest,
    spectrum_ids: Vec<ResponseSpectrumId>,
    response: SpectrumResponse,
    directional_combination: DirectionalCombination,
    physical_dof_count: usize,
}

pub struct StructuralModalResponseSpectrumScopeInput {
    pub topology_digest: Digest,
    pub source_state: StateIdentity,
    pub basis: ModalBasisIdentity,
    pub operator_digest: Digest,
    pub physical_dof_count: usize,
}

impl TryFrom<StructuralModalResponseSpectrumScopeWire> for StructuralModalResponseSpectrumScope {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralModalResponseSpectrumScopeWire) -> Result<Self, Self::Error> {
        let scope = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            workload: wire.workload,
            topology_digest: wire.topology_digest,
            source_state: wire.source_state,
            basis: wire.basis,
            operator_digest: wire.operator_digest,
            spectrum_ids: wire.spectrum_ids,
            response: wire.response,
            directional_combination: wire.directional_combination,
            physical_dof_count: wire.physical_dof_count,
        };
        scope.validate()?;
        Ok(scope)
    }
}

impl StructuralModalResponseSpectrumScope {
    pub fn new(
        workload: &StructuralModalResponseSpectrumWorkload,
        input: StructuralModalResponseSpectrumScopeInput,
    ) -> AnalysisExecutionResult<Self> {
        workload.validate()?;
        let spectra = workload.spectra()?;
        let mut scope = Self {
            schema_version: 1,
            identity: StructuralModalResponseSpectrumScopeIdentity::derive(
                "structural-modal-response-spectrum-scope-placeholder",
                &1_u8,
            )?,
            workload: workload.identity(),
            topology_digest: input.topology_digest,
            source_state: input.source_state,
            basis: input.basis,
            operator_digest: input.operator_digest,
            spectrum_ids: spectra
                .iter()
                .map(|spectrum| spectrum.spectrum_id.clone())
                .collect(),
            response: spectra[0].response,
            directional_combination: spectra[0].directional_combination,
            physical_dof_count: input.physical_dof_count,
        };
        scope.identity = scope.derive_identity()?;
        scope.validate_for(workload)?;
        Ok(scope)
    }

    pub const fn identity(&self) -> StructuralModalResponseSpectrumScopeIdentity {
        self.identity
    }
    pub const fn workload(&self) -> StructuralModalResponseSpectrumWorkloadIdentity {
        self.workload
    }
    pub const fn topology_digest(&self) -> Digest {
        self.topology_digest
    }
    pub const fn source_state(&self) -> StateIdentity {
        self.source_state
    }
    pub const fn basis(&self) -> ModalBasisIdentity {
        self.basis
    }
    pub const fn operator_digest(&self) -> Digest {
        self.operator_digest
    }
    pub fn spectrum_ids(&self) -> &[ResponseSpectrumId] {
        &self.spectrum_ids
    }
    pub const fn response(&self) -> SpectrumResponse {
        self.response
    }
    pub const fn directional_combination(&self) -> DirectionalCombination {
        self.directional_combination
    }
    pub const fn physical_dof_count(&self) -> usize {
        self.physical_dof_count
    }

    pub fn validate_for(
        &self,
        workload: &StructuralModalResponseSpectrumWorkload,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        workload.validate()?;
        let spectra = workload.spectra()?;
        let expected_ids = spectra
            .iter()
            .map(|spectrum| &spectrum.spectrum_id)
            .collect::<Vec<_>>();
        if self.workload != workload.identity()
            || self.source_state != workload.modal_basis_transition().source_state
            || self.topology_digest != workload.modal_basis_transition().target_topology_digest
            || self.spectrum_ids.iter().collect::<Vec<_>>() != expected_ids
            || self.response != spectra[0].response
            || self.directional_combination != spectra[0].directional_combination
        {
            return Err(AnalysisExecutionError::Identity(
                "response-spectrum scope differs from its workload or modal-basis transition",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            SCOPE_PREFIX,
            "structural-modal-response-spectrum-scope",
            self,
            SCOPE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            SCOPE_PREFIX,
            "structural-modal-response-spectrum-scope",
            bytes,
            SCOPE_LIMITS,
        )?)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let unique_ids = self
            .spectrum_ids
            .iter()
            .collect::<std::collections::BTreeSet<_>>();
        if self.schema_version != 1
            || self.physical_dof_count == 0
            || self.spectrum_ids.is_empty()
            || self.spectrum_ids.len() > 3
            || unique_ids.len() != self.spectrum_ids.len()
            || self.identity != self.derive_identity()?
        {
            return Err(AnalysisExecutionError::Invalid(
                "response-spectrum scope is not canonical".into(),
            ));
        }
        Ok(())
    }

    fn derive_identity(
        &self,
    ) -> AnalysisExecutionResult<StructuralModalResponseSpectrumScopeIdentity> {
        StructuralModalResponseSpectrumScopeIdentity::derive(
            "structural-modal-response-spectrum-scope",
            &(
                self.schema_version,
                self.workload,
                self.topology_digest,
                self.source_state,
                self.basis,
                self.operator_digest,
                &self.spectrum_ids,
                self.response,
                self.directional_combination,
                self.physical_dof_count,
            ),
        )
    }
}
