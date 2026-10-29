use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::frequency::FrequencyResponsePlanEntry;

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, ComplexStructuralBalance,
    StructuralComplexVector, StructuralFrequencyPhysicalSampleIdentity, StructuralFrequencyScope,
    StructuralFrequencyScopeIdentity,
};

const FREQUENCY_SAMPLE_PREFIX: &[u8] = b"runmat-structural-frequency-sample/v1\0";
const FREQUENCY_SAMPLE_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 32));

/// Method- and placement-neutral physical response at one exact plan frequency.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralFrequencyPhysicalSampleWire"
)]
pub struct StructuralFrequencyPhysicalSample {
    schema_version: u16,
    identity: StructuralFrequencyPhysicalSampleIdentity,
    scope: StructuralFrequencyScopeIdentity,
    entry: FrequencyResponsePlanEntry,
    displacement: StructuralComplexVector,
    reaction: StructuralComplexVector,
    balance: ComplexStructuralBalance,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralFrequencyPhysicalSampleWire {
    schema_version: u16,
    identity: StructuralFrequencyPhysicalSampleIdentity,
    scope: StructuralFrequencyScopeIdentity,
    entry: FrequencyResponsePlanEntry,
    displacement: StructuralComplexVector,
    reaction: StructuralComplexVector,
    balance: ComplexStructuralBalance,
}

impl TryFrom<StructuralFrequencyPhysicalSampleWire> for StructuralFrequencyPhysicalSample {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralFrequencyPhysicalSampleWire) -> Result<Self, Self::Error> {
        let sample = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            scope: wire.scope,
            entry: wire.entry,
            displacement: wire.displacement,
            reaction: wire.reaction,
            balance: wire.balance,
        };
        sample.validate_intrinsic()?;
        Ok(sample)
    }
}

impl StructuralFrequencyPhysicalSample {
    pub fn new(
        scope: &StructuralFrequencyScope,
        entry: FrequencyResponsePlanEntry,
        displacement: StructuralComplexVector,
        reaction: StructuralComplexVector,
        balance: ComplexStructuralBalance,
    ) -> AnalysisExecutionResult<Self> {
        scope.validate()?;
        scope
            .plan()
            .validate_entry(entry)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let mut sample = Self {
            schema_version: 1,
            identity: StructuralFrequencyPhysicalSampleIdentity::derive(
                "structural-frequency-sample-placeholder",
                &1_u8,
            )?,
            scope: scope.identity(),
            entry,
            displacement,
            reaction,
            balance,
        };
        sample.identity = sample.derive_identity()?;
        sample.validate_against(scope)?;
        Ok(sample)
    }

    pub const fn identity(&self) -> StructuralFrequencyPhysicalSampleIdentity {
        self.identity
    }

    pub const fn entry(&self) -> FrequencyResponsePlanEntry {
        self.entry
    }

    pub fn displacement(&self) -> &StructuralComplexVector {
        &self.displacement
    }

    pub fn reaction(&self) -> &StructuralComplexVector {
        &self.reaction
    }

    pub fn balance(&self) -> &ComplexStructuralBalance {
        &self.balance
    }

    pub fn equilibrium_residual_norm(&self) -> AnalysisExecutionResult<f64> {
        Ok(self.balance.closure_metrics(&self.reaction)?.0)
    }

    pub fn relative_balance_error(&self) -> AnalysisExecutionResult<f64> {
        Ok(self.balance.closure_metrics(&self.reaction)?.1)
    }

    pub fn validate_against(
        &self,
        scope: &StructuralFrequencyScope,
    ) -> AnalysisExecutionResult<()> {
        scope.validate()?;
        self.validate_intrinsic()?;
        scope
            .plan()
            .validate_entry(self.entry)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let dof_count = scope.physical_dof_count();
        self.displacement.validate_shape(dof_count)?;
        self.reaction.validate_shape(dof_count)?;
        self.balance.validate_for_dof_count(dof_count)?;
        if self.scope != scope.identity() {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency sample names a different physical scope",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate_intrinsic()?;
        Ok(encode_contract(
            FREQUENCY_SAMPLE_PREFIX,
            "structural-frequency-physical-sample",
            self,
            FREQUENCY_SAMPLE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            FREQUENCY_SAMPLE_PREFIX,
            "structural-frequency-physical-sample",
            bytes,
            FREQUENCY_SAMPLE_LIMITS,
        )?)
    }

    fn validate_intrinsic(&self) -> AnalysisExecutionResult<()> {
        let dof_count = self.displacement.len();
        self.displacement.validate_shape(dof_count)?;
        self.reaction.validate_shape(dof_count)?;
        self.balance.validate_for_dof_count(dof_count)?;
        self.balance.closure_metrics(&self.reaction)?;
        if self.schema_version != 1 || self.identity != self.derive_identity()? {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency sample identity does not converge",
            ));
        }
        Ok(())
    }

    fn derive_identity(
        &self,
    ) -> AnalysisExecutionResult<StructuralFrequencyPhysicalSampleIdentity> {
        // Numerical vectors obey this sample's bound, not the small metadata identity
        // envelope. Bind all physical content before deriving the typed identity.
        let content = encode_contract(
            FREQUENCY_SAMPLE_PREFIX,
            "structural-frequency-physical-sample-content",
            &(
                self.schema_version,
                self.scope,
                self.entry,
                &self.displacement,
                &self.reaction,
                &self.balance,
            ),
            FREQUENCY_SAMPLE_LIMITS,
        )?;
        StructuralFrequencyPhysicalSampleIdentity::derive(
            "structural-frequency-physical-sample",
            &runmat_execution::Digest::sha256(content),
        )
    }
}
