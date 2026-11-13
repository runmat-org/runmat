use runmat_analysis_core::{
    AnalysisProcedure, ExactFormulationRealization, PhysicsFamily, SolutionSpace,
};
use serde::{de, Deserialize, Deserializer, Serialize};

use super::{admit_formulation, AdmittedFormulationReceipt, FormulationAdmissionRequest};

/// Exact, registry-issued identity of the formulation that produced an analysis result.
///
/// This is the sole persistable admission authority. Decoding re-runs the closed registry, so a
/// stored identity cannot claim a family or realization that is no longer admitted. Coarse
/// indexing facts are derived rather than serialized twice.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisRunIdentity {
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    solution_space: SolutionSpace,
    realization: ExactFormulationRealization,
}

impl AnalysisRunIdentity {
    pub const fn family(&self) -> PhysicsFamily {
        self.family
    }

    pub const fn procedure(&self) -> AnalysisProcedure {
        self.procedure
    }

    pub const fn solution_space(&self) -> SolutionSpace {
        self.solution_space
    }

    pub fn realization(&self) -> &ExactFormulationRealization {
        &self.realization
    }

    pub(crate) fn from_receipt(receipt: AdmittedFormulationReceipt) -> Self {
        let family = receipt.family();
        let procedure = receipt.procedure();
        let solution_space = receipt.solution_space();
        Self {
            family,
            procedure,
            solution_space,
            realization: receipt.realization,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AnalysisRunIdentityWire {
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    solution_space: SolutionSpace,
    realization: ExactFormulationRealization,
}

impl<'de> Deserialize<'de> for AnalysisRunIdentity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = AnalysisRunIdentityWire::deserialize(deserializer)?;
        let receipt = admit_formulation(FormulationAdmissionRequest {
            family: wire.family,
            procedure: wire.procedure,
            solution_space: wire.solution_space,
            realization: wire.realization,
        })
        .map_err(|rejection| de::Error::custom(rejection.to_string()))?;
        Ok(receipt.into_identity())
    }
}
