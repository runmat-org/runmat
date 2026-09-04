use runmat_analysis_core::{
    AnalysisProcedure, ExactFormulationRealization, ExecutionFallbackPolicy, FormulationMaturity,
    FormulationRequirement, PhysicsFamily, SolutionSpace,
};
use serde::{de, Deserialize, Deserializer, Serialize};

use super::{admit_formulation, AdmittedFormulationReceipt, FormulationAdmissionRequest};

/// Exact, registry-issued identity of the formulation that produced an analysis result.
///
/// This is the sole persistable admission authority. Decoding re-runs the closed registry, so a
/// stored identity cannot claim a family, realization, or qualification that is no longer
/// admitted. Coarse indexing and fallback facts are derived rather than serialized twice.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisRunIdentity {
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    solution_space: SolutionSpace,
    realization: ExactFormulationRealization,
    qualification: FormulationMaturity,
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

    pub const fn qualification(&self) -> FormulationMaturity {
        self.qualification
    }

    pub const fn fallback_policy(&self) -> ExecutionFallbackPolicy {
        fallback_policy_for(self.qualification)
    }

    pub fn permits_fallback(&self) -> bool {
        self.qualification == FormulationMaturity::Experimental
            && self.fallback_policy() == ExecutionFallbackPolicy::AllowExperimental
    }

    pub(crate) fn from_receipt(receipt: AdmittedFormulationReceipt) -> Self {
        let family = receipt.family();
        let procedure = receipt.procedure();
        let solution_space = receipt.solution_space();
        let qualification = receipt.qualification();
        Self {
            family,
            procedure,
            solution_space,
            realization: receipt.realization,
            qualification,
        }
    }
}

const fn fallback_policy_for(qualification: FormulationMaturity) -> ExecutionFallbackPolicy {
    match qualification {
        FormulationMaturity::Production => ExecutionFallbackPolicy::Forbid,
        FormulationMaturity::Experimental => ExecutionFallbackPolicy::AllowExperimental,
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AnalysisRunIdentityWire {
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    solution_space: SolutionSpace,
    realization: ExactFormulationRealization,
    qualification: FormulationMaturity,
}

impl<'de> Deserialize<'de> for AnalysisRunIdentity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = AnalysisRunIdentityWire::deserialize(deserializer)?;
        let requirement = match wire.qualification {
            FormulationMaturity::Production => FormulationRequirement::Production,
            FormulationMaturity::Experimental => FormulationRequirement::Experimental,
        };
        let receipt = admit_formulation(FormulationAdmissionRequest {
            family: wire.family,
            procedure: wire.procedure,
            solution_space: wire.solution_space,
            requirement,
            realization: wire.realization,
        })
        .map_err(|rejection| de::Error::custom(rejection.to_string()))?;
        ensure_qualification_matches(wire.qualification, receipt.qualification())
            .map_err(de::Error::custom)?;
        Ok(receipt.into_identity())
    }
}

fn ensure_qualification_matches(
    claimed: FormulationMaturity,
    registered: FormulationMaturity,
) -> Result<(), &'static str> {
    if claimed == registered {
        Ok(())
    } else {
        Err("persisted formulation qualification disagrees with the current registry")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_policy_projection_supports_future_qualification() {
        assert_eq!(
            fallback_policy_for(FormulationMaturity::Experimental),
            ExecutionFallbackPolicy::AllowExperimental
        );
        assert_eq!(
            fallback_policy_for(FormulationMaturity::Production),
            ExecutionFallbackPolicy::Forbid
        );
    }

    #[test]
    fn decode_rejects_both_qualification_mismatch_directions() {
        assert!(ensure_qualification_matches(
            FormulationMaturity::Experimental,
            FormulationMaturity::Production,
        )
        .is_err());
        assert!(ensure_qualification_matches(
            FormulationMaturity::Production,
            FormulationMaturity::Experimental,
        )
        .is_err());
    }
}
