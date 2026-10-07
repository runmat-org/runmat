//! Recompile persistent authored constraints through their single shared authority.
use super::StructuralAdaptationIterationError as Error;
use crate::constraints::CanonicalStructuralConstraintPlan;
use runmat_analysis_core::AnalysisModel;
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralAdaptationConstraintLineage {
    pub authored_constraints_digest: Digest,
    pub source_plan_digest: Digest,
    pub target_plan_digest: Digest,
}

impl StructuralAdaptationConstraintLineage {
    pub(super) fn derive(
        model: &AnalysisModel,
        source: &SolverMeshArtifact,
        target: &SolverMeshArtifact,
    ) -> Result<Self, Error> {
        let convert =
            |e: crate::constraints::ConstraintPlanError| Error::InvalidConstraints(e.to_string());
        let source_plan =
            CanonicalStructuralConstraintPlan::compile(model, source).map_err(convert)?;
        let target_plan =
            CanonicalStructuralConstraintPlan::compile(model, target).map_err(convert)?;
        if source_plan.authored_constraints_digest != target_plan.authored_constraints_digest {
            return Err(Error::InvalidConstraints(
                "authored constraint intent changed".into(),
            ));
        }
        let lineage = Self {
            authored_constraints_digest: source_plan.authored_constraints_digest,
            source_plan_digest: source_plan.content_digest().map_err(convert)?,
            target_plan_digest: target_plan.content_digest().map_err(convert)?,
        };
        lineage.validate()?;
        Ok(lineage)
    }

    pub(super) fn validate(&self) -> Result<(), Error> {
        if [
            self.authored_constraints_digest,
            self.source_plan_digest,
            self.target_plan_digest,
        ]
        .iter()
        .any(|d| d.bytes() == &[0; 32])
            || self.source_plan_digest == self.target_plan_digest
        {
            return Err(Error::InvalidConstraints(
                "constraint lineage identities are invalid".into(),
            ));
        }
        Ok(())
    }
}
