use std::collections::BTreeSet;

use runmat_execution::host::{ExecutionHostRequirement, EXECUTION_HOST_SCHEMA_VERSION};
use runmat_execution::security::ExecutionTrustTier;
use runmat_types::{CapabilityRequirement, CapabilitySet, ExecutionStackRequirement};

use super::{ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramTargetCohort};
use crate::ArtifactResult;

impl ProgramArtifact {
    /// Derive scheduler admission requirements from the immutable compiled
    /// product. Callers may narrow trust policy, but may not invent runtime,
    /// target, capability, interop, or stack facts.
    pub fn execution_host_requirement(
        &self,
        recipe: &ProgramBuildRecipe,
        permitted_trust_tiers: BTreeSet<ExecutionTrustTier>,
    ) -> ArtifactResult<ExecutionHostRequirement> {
        self.validate_against(recipe)?;
        let mut capabilities = match self.executable_unit()? {
            Some(envelope) => {
                if envelope.manifest.interop != recipe.interop {
                    return Err(crate::ArtifactError::Identity(
                        "executable interop manifest differs from its build recipe".into(),
                    ));
                }
                envelope.manifest.capabilities.clone()
            }
            None => CapabilitySet::default(),
        };
        let interop = recipe.interop.clone();
        capabilities.0.insert(CapabilityRequirement::HostRuntime);
        if !interop.adapters.is_empty() || !interop.foreign_types.is_empty() {
            capabilities.0.insert(CapabilityRequirement::ForeignRuntime);
        }
        if self.form == ExecutableForm::NativeObjectV1 {
            capabilities.0.insert(CapabilityRequirement::NativeCode);
        }
        let adapter_stack = interop
            .adapters
            .iter()
            .map(|requirement| requirement.execution_stack)
            .max()
            .unwrap_or_default();
        let execution_stack = if self.form == ExecutableForm::NativeObjectV1 {
            ExecutionStackRequirement::Process
        } else {
            adapter_stack
        };
        let requirement = ExecutionHostRequirement {
            schema_version: EXECUTION_HOST_SCHEMA_VERSION,
            environment: recipe.program_revision.environment(),
            native_target: match self.target.cohort {
                ProgramTargetCohort::Portable => None,
                ProgramTargetCohort::Native => self.target.native.clone(),
            },
            capabilities,
            interop,
            execution_stack,
            permitted_trust_tiers,
        };
        requirement
            .validate()
            .map_err(|error| crate::ArtifactError::Invalid(error.to_string()))?;
        Ok(requirement)
    }
}
