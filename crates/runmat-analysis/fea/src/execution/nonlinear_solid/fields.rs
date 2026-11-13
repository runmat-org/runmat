//! Original terminal fields. Displacement is never used to reconstruct elastic stress.
use super::*;
use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::{
        FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, FieldTopologyIdentity,
    },
    post::fields::{
        structural_static_observable_field_definitions, take_structural_observable_fields,
    },
    results::field::{
        RecoveredFieldArtifact, RecoveredFieldDefinition, StructuralRecoveredFieldTopology,
    },
};
use runmat_meshing_core::SolverMeshArtifact;

pub fn nonlinear_native_field_definitions(
    request: &FieldRecoveryRequest,
) -> AnalysisExecutionResult<Vec<RecoveredFieldDefinition>> {
    request.validate()?;
    let available = structural_static_observable_field_definitions().map_err(invalid)?;
    request
        .plan
        .requirements
        .iter()
        .map(|r| {
            if r.recovery != runmat_analysis_core::RecoveryPolicy::Native {
                return Err(invalid(
                    "original J2 field publication currently requires native final fields",
                ));
            }
            available
                .iter()
                .find(|d| {
                    d.quantity() == r.quantity
                        && d.location() == r.output_location
                        && d.value_shape() == &r.value_shape
                        && d.entity_domain() == r.entity_domain
                })
                .cloned()
                .ok_or_else(|| invalid("original J2 field definition is unavailable"))
        })
        .collect()
}

impl NonlinearSolidBlockArtifact {
    pub fn native_field_request(
        &self,
        workload: &NonlinearSolidWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<FieldRecoveryRequest> {
        self.validate()?;
        workload.validate()?;
        crate::pipeline::nonlinear::require_nonlinear_terminal_field_outputs(
            &workload.model.sequence.steps[0],
        )
        .map_err(invalid)?;
        if !self.complete
            || self.request.workload != workload.identity()?
            || self.request.topology != Digest::from_bytes(*mesh.canonical_digest.bytes())
        {
            return Err(invalid(
                "original terminal field source differs from workload, mesh or complete state",
            ));
        }
        let topology = StructuralRecoveredFieldTopology::derive(&workload.model, Some(mesh))
            .map_err(invalid)?;
        let constraints =
            CanonicalStructuralConstraintPlan::compile_for_model(&workload.model, Some(mesh))
                .map_err(invalid)?;
        let request = FieldRecoveryRequest::new(
            workload.producer_identity()?,
            FieldRecoverySource::SolverResult(self.source_result_identity()?),
            FieldTopologyIdentity::Structural {
                topology: topology.identity,
            },
            FieldRecoveryPlan::resolve_structural(
                &workload.model.sequence.steps[0],
                topology,
                constraints.groups.len(),
            )?,
        )?;
        nonlinear_native_field_definitions(&request)?;
        Ok(request)
    }

    pub fn recover_native_fields(
        &self,
        workload: &NonlinearSolidWorkload,
        mesh: &SolverMeshArtifact,
        request: &FieldRecoveryRequest,
        additional_retained_bytes: u64,
    ) -> AnalysisExecutionResult<RecoveredFieldArtifact> {
        if *request != self.native_field_request(workload, mesh)? {
            return Err(invalid("original J2 recovery request differs"));
        }
        let mut original = crate::pipeline::nonlinear::recover_original_nonlinear_block(
            workload,
            mesh,
            self,
            additional_retained_bytes,
        )
        .map_err(invalid)?;
        let mut fields =
            take_structural_observable_fields(&mut original.run.fields).map_err(invalid)?;
        if let Some(constraints) = original.run.structural_constraint_result.as_deref() {
            fields.extend(
                crate::post::recover_constraint_group_observable_fields(constraints)
                    .map_err(invalid)?,
            );
        }
        let definitions = nonlinear_native_field_definitions(request)?;
        fields.retain(|f| definitions.contains(f.definition()));
        request
            .plan
            .validate_definitions(fields.iter().map(|f| f.definition().clone()))?;
        let mut topology = StructuralRecoveredFieldTopology::derive(&workload.model, Some(mesh))
            .map_err(invalid)?;
        topology.entity_counts.constraint_groups = original
            .run
            .structural_constraint_result
            .as_ref()
            .map_or(0, |c| c.groups.len());
        RecoveredFieldArtifact::new(
            request.identity(),
            request.source,
            topology.identity.discretization_digest(),
            topology.entity_counts,
            fields,
        )
        .map_err(invalid)
    }
}
