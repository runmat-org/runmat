//! Exact pre-quadrature plan rederivation; worker allocation remains the shared
//! runner's responsibility.
use super::*;
use crate::planning::resource::{
    build_magnetostatic_resource_plan_with_system, FeaResourceDemand, FeaResourcePlan,
    LocalExecutionPlan, PreparedMagnetostaticResourceFacts,
};
impl MagnetostaticWorkload {
    pub fn execute_with_resource_plan(
        &self,
        mesh: &SolverMeshArtifact,
        plan: &FeaResourcePlan,
    ) -> AnalysisExecutionResult<MagnetostaticSolverResult> {
        plan.validate_identity().map_err(invalid)?;
        let FeaResourceDemand::Magnetostatic(demand) = &plan.input.demand else {
            return Err(invalid("magnetic solve requires its own resource demand"));
        };
        let output_set = plan
            .input
            .output_set
            .clone()
            .ok_or_else(|| invalid("magnetic plan lacks complete output-set reservation"))?;
        // The admitted plan (whose demand binds the system identity) is rebuilt once; its compiled input and derived system are the
        // ones solved, so the projection and sparse support are derived exactly once.
        let (expected, input, system) = build_magnetostatic_resource_plan_with_system(
            self,
            mesh,
            PreparedMagnetostaticResourceFacts {
                local_execution: LocalExecutionPlan::from_cpu_millicores(
                    plan.request.cpu_millicores,
                )
                .map_err(invalid)?,
                inputs: demand.inputs.clone(),
                result_output: demand.result_output.clone(),
                receipt_output: demand.receipt_output.clone(),
                state_profiles_output: demand.state_profiles_output.clone(),
                output_set,
            },
        )
        .map_err(invalid)?;
        if &expected != plan {
            return Err(invalid(
                "magnetic resource plan differs from exact workload/topology/shared recurrence",
            ));
        }
        // The retained layout and native selections use the solve's own projection.
        let sampling = crate::results::field::SampledVolumeFieldTopology::derive_from_projection(
            mesh,
            &input.projection,
            crate::results::field::VolumeFieldSamplingRule::TetrahedronKeastFifteen,
        )
        .map_err(invalid)?;
        if system.plan.identity().map_err(invalid)? != demand.system_plan_identity {
            return Err(invalid(
                "magnetic assembly support differs from admitted plan",
            ));
        }
        let formulation = self.formulation_with_sampling(mesh, Some(&sampling))?;
        let response = crate::physics::magnetic::solve_grounded_volume_derived(
            &input.projection,
            &input.volumes,
            &input.prescribed_potential,
            self.relative_residual_target,
            system,
        )
        .map_err(super::magnetic_failure)?;
        drop(input);
        MagnetostaticSolverResult::new(self, mesh, formulation, sampling, response)
    }
}
