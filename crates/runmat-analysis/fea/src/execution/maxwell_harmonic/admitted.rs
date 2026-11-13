//! Exact worker-side plan rederivation before quadrature or operator allocation.
use super::*;
use crate::planning::resource::{
    build_maxwell_harmonic_resource_plan, FeaResourceDemand, FeaResourcePlan, LocalExecutionPlan,
    PreparedMaxwellHarmonicResourceFacts,
};
impl MaxwellHarmonicWorkload {
    pub fn execute_with_resource_plan(
        &self,
        mesh: &SolverMeshArtifact,
        plan: &FeaResourcePlan,
    ) -> AnalysisExecutionResult<MaxwellHarmonicSolverResult> {
        plan.validate_identity().map_err(invalid)?;
        let FeaResourceDemand::MaxwellHarmonic(demand) = &plan.input.demand else {
            return Err(invalid(
                "Maxwell requires its own admitted complex resource demand",
            ));
        };
        let output_set = plan
            .input
            .output_set
            .clone()
            .ok_or_else(|| invalid("Maxwell requires complete admitted output set"))?;
        let expected = build_maxwell_harmonic_resource_plan(
            self,
            mesh,
            PreparedMaxwellHarmonicResourceFacts {
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
            return Err(invalid("Maxwell resource plan differs from original workload/topology/physical support/shared GMRES"));
        }
        let input = self.compile_input(mesh)?;
        let system =
            crate::physics::maxwell::MaxwellSystemPlan::derive(mesh, &input.grounded_faces)
                .map_err(invalid)?;
        if system.identity().map_err(invalid)? != demand.system_plan_identity {
            return Err(invalid(
                "Maxwell physical support differs from admitted system plan",
            ));
        }
        let formulation = self.formulation(mesh)?;
        let response = crate::physics::maxwell::solve_with_plan(
            mesh,
            &input,
            self.relative_residual_target,
            &system,
        )
        .map_err(invalid)?;
        MaxwellHarmonicSolverResult::new(self, mesh, formulation, input, response)
    }
}
