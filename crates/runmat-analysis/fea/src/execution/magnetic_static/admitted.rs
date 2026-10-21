//! Exact pre-quadrature plan rederivation; worker allocation remains the shared
//! runner's responsibility.
use super::*;
use crate::planning::resource::{
    build_magnetostatic_resource_plan, FeaResourceDemand, FeaResourcePlan,
    PreparedMagnetostaticResourceFacts,
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
        let expected = build_magnetostatic_resource_plan(
            self,
            mesh,
            PreparedMagnetostaticResourceFacts {
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
        let formulation = self.formulation(mesh)?;
        let input = crate::physics::magnetic::compile_canonical_input(
            &self.model,
            mesh,
            self.step_id.as_str(),
            self.relative_residual_target,
        )
        .map_err(invalid)?;
        let system = crate::physics::magnetic::MagneticSystemPlan::derive(&input.projection)
            .map_err(invalid)?;
        if system.identity().map_err(invalid)? != demand.system_plan_identity {
            return Err(invalid(
                "magnetic assembly support differs from admitted plan",
            ));
        }
        let response = crate::physics::magnetic::solve_full_grounded_volume_with_plan(
            &input.projection,
            &input.volumes,
            self.relative_residual_target,
            &system,
        )
        .map_err(invalid)?;
        MagnetostaticSolverResult::new(self, mesh, formulation, response)
    }
}
