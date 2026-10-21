//! Reauthenticate the complete governed CPU request before original numerical execution.
use super::*;
use crate::planning::resource::{
    build_fluid_steady_resource_plan, FeaResourceDemand, FeaResourcePlan, LocalExecutionPlan,
    PreparedFluidSteadyResourceFacts,
};
use runmat_meshing_core::SolverMeshArtifact;

impl FluidSteadyWorkload {
    pub fn execute_with_resource_plan(
        &self,
        mesh: &SolverMeshArtifact,
        plan: &FeaResourcePlan,
    ) -> AnalysisExecutionResult<FluidSteadySolverResult> {
        plan.validate_identity().map_err(invalid)?;
        let FeaResourceDemand::FluidSteadySolve(demand) = &plan.input.demand else {
            return Err(invalid(
                "steady fluid requires its own admitted physical CPU demand",
            ));
        };
        let output_set = plan
            .input
            .output_set
            .clone()
            .ok_or_else(|| invalid("fluid admitted output set missing"))?;
        let expected = build_fluid_steady_resource_plan(
            self,
            mesh,
            PreparedFluidSteadyResourceFacts {
                local_execution: LocalExecutionPlan::from_cpu_millicores(
                    plan.request.cpu_millicores,
                )
                .map_err(invalid)?,
                inputs: demand.inputs.clone(),
                result_output: demand.result_output.clone(),
                receipt_output: demand.receipt_output.clone(),
                output_set,
            },
        )
        .map_err(invalid)?;
        if &expected != plan {
            return Err(invalid(
                "fluid resource plan differs from original workload/material/finite-volume support",
            ));
        }
        self.execute(mesh)
    }
}
