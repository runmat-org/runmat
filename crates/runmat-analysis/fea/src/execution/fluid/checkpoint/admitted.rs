//! Exact governed phase before original temporal execution.
use super::*;
use crate::planning::resource::{FeaResourceDemand, FeaResourcePlan, PreparedPhaseResourceFacts};
impl FluidTransientBlockWorkload {
    pub fn prepare_with_resource_plan<'a>(
        &'a self,
        mesh: &SolverMeshArtifact,
        plan: &FeaResourcePlan,
    ) -> AnalysisExecutionResult<PreparedFluidTransientContinuation<'a>> {
        plan.validate_identity().map_err(invalid)?;
        let FeaResourceDemand::FluidTransientBlock(demand) = &plan.input.demand else {
            return Err(invalid(
                "fluid transient needs its own original physical demand",
            ));
        };
        let prepared = self.prepare(mesh)?;
        let projection = crate::planning::resource::fluid_transient::projection_from_prepared(
            self, mesh, &prepared,
        )
        .map_err(invalid)?;
        let expected = crate::planning::resource::fluid_transient::plan_from_projection(
            self,
            projection,
            PreparedPhaseResourceFacts {
                inputs: demand.inputs.clone(),
                output: demand.output.clone(),
                output_set: plan
                    .input
                    .output_set
                    .clone()
                    .ok_or_else(|| invalid("fluid transient output-set budget missing"))?,
            },
            crate::planning::resource::LocalExecutionPlan::from_cpu_millicores(
                plan.request.cpu_millicores,
            )
            .map_err(invalid)?,
        )
        .map_err(invalid)?;
        if &expected != plan {
            return Err(invalid(
                "fluid transient admission differs from original source/checkpoint",
            ));
        }
        Ok(prepared)
    }
    pub fn execute_with_resource_plan(
        &self,
        mesh: &SolverMeshArtifact,
        plan: &FeaResourcePlan,
    ) -> AnalysisExecutionResult<FluidTransientCheckpoint> {
        self.prepare_with_resource_plan(mesh, plan)?.execute()
    }
}
