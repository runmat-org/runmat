//! Exact pre-assembly resource-plan rederivation, not scheduler allocation or placement.
use super::*;
use crate::planning::resource::{
    build_electrical_static_resource_plan, FeaResourceDemand, FeaResourcePlan, LocalExecutionPlan,
    PreparedElectricalStaticResourceFacts,
};

impl ElectricalStaticWorkload {
    /// Uses the supplied requested-heap plan only after rederiving its numerical authority.
    /// Worker allocation is a separate shared-execution gate; native dispatch stays closed.
    pub fn execute_with_resource_plan(
        &self,
        mesh: &SolverMeshArtifact,
        plan: &FeaResourcePlan,
    ) -> AnalysisExecutionResult<ElectricalStaticSolverResult> {
        plan.validate_identity().map_err(invalid)?;
        let FeaResourceDemand::ElectricalStatic(demand) = &plan.input.demand else {
            return Err(invalid("electrical solve requires its own resource demand"));
        };
        let output_set = plan
            .input
            .output_set
            .clone()
            .ok_or_else(|| invalid("electrical resource plan lacks output-set reservation"))?;
        let expected = build_electrical_static_resource_plan(
            self,
            mesh,
            PreparedElectricalStaticResourceFacts {
                local_execution: LocalExecutionPlan::from_cpu_millicores(
                    plan.request.cpu_millicores,
                )
                .map_err(invalid)?,
                inputs: demand.inputs.clone(),
                result_output: demand.result_output.clone(),
                receipt_output: demand.receipt_output.clone(),
                state_profiles_output: demand.state_profiles_output.clone(),
                output_set,
                machine_memory_bytes: demand.solver_memory_budget.0,
            },
        )
        .map_err(invalid)?;
        if &expected != plan {
            return Err(invalid(
                "electrical resource plan is not derived from this workload and mesh",
            ));
        }
        // All planned ceilings and exact topology/solver work have been checked before the
        // numerical owner can allocate a projection, property map or physical matrix.
        let formulation = self.formulation(mesh)?;
        let conductivity = self.element_conductivity(mesh)?;
        let response = crate::physics::electrical::solve_canonical_static_with_admission(
            &self.model,
            mesh,
            self.step_id.as_str(),
            self.relative_residual_target,
            Some(crate::physics::electrical::PlannedElectricalSolve {
                admission: &demand.solver_admission,
                direct: demand.direct_solve,
            }),
            conductivity.as_deref(),
            self.sample_point(),
        )
        .map_err(super::electrical_failure)?;
        ElectricalStaticSolverResult::new(self, mesh, formulation, response)
    }
}
