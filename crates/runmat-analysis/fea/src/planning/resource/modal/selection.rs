use super::*;

/// A modal workload's compiled constraints, observations and prepared output facts.
pub struct StructuralModalResourcePlanningInput<'a, 'model> {
    pub mesh: Option<&'a SolverMeshArtifact>,
    pub compilation: &'a crate::constraints::ConstraintCompilation<'model>,
    pub observations: &'a ResolvedObservationPlan,
    pub workload: &'a StructuralModalWorkload,
    pub prestress_state: Option<&'a CanonicalSemanticState>,
    pub accelerator_requests: &'a [AcceleratorRequest],
    pub prepared: PreparedStructuralModalResourceFacts,
}

/// The modal solve path a plan recorded (`None` for other plans).
pub fn recorded_modal_solve_path(plan: &FeaResourcePlan) -> Option<SubspaceSolvePath> {
    match &plan.input.demand {
        FeaResourceDemand::StructuralModalExtraction(demand) => {
            demand.operator.solve.subspace_solve_path()
        }
        _ => None,
    }
}

/// Plans a modal workload on the path `selection` names.
///
/// Path-selection law: with the effective ceiling min(authored resident budget, machine
/// memory when known), the shift-invert factorization is chosen when its whole plan (memory
/// and scratch) fits the ceiling; otherwise preconditioned LOBPCG when its plan fits;
/// otherwise [`ResourcePlanningError::ModalSolveCeiling`] names both requirements. The plan
/// records the path, and the worker's re-derivation and the solver follow the record.
pub fn build_structural_modal_resource_plan_selecting(
    input: StructuralModalResourcePlanningInput<'_, '_>,
    selection: ModalSolveSelection,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let StructuralModalResourcePlanningInput {
        mesh,
        compilation,
        observations,
        workload,
        prestress_state,
        accelerator_requests,
        prepared,
    } = input;
    let plan_path = |path: SubspaceSolvePath| {
        build_structural_modal_resource_plan_on_path(
            StructuralModalResourcePlanningInput {
                mesh,
                compilation,
                observations,
                workload,
                prestress_state,
                accelerator_requests,
                prepared: prepared.clone(),
            },
            path,
        )
    };
    let machine_memory_bytes = match selection {
        ModalSolveSelection::Recorded(path) => return plan_path(path),
        ModalSolveSelection::WithinCeiling {
            machine_memory_bytes,
        } => machine_memory_bytes,
    };
    let authored = workload
        .procedure_step()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
        .accuracy
        .resources
        .resident_byte_budget();
    let ceiling = machine_memory_bytes.map_or(authored, |machine| machine.min(authored));
    let fits = |plan: &FeaResourcePlan| {
        plan.request.memory_bytes <= ceiling && plan.request.scratch_bytes <= ceiling
    };
    let required = |planned: Result<FeaResourcePlan, ResourcePlanningError>| match planned {
        Ok(plan) if fits(&plan) => Ok(Ok(plan)),
        Ok(plan) => Ok(Err(plan
            .request
            .memory_bytes
            .max(plan.request.scratch_bytes))),
        Err(ResourcePlanningError::AccuracyCeiling {
            resource: super::super::RESIDENT_MEMORY_RESOURCE,
            required,
            ..
        }) => Ok(Err(required)),
        Err(error) => Err(error),
    };
    let factorization_resident_bytes =
        match required(plan_path(SubspaceSolvePath::ShiftInvertFactorization))? {
            Ok(plan) => return Ok(plan),
            Err(bytes) => bytes,
        };
    let lobpcg_resident_bytes = match required(plan_path(SubspaceSolvePath::PreconditionedLobpcg))?
    {
        Ok(plan) => return Ok(plan),
        Err(bytes) => bytes,
    };
    Err(ResourcePlanningError::ModalSolveCeiling {
        factorization_resident_bytes,
        lobpcg_resident_bytes,
        maximum: ceiling,
    })
}
