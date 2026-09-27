use super::*;

#[path = "linear_pipeline/observation.rs"]
mod observation;
#[path = "linear_pipeline/publication.rs"]
mod publication;
#[path = "linear_pipeline/recovery.rs"]
mod recovery;
#[path = "linear_pipeline/solve.rs"]
mod solve;
#[path = "linear_pipeline/state.rs"]
mod state;

struct SolvedPipeline {
    solve: StructuralLinearWorkload,
    prepared: runmat_analysis_execution::PreparedStructuralLinearProgram,
    result: StructuralLinearWorkloadResult,
    constraint_plan: CanonicalStructuralConstraintPlan,
    solver_result: runmat_analysis_execution::PreparedAnalysisObjects,
    numerical_receipt: runmat_analysis_execution::PreparedAnalysisObjects,
    source_result: SolverResultIdentity,
}

struct RecoveredPipeline {
    host: AnalysisHostWorkload,
    inputs: Vec<runmat_analysis_execution::PreparedAnalysisObjects>,
    artifact: runmat_analysis_fea::results::field::RecoveredFieldArtifact,
}

struct ObservedPipeline {
    catalog: runmat_analysis_fea::results::observation::ObservationPublicationCatalog,
    assembly_plan: runmat_analysis_fea::planning::resource::FeaResourcePlan,
    assembly_attempt: ResourceUsageAttempt,
    partition_reduction_plans: Vec<runmat_analysis_fea::planning::resource::FeaResourcePlan>,
    partition_reduction_attempts: Vec<ResourceUsageAttempt>,
}

#[test]
fn native_kernel_executes_path_free_structural_linear_workload() {
    let solved = solve::execute();
    let recovered = recovery::execute(&solved);
    let observations = observation::execute(&solved, &recovered);
    publication::execute(&solved, &observations);
    state::execute(&solved);
}
