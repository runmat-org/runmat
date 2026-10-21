//! Original nonuniform history through admitted accuracy and terminated state publishers.
use super::*;
use runmat_analysis_execution::{
    import_accuracy, import_semantic_state, prepare_accuracy,
    prepare_nonlinear_solid_state_publication, AccuracyPreparation, ImportedNonlinearSolidBlock,
    NonlinearSolidStatePublicationPreparation,
};
use runmat_analysis_fea::{
    execution::{
        AccuracySourceKind, AnalysisNumericalObjectKind, CanonicalSemanticStatePayload,
        SequenceIdentity,
    },
    results::accuracy::{publication_gate, AccuracyStatus, PublicationDecision},
};

pub(super) fn verify(
    root: &std::path::Path,
    task: &PreparedNonlinearSolidBlockTask,
    source_workload: &NonlinearSolidWorkload,
    original: &ImportedNonlinearSolidBlock,
) {
    let policy = task.output_policy;
    let limits = policy.inventory_limits();
    let workload = task
        .inputs
        .iter()
        .find(|i| i.manifest.kind == AnalysisNumericalObjectKind::NonlinearSolidWorkload)
        .unwrap()
        .clone();
    let mesh = task
        .inputs
        .iter()
        .find(|i| i.manifest.kind == AnalysisNumericalObjectKind::Mesh)
        .unwrap()
        .clone();
    let accuracy = prepare_accuracy(AccuracyPreparation {
        solve_host: task.host.clone(),
        source_kind: AccuracySourceKind::NonlinearSolid,
        source_numerical_receipt: original.receipt_objects.clone(),
        source_observations: None,
        source_context_inputs: vec![workload.clone(), mesh, original.objects.clone()],
        sequence: SequenceIdentity::from_solver_contract(&source_workload.model.sequence).unwrap(),
        step: source_workload.model.sequence.steps[0].clone(),
        program_revision: task.program.request.recipe.program_revision.clone(),
        chunk_bytes: policy.chunk_bytes(),
        inventory_limits: limits,
    })
    .unwrap();
    let response = process::run_terminated(root, &accuracy.program, policy);
    let store = FilesystemObjectStore::open(root.join("store"), limits.max_object_bytes).unwrap();
    let cold = import_accuracy(&store, response, &accuracy, limits).unwrap();
    let report = cold.artifact.evaluation().report();
    assert_eq!(report.status, AccuracyStatus::Indeterminate);
    assert_eq!(
        report
            .contributions
            .iter()
            .find(|c| c.criterion_id == "newton")
            .unwrap()
            .status,
        AccuracyStatus::Satisfied
    );
    assert_eq!(
        report
            .contributions
            .iter()
            .find(|c| c.criterion_id == "mesh")
            .unwrap()
            .status,
        AccuracyStatus::Indeterminate
    );
    assert_eq!(
        publication_gate(&accuracy.request.plan, cold.artifact.evaluation())
            .unwrap()
            .decision,
        PublicationDecision::Block
    );
    drop(store);
    let declarations = &source_workload.model.sequence.steps[0].state_outputs;
    assert_eq!(declarations.len(), 3);
    for declaration in declarations {
        let prepared =
            prepare_nonlinear_solid_state_publication(NonlinearSolidStatePublicationPreparation {
                source_host: task.host.clone(),
                workload: workload.clone(),
                block: original.objects.clone(),
                declaration: declaration.clone(),
                program_revision: task.program.request.recipe.program_revision.clone(),
                output_policy: policy,
            })
            .unwrap();
        let response = process::run_terminated(root, &prepared.program, policy);
        let store =
            FilesystemObjectStore::open(root.join("store"), limits.max_object_bytes).unwrap();
        let state = import_semantic_state(&store, response, &prepared, limits).unwrap();
        assert_eq!(state.kind, declaration.kind);
        let CanonicalSemanticStatePayload::NonlinearSolid { state: history } = state.payload else {
            panic!("nonuniform material history must retain the original full block");
        };
        assert_eq!(history.block.as_ref(), &original.block);
        assert_eq!(history.workload.as_ref(), source_workload);
    }
}
