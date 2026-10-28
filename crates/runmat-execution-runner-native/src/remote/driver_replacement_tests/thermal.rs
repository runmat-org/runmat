use super::*;
use runmat_analysis_execution::{
    prepare_geometry_materialization_program, prepare_geometry_materialization_request,
    GeometryMaterializationPreparation, PreparedGeometrySource,
};
use runmat_execution_artifact::ProgramExecutionResponse;

#[path = "thermal/result.rs"]
mod result;
#[path = "thermal/source.rs"]
mod source;
pub(super) use result::{assert_branch, import_cold};
pub(super) use source::{Branch, Case};

pub(super) struct ColdReplay {
    pub store: crate::remote::object_transfer::RemoteObjectStore,
    pub response: ProgramExecutionResponse,
    pub expectation: crate::StructuralStudyResultExpectation,
    pub thermal: Box<runmat_analysis_execution::CompletedThermalStudy>,
}

impl ColdReplay {
    pub(super) fn reject_foreign_thermal(&self, donor: &Self) {
        crate::exact_structural_study_root::assert_foreign_thermal_substitution_rejected(
            &self.store,
            &self.response,
            &self.expectation,
            &donor.store,
            &donor.thermal,
        );
    }
}

pub(super) fn write_authored_project(root: &std::path::Path, case: Case) {
    std::fs::write(root.join("src/heated.fea"), case.source()).unwrap();
    std::fs::create_dir_all(root.join("src/geometry")).unwrap();
    std::fs::write(root.join("src/geometry/box.brep"), BOX).unwrap();
}

pub(super) fn prepare_authored_root(
    root: &std::path::Path,
    revision: ProgramRevision,
    access: ArtifactAccess,
    case: Case,
) -> crate::PreparedStructuralStudyRoot {
    let path = root.join("src/heated.fea");
    let checked = runmat_runtime::analysis::check_fea_path_in_root(root, &path).unwrap();
    assert!(checked.is_valid(), "{:#?}", checked.diagnostics);
    let resolved = checked
        .resolve_study("src/heated.fea", "heated_modes")
        .unwrap();
    let realized = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, &[])
        .unwrap();
    let source = PreparedGeometrySource::imported("box.brep", BOX, 1024, &access).unwrap();
    let geometry_request = prepare_geometry_materialization_request(
        resolved.geometry,
        realized.geometry.bindings.clone(),
        &source,
        crate::native_geometry_provider_profile().unwrap(),
        GeometryMaterializationPreparation::default(),
    )
    .unwrap();
    let geometry = prepare_geometry_materialization_program(
        geometry_request,
        revision.clone(),
        &access,
        &source.roots,
    )
    .unwrap();
    let mut meshing_request = fixture::coarse_box_request();
    meshing_request.element_order = case.order;
    let geometry_meshing =
        crate::prepare_geometry_meshing_root(crate::GeometryMeshingRootPreparation {
            program_revision: revision.clone(),
            artifact_access: access.clone(),
            geometry,
            geometry_source_objects: source.objects,
            meshing_request,
            preferred_edges_per_partition: 64,
            preferred_faces_per_partition: 64,
        })
        .unwrap();
    let authored_study_digest = runmat_analysis_fea::study::structural_study_source_digest(
        &realized,
        &resolved.material_catalog,
    )
    .unwrap();
    let planning = runmat_analysis_fea::study::StructuralStudyPlanningPolicy {
        backend: ComputeBackend::Cpu,
        numerics: runmat_analysis_fea::study::StructuralNumericalPlanningPolicy::production(),
    };
    crate::prepare_structural_study_root(crate::StructuralStudyRootPreparation {
        frozen_project_digest: Some(*revision.graph_digest()),
        program_revision: revision,
        artifact_access: access,
        authored_study_digest,
        study: realized,
        materials: resolved.material_catalog,
        planning: crate::StudyRootPlanning::Structural(planning),
        analysis_accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        memory_per_worker_bytes: None,
        accelerator_admission: Default::default(),
        input_materializations: Vec::new(),
        initial_fields: Vec::new(),
        geometry_meshing,
    })
    .unwrap()
}

#[test]
fn authored_matrix_variants_typecheck_with_explicit_physical_loads() {
    for case in Case::matrix() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join("src")).unwrap();
        write_authored_project(directory.path(), case);
        let checked = runmat_runtime::analysis::check_fea_path_in_root(
            directory.path(),
            &directory.path().join("src/heated.fea"),
        )
        .unwrap();
        assert!(checked.is_valid(), "{case:?}: {:#?}", checked.diagnostics);
        let resolved = checked
            .resolve_study("src/heated.fea", "heated_modes")
            .unwrap();
        let realized = resolved
            .study
            .realize_with_models(&resolved.models, &resolved.geometry, &[])
            .unwrap();
        let terminal = realized.sequence.steps.last().unwrap();
        let reduced = case.branch == Branch::ModalReducedHarmonic;
        assert_eq!(terminal.state_inputs.len(), if reduced { 2 } else { 1 });
        assert_eq!(
            terminal.solution_space == runmat_analysis_core::SolutionSpace::ModalReduced,
            reduced
        );
    }
}
