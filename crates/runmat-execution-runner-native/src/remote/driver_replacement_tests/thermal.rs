use super::*;
use runmat_analysis_execution::{
    prepare_geometry_materialization_program, prepare_geometry_materialization_request,
    GeometryMaterializationPreparation, PreparedGeometrySource,
};

pub(super) fn write_authored_project(root: &std::path::Path) {
    std::fs::write(root.join("src/heated.fea"), SOURCE).unwrap();
    std::fs::create_dir_all(root.join("src/geometry")).unwrap();
    std::fs::write(root.join("src/geometry/box.brep"), BOX).unwrap();
}

pub(super) fn prepare_authored_root(
    root: &std::path::Path,
    revision: ProgramRevision,
    access: ArtifactAccess,
) -> crate::PreparedStructuralStudyRoot {
    let path = root.join("src/heated.fea");
    let checked = runmat_runtime::analysis::check_fea_path_in_root(root, &path).unwrap();
    assert!(checked.is_valid(), "{:#?}", checked.diagnostics);
    let resolved = checked
        .resolve_study("project/src/heated.fea", "heated_modes")
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
    let geometry_meshing =
        crate::prepare_geometry_meshing_root(crate::GeometryMeshingRootPreparation {
            program_revision: revision.clone(),
            artifact_access: access.clone(),
            geometry,
            geometry_source_objects: source.objects,
            meshing_request: fixture::coarse_box_request(),
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
        formulation_requirement: realized.formulation_requirement,
        numerics: runmat_analysis_fea::study::StructuralNumericalPlanningPolicy::production(),
    };
    crate::prepare_structural_study_root(crate::StructuralStudyRootPreparation {
        frozen_project_digest: Some(*revision.graph_digest()),
        program_revision: revision,
        artifact_access: access,
        authored_study_digest,
        study: realized,
        materials: resolved.material_catalog,
        planning,
        analysis_accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        input_materializations: Vec::new(),
        geometry_meshing,
    })
    .unwrap()
}

pub(super) fn assert_authored_result(result: &crate::ExactStructuralStudyResult) {
    let thermal = result
        .thermal
        .as_ref()
        .expect("cold typed root must retain thermal source");
    assert_eq!(result.compiled_study.as_ref(), Some(&thermal.compiled));
    assert_eq!(thermal.states.len(), 1);
    assert_eq!(
        thermal.states[0].state.state_id,
        runmat_analysis_core::StateId("heated".into())
    );
    let ExactStructuralStudyAnalysisResult::StructuralPrestressedModal(modal) = &result.analysis
    else {
        panic!("authored chain must finish with prestressed modal");
    };
    let source = modal
        .preload_result
        .linearization
        .as_ref()
        .unwrap()
        .thermal_source()
        .unwrap();
    assert_eq!(
        source.transition().source_state,
        thermal.states[0].state.identity
    );
    // The importer retains the authored template, not the worker invocation.
    // Rebind from the independently imported thermal state before checking the
    // result against the complete input; do not copy its own claimed transition.
    let mut bound = (*modal.preload_workload).clone();
    assert!(bound.state_input.is_none());
    assert!(modal
        .preload_result
        .validate_thermal_workload(&bound)
        .is_err());
    let transition = runmat_analysis_fea::execution::CanonicalStateTransition::bind_upstream(
        &bound.model,
        &bound.step_id,
        &thermal.states[0].state,
        runmat_analysis_core::StateUse::Prestress,
        Digest::from_bytes(*thermal.preparation().mesh.canonical_digest.bytes()),
    )
    .unwrap();
    assert_eq!(&transition, source.transition());
    bound.state_input = Some(
        runmat_analysis_fea::execution::StructuralLinearStateInput::ThermalPrestress(transition),
    );
    modal
        .preload_result
        .validate_thermal_workload(&bound)
        .unwrap();
    modal.modal_result.validate().unwrap();
}

const SOURCE: &str =
    include_str!("../../../../../docs/fea/examples/thermal-prestress-experimental.fea");
