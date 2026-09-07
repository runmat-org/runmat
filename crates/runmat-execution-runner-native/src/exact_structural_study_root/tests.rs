use super::*;
use runmat_analysis_fea::execution::StructuralLinearWorkload;
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
use runmat_analysis_fea::ComputeBackend;
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::ArtifactAccess;
use runmat_geometry_io::ExactCadImportOptions;
use runmat_meshing_core::{
    resolve_meshing_request, GeometryTolerancePolicy, MeshingRequestSettings,
};

pub(crate) fn revision() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"graph"),
        Digest::sha256(b"source"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
}

pub(crate) fn access(authority: &str) -> ArtifactAccess {
    ArtifactAccess {
        authorization_scope: authority.into(),
        encryption_context: Digest::sha256(authority.as_bytes()),
    }
}

fn workload() -> StructuralLinearWorkload {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let step_id = model.sequence.steps[0].step_id.clone();
    StructuralLinearWorkload {
        schema_version: 1,
        model,
        step_id,
        backend: ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        require_solver_mesh_for_solid: false,
        thermo_mechanical_context: None,
        electro_thermal_context: None,
        initial_guess_transition: None,
    }
}

fn workloads(
) -> std::collections::BTreeMap<runmat_analysis_core::ProcedureStepId, StructuralLinearWorkload> {
    let workload = workload();
    std::collections::BTreeMap::from([(workload.step_id.clone(), workload)])
}

pub(crate) fn prepared(
    authority: &str,
    expected_source: Digest,
    authored_study_digest: Digest,
) -> PreparedExactStructuralStudyRoot {
    let source = b"portable exact CAD source";
    let revision = revision();
    let access = access(authority);
    let import_options = ExactCadImportOptions::default();
    let meshing_request = resolve_meshing_request(
        GeometryTolerancePolicy {
            source_tolerance_m: 0.0,
            absolute_floor_m: 1.0e-12,
            model_relative_term: 1.0e-12,
            requested_deviation_m: 1.0e-4,
            maximum_healing_displacement_m: 1.0e-6,
        },
        MeshingRequestSettings::default(),
    )
    .unwrap();
    let exact_meshing = crate::prepare_exact_meshing_root(crate::ExactMeshingRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        source_name: "part.brep",
        source_bytes: source,
        import_options,
        contact_definitions: Vec::new(),
        meshing_request,
        preferred_edges_per_partition: 8,
        preferred_faces_per_partition: 8,
        maximum_chunk_bytes: 8,
    })
    .unwrap();
    prepare_exact_structural_study_root(ExactStructuralStudyRootPreparation {
        program_revision: revision,
        artifact_access: access,
        authored_study_digest,
        frozen_project_digest: Some(Digest::sha256(b"project")),
        geometry_source_digest: expected_source,
        workloads: workloads(),
        signal_materializations: Vec::new(),
        exact_meshing,
    })
    .unwrap()
}

#[test]
fn whole_study_root_is_path_free_and_reuses_fixed_programs() {
    let source_digest = Digest::sha256(b"portable exact CAD source");
    let first = prepared("run-a", source_digest, Digest::sha256(b"study"));
    let second = prepared("run-b", source_digest, Digest::sha256(b"study"));
    assert_eq!(
        first.required_programs().unwrap(),
        second.required_programs().unwrap()
    );
    assert_eq!(first.request().recipe, second.request().recipe);
    assert_ne!(first.request().arguments, second.request().arguments);
    assert_eq!(first.required_programs().unwrap().len(), 4);
    assert_eq!(
        first.transport_objects().unwrap().len(),
        first.objects().len()
    );

    let directory = tempfile::tempdir().unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        directory.path(),
        64 * 1024 * 1024,
    )
    .unwrap();
    first.write_to(&mut store).unwrap();
    let decoded = decode_exact_structural_study_root(first.request(), &store).unwrap();
    assert_eq!(decoded.geometry_source_digest, source_digest);
    assert_eq!(decoded.workloads, workloads());
    assert_eq!(decoded.artifact_access, access("run-a"));
}

#[test]
fn whole_study_root_rejects_a_substituted_source_identity() {
    let prepared = prepared(
        "run",
        Digest::sha256(b"different source"),
        Digest::sha256(b"study"),
    );
    let directory = tempfile::tempdir().unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        directory.path(),
        64 * 1024 * 1024,
    )
    .unwrap();
    prepared.write_to(&mut store).unwrap();
    assert!(decode_exact_structural_study_root(prepared.request(), &store).is_err());
}
