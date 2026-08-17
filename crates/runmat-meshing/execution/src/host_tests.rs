use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ExecutableForm;
use runmat_meshing_core::MeshingStageKind;

use crate::task_tests::Fixture;
use crate::{
    prepare_exact_geometry_input, prepare_exact_geometry_objects, MeshingHostWorkload,
    MESHING_HOST_TARGET_PROFILE,
};

#[test]
fn host_workload_has_a_stable_bounded_round_trip() {
    let fixture = Fixture::new(MeshingStageKind::SurfaceMesh);
    let host = host(&fixture);
    let bytes = host.canonical_bytes().unwrap();

    assert_eq!(
        MeshingHostWorkload::from_canonical_bytes(&bytes).unwrap(),
        host
    );

    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(MeshingHostWorkload::from_canonical_bytes(&trailing).is_err());
    assert!(MeshingHostWorkload::from_canonical_bytes(&bytes[..bytes.len() - 1]).is_err());
    let mut wrong_domain = bytes;
    wrong_domain[0] ^= 1;
    assert!(MeshingHostWorkload::from_canonical_bytes(&wrong_domain).is_err());
}

#[test]
fn program_form_contains_only_contracts_and_externalized_roots() {
    let fixture = Fixture::new(MeshingStageKind::SurfaceMesh);
    let host = host(&fixture);
    let request = host
        .program_request(revision(), std::slice::from_ref(&fixture.input))
        .unwrap();

    assert_eq!(request.artifact.form, ExecutableForm::MeshingWorkload);
    assert_eq!(request.recipe.target.profile, MESHING_HOST_TARGET_PROFILE);
    assert_eq!(request.arguments.len(), 1);
    assert_eq!(request.recipe.compile_options.len(), 1);
    assert!(request
        .recipe
        .compile_options
        .iter()
        .all(|option| option.starts_with("meshing-workload:")));
    assert!(matches!(
        request.arguments[0],
        runmat_execution::value::ValuePayload::Object(_)
    ));
    assert_eq!(
        MeshingHostWorkload::from_program_request(&request).unwrap(),
        host
    );
}

#[test]
fn program_request_rejects_tampering_and_wrong_input_authority() {
    let fixture = Fixture::new(MeshingStageKind::SurfaceMesh);
    let host = host(&fixture);
    let mut request = host
        .program_request(revision(), std::slice::from_ref(&fixture.input))
        .unwrap();
    request.artifact.executable_bytes.push(0);
    assert!(MeshingHostWorkload::from_program_request(&request).is_err());

    let mut wrong_root = fixture.input.clone();
    wrong_root.authorization_scope = "wrong-run".into();
    assert!(host
        .program_request(revision(), std::slice::from_ref(&wrong_root))
        .is_err());
}

#[test]
fn vm_facing_form_rejects_inline_geometry_arguments() {
    let fixture = Fixture::new(MeshingStageKind::SurfaceMesh);
    let host = host(&fixture);
    let mut request = host
        .program_request(revision(), std::slice::from_ref(&fixture.input))
        .unwrap();
    request.arguments[0] = runmat_execution::value::ValuePayload::Inline(Box::new(
        runmat_execution::value::InlineValue::Dense(runmat_execution::value::DenseValue {
            element_type: runmat_execution::value::ElementType::U8,
            shape: vec![3],
            little_endian_data: vec![1, 2, 3],
        }),
    ));

    assert!(MeshingHostWorkload::from_program_request(&request).is_err());
}

#[test]
fn exact_geometry_document_round_trips_and_binds_its_externalized_root() {
    let (document, topology, evaluators) = runmat_geometry_fixtures::exact_circle();
    let objects = prepare_exact_geometry_objects(
        document,
        topology,
        evaluators,
        None,
        None,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let document = objects.document.clone();
    let access = crate::MeshingArtifactAccess {
        authorization_scope: "exact-host-run".into(),
        encryption_context: Digest::sha256(b"exact-host-context"),
    };
    let input =
        prepare_exact_geometry_input(objects, access.clone(), ObjectInventoryLimits::default())
            .unwrap();
    let mut fixture = Fixture::new(MeshingStageKind::SurfaceMesh);
    fixture.context.artifact_access = access.clone();
    fixture.bind_exact_geometry(&document, input.root_input().clone());
    let host = MeshingHostWorkload::new(
        fixture.workload.clone(),
        fixture.identity.clone(),
        fixture.request.clone(),
        access,
        Some(document.clone()),
    )
    .unwrap();
    let bytes = host.canonical_bytes().unwrap();
    assert_eq!(
        MeshingHostWorkload::from_canonical_bytes(&bytes).unwrap(),
        host
    );
    host.program_request(revision(), std::slice::from_ref(input.root_input()))
        .unwrap();

    assert!(MeshingHostWorkload::new(
        fixture.workload.clone(),
        fixture.identity.clone(),
        fixture.request.clone(),
        fixture.context.artifact_access.clone(),
        None,
    )
    .is_err());
    let mut wrong_document = document;
    wrong_document.source.content_digest =
        runmat_geometry_core::GeometryDigest::from_bytes([88; 32]);
    assert!(MeshingHostWorkload::new(
        fixture.workload,
        fixture.identity,
        fixture.request,
        fixture.context.artifact_access,
        Some(wrong_document),
    )
    .is_err());
}

fn host(fixture: &Fixture) -> MeshingHostWorkload {
    MeshingHostWorkload::new(
        fixture.workload.clone(),
        fixture.identity.clone(),
        fixture.request.clone(),
        fixture.context.artifact_access.clone(),
        None,
    )
    .unwrap()
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"meshing-host-graph"),
        Digest::sha256(b"meshing-host-source"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"meshing-host-runtime-v2"),
            Digest::sha256(b"meshing-host-catalog-v2"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
}
