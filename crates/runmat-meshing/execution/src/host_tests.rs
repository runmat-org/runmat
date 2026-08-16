use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::ExecutableForm;
use runmat_meshing_core::MeshingStageV2;

use crate::task_tests::Fixture;
use crate::{MeshingHostWorkloadV2, MESHING_HOST_TARGET_PROFILE};

#[test]
fn host_workload_has_a_stable_bounded_round_trip() {
    let fixture = Fixture::new(MeshingStageV2::SurfaceMesh);
    let host = host(&fixture);
    let bytes = host.canonical_bytes().unwrap();

    assert_eq!(
        MeshingHostWorkloadV2::from_canonical_bytes(&bytes).unwrap(),
        host
    );

    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(MeshingHostWorkloadV2::from_canonical_bytes(&trailing).is_err());
    assert!(MeshingHostWorkloadV2::from_canonical_bytes(&bytes[..bytes.len() - 1]).is_err());
    let mut wrong_domain = bytes;
    wrong_domain[0] ^= 1;
    assert!(MeshingHostWorkloadV2::from_canonical_bytes(&wrong_domain).is_err());
}

#[test]
fn program_form_contains_only_contracts_and_externalized_roots() {
    let fixture = Fixture::new(MeshingStageV2::SurfaceMesh);
    let host = host(&fixture);
    let request = host
        .program_request(revision(), std::slice::from_ref(&fixture.input))
        .unwrap();

    assert_eq!(request.artifact.form, ExecutableForm::MeshingWorkloadV2);
    assert_eq!(request.recipe.target.profile, MESHING_HOST_TARGET_PROFILE);
    assert_eq!(request.arguments.len(), 1);
    assert!(matches!(
        request.arguments[0],
        runmat_execution::value::ValuePayload::Object(_)
    ));
    assert_eq!(
        MeshingHostWorkloadV2::from_program_request(&request).unwrap(),
        host
    );
}

#[test]
fn program_request_rejects_tampering_and_wrong_input_authority() {
    let fixture = Fixture::new(MeshingStageV2::SurfaceMesh);
    let host = host(&fixture);
    let mut request = host
        .program_request(revision(), std::slice::from_ref(&fixture.input))
        .unwrap();
    request.artifact.executable_bytes.push(0);
    assert!(MeshingHostWorkloadV2::from_program_request(&request).is_err());

    let mut wrong_root = fixture.input.clone();
    wrong_root.authorization_scope = "wrong-run".into();
    assert!(host
        .program_request(revision(), std::slice::from_ref(&wrong_root))
        .is_err());
}

#[test]
fn vm_facing_form_rejects_inline_geometry_arguments() {
    let fixture = Fixture::new(MeshingStageV2::SurfaceMesh);
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

    assert!(MeshingHostWorkloadV2::from_program_request(&request).is_err());
}

fn host(fixture: &Fixture) -> MeshingHostWorkloadV2 {
    MeshingHostWorkloadV2::new(
        fixture.workload.clone(),
        fixture.identity.clone(),
        fixture.request.clone(),
        fixture.context.artifact_access.clone(),
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
