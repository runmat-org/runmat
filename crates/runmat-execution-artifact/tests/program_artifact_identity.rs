#[path = "support/executable_unit.rs"]
mod executable_unit_support;
mod support;

use runmat_execution_artifact::{
    ExecutableForm, ExecutionBundleBuilder, NativeObjectPayload, ProgramArtifact,
};

fn cuda_requirement(
    minimum_allocation_bytes: u64,
) -> runmat_execution::resource::AcceleratorRequest {
    use runmat_execution::resource::{
        AcceleratorClass, AcceleratorFeature, AcceleratorProvider, AcceleratorProviderId,
        AcceleratorProviderVersion,
    };

    runmat_execution::resource::AcceleratorRequest {
        class: AcceleratorClass::new("gpu").unwrap(),
        count: 1,
        minimum_allocation_bytes,
        provider: Some(AcceleratorProvider {
            id: AcceleratorProviderId::new("runmat.cuda").unwrap(),
            version: AcceleratorProviderVersion::new("1.0.0").unwrap(),
            abi_fingerprint: runmat_execution::Digest::sha256(b"cuda-native-v1"),
        }),
        required_features: [AcceleratorFeature::Compute].into_iter().collect(),
    }
}

#[test]
fn native_object_artifact_retains_bounded_digest_checked_parts() {
    let (_temp, _project, revision) = support::frozen_project();
    let mut recipe = support::recipe(revision);
    recipe.target = runmat_execution_artifact::ProgramTarget::native(
        "native-object-v1-test",
        runmat_execution_artifact::NativeTargetIdentity {
            architecture: runmat_execution::host::NativeArchitecture::Aarch64,
            operating_system: runmat_execution::host::NativeOperatingSystem::Macos,
            pointer_width: 64,
            abi: runmat_execution::host::NativeAbi::new("runmat-native-abi-v1").unwrap(),
            object_format: runmat_execution::host::NativeObjectFormat::MachO,
        },
    );
    let payload = NativeObjectPayload::new(
        runmat_execution::host::NativeObjectFormat::MachO,
        br#"{"schema_version":1}"#.to_vec(),
        b"relocatable-object".to_vec(),
    )
    .unwrap();
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::NativeObjectV1,
        payload.to_canonical_bytes().unwrap(),
    )
    .unwrap();
    assert_eq!(artifact.native_object().unwrap(), Some(payload));

    let mut tampered = artifact;
    let last = tampered.executable_bytes.last_mut().unwrap();
    *last ^= 1;
    assert!(tampered.validate_against(&recipe).is_err());
}

#[test]
fn recipe_and_materialized_artifact_have_distinct_exact_identities() {
    let (_temp, project, revision) = support::frozen_project();
    let bundle = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_materialized_program(
            support::recipe(revision),
            ExecutableForm::InterpreterBytecodeV1,
            b"bytecode-a".to_vec(),
        )
        .build()
        .unwrap();
    let recipe = &bundle.manifest.recipes[0];
    let artifact = &bundle.manifest.artifacts[0];
    assert_ne!(recipe.id().unwrap().0, artifact.id.0);
    artifact.validate_against(recipe).unwrap();

    let changed = ProgramArtifact::materialize(
        recipe,
        ExecutableForm::InterpreterBytecodeV1,
        b"bytecode-b".to_vec(),
    )
    .unwrap();
    assert_ne!(changed.id, artifact.id);
}

#[test]
fn accelerator_requirements_are_part_of_recipe_and_artifact_identity() {
    let (_temp, _project, revision) = support::frozen_project();
    let plain = support::recipe(revision);
    let mut accelerated = plain.clone();
    accelerated.accelerators = vec![cuda_requirement(1 << 20)];

    assert_ne!(plain.id().unwrap(), accelerated.id().unwrap());
    let plain_artifact = ProgramArtifact::materialize(
        &plain,
        ExecutableForm::InterpreterBytecodeV1,
        b"same-program".to_vec(),
    )
    .unwrap();
    let accelerated_artifact = ProgramArtifact::materialize(
        &accelerated,
        ExecutableForm::InterpreterBytecodeV1,
        b"same-program".to_vec(),
    )
    .unwrap();
    assert_ne!(plain_artifact.id, accelerated_artifact.id);
}

#[test]
fn scheduler_resources_may_strengthen_but_not_erase_recipe_requirements() {
    let (_temp, _project, revision) = support::frozen_project();
    let mut recipe = support::recipe(revision);
    recipe.accelerators = vec![cuda_requirement(1 << 20)];
    let mut resources = runmat_execution::resource::ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: 1 << 30,
        scratch_bytes: 1 << 30,
        max_wall_millis: 60_000,
        max_artifact_bytes: 1 << 30,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: Vec::new(),
        required_capabilities: Default::default(),
    };
    assert!(recipe.validate_resource_request(&resources).is_err());

    resources.accelerators = vec![cuda_requirement(2 << 20)];
    recipe.validate_resource_request(&resources).unwrap();
}

#[test]
fn artifact_tampering_and_revision_mismatch_are_rejected() {
    let (_temp, project, revision) = support::frozen_project();
    let mut bundle = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_materialized_program(
            support::recipe(revision),
            ExecutableForm::InterpreterBytecodeV1,
            b"bytecode".to_vec(),
        )
        .build()
        .unwrap();
    bundle.manifest.artifacts[0].executable_bytes.push(0);
    assert!(bundle.validate().is_err());

    let mut wrong_revision = support::recipe(support::revision_for(&project));
    wrong_revision.program_revision = runmat_execution::ProgramRevision::new(
        runmat_execution::Digest::sha256(b"wrong"),
        wrong_revision.program_revision.source_digest().to_owned(),
        wrong_revision.program_revision.environment(),
    )
    .unwrap();
    assert!(
        ExecutionBundleBuilder::native(&project, support::revision_for(&project))
            .unwrap()
            .with_recipe(wrong_revision)
            .build()
            .is_err()
    );
}

#[test]
fn executable_interop_must_match_the_recipe() {
    let (_temp, _project, revision) = support::frozen_project();
    let mut recipe = executable_unit_support::recipe(support::recipe(revision.clone()));
    recipe
        .interop
        .adapters
        .push(runmat_types::ForeignAdapterRequirement {
            adapter: runmat_types::ForeignAdapterId::new("java").unwrap(),
            minimum_version: 1,
            capabilities: runmat_types::CapabilitySet::default(),
            execution_stack: runmat_types::ExecutionStackRequirement::Process,
            artifact_identities: vec![runmat_types::ForeignArtifactIdentity::new(
                "java:v1:fixture",
            )
            .unwrap()],
        });
    recipe.interop.validate().unwrap();
    let error = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::ExecutableUnitV3,
        executable_unit_support::bytes(revision),
    )
    .expect_err("recipe and executable interop must converge");
    assert!(error.to_string().contains("interop manifest"));
}
