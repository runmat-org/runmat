use runmat_execution::identity::{AttemptId, ValueId, WorkerId};
use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION;
use runmat_execution::{
    Digest, ExecutionScopeId, JobId, OutputContract, PoolBackend, PoolId, ProgramEnvironment,
    ProgramExecutionAssignment, ProgramFunctionId, ProgramRevision, TaskId,
};

use super::*;
use crate::ExecutableForm;

#[test]
fn contract_failure_categories_survive_artifact_admission() {
    assert!(matches!(
        contract_error(runmat_execution::ContractError::Limit {
            field: "frames",
            limit: 4,
        }),
        ArtifactError::Limit(_)
    ));
    assert!(matches!(
        contract_error(runmat_execution::ContractError::Invalid {
            field: "source",
            reason: "out of map".into(),
        }),
        ArtifactError::Invalid(_)
    ));
}

pub(super) fn request() -> ProgramExecutionRequest {
    let revision = ProgramRevision::new(
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
    .unwrap();
    let recipe = ProgramBuildRecipe {
        schema_version: crate::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision,
        entrypoint: "7".into(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: "interpreter".into(),
        target: crate::ProgramTarget::portable("portable-test"),
        interop: runmat_types::InteropManifest::empty(),
        accelerators: Vec::new(),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        foreign_policy: None,
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::InterpreterBytecodeV2,
        b"program".to_vec(),
    )
    .unwrap();
    ProgramExecutionRequest {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe,
        artifact,
        callable: ProgramCallable::semantic(ProgramFunctionId(7), Some("test_function".into())),
        context: ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments: Vec::new(),
        requested_outputs: 1,
        environment: EnvironmentSnapshot::default(),
        console: None,
    }
}

#[test]
fn remote_direct_program_object_preserves_descriptor_wire_bytes() {
    let request = request();
    let descriptor = ProgramExecutionDescriptor {
        schema_version: request.schema_version,
        recipe: request.recipe,
        artifact: request.artifact,
        callable: request.callable,
        requested_outputs: request.requested_outputs,
        environment: request.environment,
        console: request.console,
    };
    let direct_bytes = serde_json::to_vec(&descriptor).unwrap();
    let envelope_bytes =
        serde_json::to_vec(&RemoteProgramObject::Direct(Box::new(descriptor.clone()))).unwrap();
    assert_eq!(envelope_bytes, direct_bytes);
    admit_remote_program_object_bytes(&envelope_bytes).unwrap();
    assert!(matches!(
        serde_json::from_slice::<RemoteProgramObject>(&envelope_bytes).unwrap(),
        RemoteProgramObject::Direct(recovered) if *recovered == descriptor
    ));
}

fn executable_unit_request() -> ProgramExecutionRequest {
    // Keep the frozen compatibility fixture untouched. This test needs a
    // current envelope so request validation can reach callable checks.
    let mut frozen: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../../runmat-execution/tests/fixtures/executable-unit-current.json"
    ))
    .unwrap();
    // This positive control has no dependency graph. Supply the current
    // required field explicitly; the historical bytes remain unchanged.
    frozen["manifest"]["dependencies"] = serde_json::json!([]);
    frozen["dependency_products"] = serde_json::json!([]);
    let mut envelope: runmat_execution::ExecutableUnitEnvelope =
        serde_json::from_value(frozen).unwrap();
    envelope.schema_version = runmat_execution::EXECUTABLE_UNIT_SCHEMA_VERSION;
    envelope.manifest.schema_version = runmat_execution::EXECUTABLE_UNIT_SCHEMA_VERSION;
    let bytes = envelope.canonical_bytes().unwrap();
    let envelope = runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(&bytes)
        .expect("current control envelope remains canonical");
    let identity = &envelope.manifest.identity;
    let recipe = ProgramBuildRecipe {
        schema_version: crate::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: identity.program.clone(),
        entrypoint: identity.entrypoint.clone(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: "interpreter".into(),
        target: crate::ProgramTarget::portable("portable-executable-unit-test"),
        interop: envelope.manifest.interop.clone(),
        accelerators: Vec::new(),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        foreign_policy: Some(
            crate::foreign_policy_object(
                br#"{"fixture":"direct-request","schema_version":1}"#.to_vec(),
            )
            .unwrap()
            .descriptor,
        ),
        expected_artifact_id: None,
    };
    let artifact =
        ProgramArtifact::materialize(&recipe, ExecutableForm::ExecutableUnitV4, bytes).unwrap();
    ProgramExecutionRequest {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe,
        artifact,
        callable: ProgramCallable::semantic(
            direct_executable_entrypoint(identity)
                .expect("frozen fixture has a callable entrypoint"),
            Some("main".into()),
        ),
        context: ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments: Vec::new(),
        requested_outputs: 1,
        environment: EnvironmentSnapshot::default(),
        console: None,
    }
}

fn source_unit_request() -> ProgramExecutionRequest {
    let mut request = executable_unit_request();
    let mut envelope = runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(
        &request.artifact.executable_bytes,
    )
    .unwrap();
    let text = "source = 1;";
    let map = runmat_execution::ExecutableSourceMapV2::new(vec![
        runmat_execution::ExecutableSourceRecordV2 {
            source_id: runmat_types::ProgramSourceId(0),
            owner_identity: "fixture".into(),
            relative_path: "source.m".into(),
            display_name: "source.m".into(),
            content_revision: Digest::sha256(text.as_bytes()),
            text: text.into(),
        },
    ])
    .unwrap();
    let bytes = serde_json::to_vec(&map).unwrap();
    envelope.manifest.revisions.source_map_schema = 2;
    let source = runmat_execution::ExecutableComponentKind::SourceMap;
    let descriptor = envelope
        .manifest
        .components
        .iter_mut()
        .find(|component| component.kind == source)
        .expect("source descriptor");
    *descriptor =
        runmat_execution::ExecutableComponentDescriptor::from_payload(source, 2, &bytes).unwrap();
    envelope
        .payloads
        .iter_mut()
        .find(|payload| payload.kind == source)
        .expect("source payload")
        .bytes = bytes;
    request.artifact = ProgramArtifact::materialize(
        &request.recipe,
        ExecutableForm::ExecutableUnitV4,
        envelope.canonical_bytes().unwrap(),
    )
    .unwrap();
    request
}

fn inline_argument(value: &str) -> ValuePayload {
    ValuePayload::Inline(Box::new(runmat_execution::value::InlineValue::String(
        value.into(),
    )))
}

fn source_exception(ordinal: u32) -> ValuePayload {
    use runmat_execution::value::{
        ExceptionCallFrame, ExceptionFailureOrigin, ExceptionGpuGatherRetry,
        ExceptionRuntimeMetadata, ExceptionValue, InlineValue,
    };
    ValuePayload::Inline(Box::new(InlineValue::Cell {
        shape: vec![1, 1],
        values: vec![ValuePayload::Inline(Box::new(InlineValue::Exception(
            ExceptionValue {
                identifier: "RunMat:test:Nested".into(),
                message: "nested source".into(),
                stack: vec!["source_function".into()],
                causes: Vec::new(),
                runtime: Some(ExceptionRuntimeMetadata {
                    span: None,
                    builtin: None,
                    task_id: None,
                    call_frames: vec![ExceptionCallFrame {
                        function: "source_function".into(),
                        source: runmat_execution::value::ExceptionFrameSource::Admitted(
                            runmat_types::ProgramSourceId(ordinal),
                        ),
                        span: None,
                    }],
                    call_frames_elided: 0,
                    call_stack: vec!["source_function @ source.m:1:1".into()],
                    phase: None,
                    gpu_gather_retry: ExceptionGpuGatherRetry::Never,
                    failure_origin: ExceptionFailureOrigin::LanguageEvaluation,
                }),
            },
        )))],
    }))
}

fn source_region_request() -> ProgramExecutionRequest {
    let region = runmat_types::ParallelRegionId(runmat_types::RegionId {
        function: runmat_types::ProgramFunctionId(7),
        ordinal: 3,
    });
    let mut request = request();
    request.callable = ProgramCallable::parallel_region(region);
    request.context = ProgramInvocationContext::ParallelTask {
        task: runmat_execution::ParallelTaskContext {
            region,
            chunk: runmat_execution::ParallelChunk {
                ordinal: 0,
                start: 0,
                len: 1,
            },
            randomness: runmat_execution::ParallelRandomnessContext::Inherit,
        },
    };
    request.recipe.entrypoint = request.callable.recipe_entrypoint();
    let text = "source_function = 1;";
    let map = runmat_execution::ExecutableSourceMapV2::new(vec![
        runmat_execution::ExecutableSourceRecordV2 {
            source_id: runmat_types::ProgramSourceId(0),
            owner_identity: "fixture".into(),
            relative_path: "source.m".into(),
            display_name: "source.m".into(),
            text: text.into(),
            content_revision: Digest::sha256(text.as_bytes()),
        },
    ])
    .unwrap();
    let payload = crate::InterpreterRegionSourcesV1::new(
        request.recipe.program_revision.clone(),
        map,
        b"opaque interpreter script".to_vec(),
    )
    .unwrap();
    request.artifact = ProgramArtifact::materialize(
        &request.recipe,
        ExecutableForm::InterpreterRegionSourcesV1,
        payload.encode().unwrap(),
    )
    .unwrap();
    request
}

#[test]
fn exact_artifact_map_bounds_nested_exception_arguments_outputs_and_failures() {
    let mut source_less = request();
    source_less.arguments.push(source_exception(0));
    assert!(source_less.validate().is_err());
    let source_less_output = ProgramExecutionResponse::Success {
        outputs: vec![source_exception(0)],
    };
    assert!(source_less_output.validate_against(&request()).is_err());

    let mut selected = source_region_request();
    selected.arguments.push(source_exception(0));
    selected
        .validate()
        .expect("last valid source ordinal is admitted");
    selected.arguments[0] = source_exception(1);
    assert!(selected.validate().is_err());
    selected.arguments[0] = source_exception(0);
    let valid = ProgramExecutionResponse::Success {
        outputs: vec![source_exception(0)],
    };
    valid.validate_against(&selected).unwrap();
    let invalid = ProgramExecutionResponse::Success {
        outputs: vec![source_exception(1)],
    };
    assert!(invalid.validate_against(&selected).is_err());
    let external_object = result_reference(b"source-bearing-result");
    ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![source_exception(0)],
        result_objects: vec![external_object.clone()],
    }
    .validate_against(&selected)
    .unwrap();
    assert!(ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![source_exception(1)],
        result_objects: vec![external_object],
    }
    .validate_against(&selected)
    .is_err());

    let region = runmat_types::ParallelRegionId(runmat_types::RegionId {
        function: runmat_types::ProgramFunctionId(7),
        ordinal: 3,
    });
    let scope_id = ExecutionScopeId::derive(&[b"source-bound-spmd-scope"]);
    let pool = runmat_execution::PoolHandle {
        id: PoolId::derive(&[b"source-bound-spmd-pool"]),
        scope_id,
        generation: 1,
    };
    let spmd_context = runmat_execution::SpmdTaskContext {
        gang: runmat_execution::GangHandle {
            id: runmat_execution::GangId::derive(&[b"source-bound-spmd-gang"]),
            scope_id,
            generation: 1,
            pool,
            labs: runmat_types::LabCount(1),
        },
        region,
        rank: runmat_types::LabRank(1),
    };
    let mut spmd = selected.clone();
    spmd.callable = ProgramCallable::spmd_region(region);
    spmd.context = ProgramInvocationContext::SpmdTask { task: spmd_context };
    spmd.recipe.entrypoint = spmd.callable.recipe_entrypoint();
    spmd.artifact = ProgramArtifact::materialize(
        &spmd.recipe,
        ExecutableForm::InterpreterRegionSourcesV1,
        selected.artifact.executable_bytes.clone(),
    )
    .unwrap();
    ProgramExecutionResponse::SpmdSuccess {
        outputs: vec![Some(runmat_execution::SpmdOutputValue::Value(
            source_exception(0),
        ))],
    }
    .validate_against(&spmd)
    .unwrap();
    assert!(ProgramExecutionResponse::SpmdSuccess {
        outputs: vec![Some(runmat_execution::SpmdOutputValue::Value(
            source_exception(1),
        ))],
    }
    .validate_against(&spmd)
    .is_err());

    let failure = ProgramRuntimeFailure {
        message: "source failure".into(),
        identifier: None,
        span: None,
        builtin: None,
        task_id: None,
        call_frames: vec![runmat_execution::ProgramCallFrame {
            function: "source_function".into(),
            program_source_id: Some(runmat_types::ProgramSourceId(1)),
            span: None,
            detached_diagnostic: None,
        }],
        call_frames_elided: 0,
        call_stack: vec!["source_function @ source.m:1:1".into()],
        phase: None,
    };
    assert!(ProgramExecutionResponse::RuntimeFailure {
        failure: failure.clone(),
    }
    .validate_against(&selected)
    .is_err());
    let mut valid_failure = failure;
    valid_failure.call_frames[0].program_source_id = Some(runmat_types::ProgramSourceId(0));
    ProgramExecutionResponse::RuntimeFailure {
        failure: valid_failure,
    }
    .validate_against(&selected)
    .unwrap();
}

#[test]
fn executable_unit_map_bounds_nested_arguments_outputs_and_failure_frames() {
    let mut legacy = executable_unit_request();
    legacy.arguments.push(source_exception(0));
    assert!(
        legacy.validate().is_err(),
        "schema-1 unit has no source map"
    );
    let legacy = executable_unit_request();
    assert!(ProgramExecutionResponse::Success {
        outputs: vec![source_exception(0)],
    }
    .validate_against(&legacy)
    .is_err());

    let mut selected = source_unit_request();
    selected.arguments.push(source_exception(0));
    selected.validate().expect("last valid unit source ordinal");
    selected.arguments[0] = source_exception(1);
    assert!(selected.validate().is_err());
    selected.arguments[0] = source_exception(0);
    ProgramExecutionResponse::Success {
        outputs: vec![source_exception(0)],
    }
    .validate_against(&selected)
    .unwrap();
    assert!(ProgramExecutionResponse::Success {
        outputs: vec![source_exception(1)],
    }
    .validate_against(&selected)
    .is_err());

    let failure = ProgramRuntimeFailure {
        message: "unit source failure".into(),
        identifier: None,
        span: None,
        builtin: None,
        task_id: None,
        call_frames: vec![runmat_execution::ProgramCallFrame {
            function: "source_function".into(),
            program_source_id: Some(runmat_types::ProgramSourceId(0)),
            span: None,
            detached_diagnostic: None,
        }],
        call_frames_elided: 0,
        call_stack: vec!["source_function @ source.m:1:1".into()],
        phase: None,
    };
    ProgramExecutionResponse::RuntimeFailure {
        failure: failure.clone(),
    }
    .validate_against(&selected)
    .unwrap();

    let mut detached = failure.clone();
    detached.call_frames[0].program_source_id = None;
    detached.call_frames[0].detached_diagnostic =
        Some(runmat_execution::ProgramDetachedDiagnosticSource {
            display_name: "retired.m".into(),
            line: Some(1),
            column: Some(1),
        });
    assert!(
        ProgramExecutionResponse::RuntimeFailure { failure: detached }
            .validate_against(&selected)
            .is_err()
    );
    let mut outside = failure;
    outside.call_frames[0].program_source_id = Some(runmat_types::ProgramSourceId(1));
    assert!(
        ProgramExecutionResponse::RuntimeFailure { failure: outside }
            .validate_against(&selected)
            .is_err()
    );
}

fn result_reference(bytes: &[u8]) -> ValueRef {
    let logical_digest = Digest::sha256(bytes);
    ValueRef {
        schema_version: VALUE_PAYLOAD_SCHEMA_VERSION,
        id: ValueId::derive(&[b"program-response-test", logical_digest.bytes()]),
        logical_digest,
        encoded_length: bytes.len() as u64,
        media_type: "application/vnd.runmat.test-object".into(),
        value_schema: "runmat.test-object.v1".into(),
        encryption_context: Digest::sha256(b"test-encryption-context"),
        kind: ValueRefKind::ResultObject,
        authorization_scope: "test-scope".into(),
        resident_fence: None,
    }
}

#[test]
fn exact_program_request_validates_every_identity_boundary() {
    request().validate().unwrap();
    let mut mismatched = request();
    mismatched.callable = ProgramCallable::semantic(ProgramFunctionId(8), None);
    assert!(mismatched.validate().is_err());
    let mut tampered = request();
    tampered.artifact.executable_bytes.push(0);
    assert!(tampered.validate().is_err());
}

#[test]
fn exact_program_request_rejects_unknown_schemas_and_output_drift() {
    let mut stale = request();
    stale.schema_version = PROGRAM_EXECUTION_REQUEST_SCHEMA_V5;
    assert!(stale.validate().is_err());
    let mut unknown = request();
    unknown.schema_version += 1;
    assert!(unknown.validate().is_err());
    let mut outputs = request();
    outputs.requested_outputs = 2;
    assert!(outputs.validate().is_err());
}

#[test]
fn frozen_v5_request_is_rejected_before_payload_deserialization() {
    let frozen = br#"{"schema_version":5,"arguments":[{"form":"inline","value":{"type":"output_list","value":[]}}]}"#;
    let error = admit_program_execution_request_bytes(frozen).unwrap_err();
    assert!(error
        .to_string()
        .contains(&format!("unsupported program execution request schema 5; expected {PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION}")));
    let old_failure_frame_contract = br#"{"schema_version":6,"arguments":[]}"#;
    assert!(
        admit_program_execution_request_bytes(old_failure_frame_contract)
            .unwrap_err()
            .to_string()
            .contains(&format!("unsupported program execution request schema 6; expected {PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION}"))
    );
    assert!(serde_json::from_slice::<ProgramExecutionRequest>(frozen).is_err());
}

#[test]
fn execution_identity_round_trips_without_becoming_artifact_identity() {
    let mut request = request();
    request.assignment = Some(ProgramExecutionAssignment {
        scope_id: ExecutionScopeId::derive(&[b"scope"]),
        pool_id: PoolId::derive(&[b"pool"]),
        task_id: TaskId::derive(&[b"task"]),
        attempt_id: AttemptId::derive(&[b"attempt"]),
        worker_id: WorkerId::derive(&[b"worker"]),
        backend: PoolBackend::Remote,
        resources: Default::default(),
    });
    request.job_id = Some(JobId::derive(&[b"job"]));

    let encoded = serde_json::to_vec(&request).unwrap();
    let decoded: ProgramExecutionRequest = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, request);
    decoded.validate().unwrap();

    let mut with_unknown = serde_json::to_value(&request).unwrap();
    with_unknown
        .as_object_mut()
        .unwrap()
        .insert("worker_hint".into(), serde_json::json!("untyped"));
    assert!(serde_json::from_value::<ProgramExecutionRequest>(with_unknown).is_err());
}

#[test]
fn script_form_has_an_explicit_argument_free_entrypoint() {
    let mut script = request();
    script.recipe.entrypoint = "script".into();
    script.callable = ProgramCallable::semantic(ProgramFunctionId(0), None);
    script.artifact = ProgramArtifact::materialize(
        &script.recipe,
        ExecutableForm::InterpreterScriptV1,
        b"script-bytecode".to_vec(),
    )
    .unwrap();
    script.validate().unwrap();
    script
        .arguments
        .push(runmat_execution::value::ValuePayload::Inline(Box::new(
            runmat_execution::value::InlineValue::String("unexpected".into()),
        )));
    assert!(script.validate().is_err());
}

#[test]
fn named_executable_unit_binds_recipe_name_and_typed_function_identity() {
    let request = executable_unit_request();
    assert_eq!(request.recipe.entrypoint, "main");
    request.validate().unwrap();

    let mut wrong_function = request.clone();
    wrong_function.callable = ProgramCallable::semantic(ProgramFunctionId(1), Some("main".into()));
    let error = wrong_function.validate().unwrap_err();
    assert!(error
        .to_string()
        .contains("executable unit request does not match its declared entrypoint"));

    let mut wrong_name = request.recipe.clone();
    wrong_name.entrypoint = "different_entrypoint".into();
    let error = ProgramArtifact::materialize(
        &wrong_name,
        ExecutableForm::ExecutableUnitV4,
        request.artifact.executable_bytes.clone(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("entrypoint"));
}

#[test]
fn compiler_bound_script_regions_accept_arguments_but_top_level_scripts_do_not() {
    let region = runmat_types::ParallelRegionId(runmat_types::RegionId {
        function: runmat_types::ProgramFunctionId(7),
        ordinal: 3,
    });
    let mut parallel = request();
    parallel.callable = ProgramCallable::parallel_region(region);
    parallel.context = ProgramInvocationContext::ParallelTask {
        task: runmat_execution::ParallelTaskContext {
            region,
            chunk: runmat_execution::ParallelChunk {
                ordinal: 0,
                start: 0,
                len: 1,
            },
            randomness: runmat_execution::ParallelRandomnessContext::Inherit,
        },
    };
    parallel.recipe.entrypoint = parallel.callable.recipe_entrypoint();
    parallel.artifact = ProgramArtifact::materialize(
        &parallel.recipe,
        ExecutableForm::InterpreterScriptV2,
        b"compiler-bound-parallel-region".to_vec(),
    )
    .unwrap();
    parallel.arguments = vec![inline_argument("parallel capture")];
    parallel.validate().unwrap();

    let scope_id = ExecutionScopeId::derive(&[b"script-region-scope"]);
    let pool = runmat_execution::PoolHandle {
        id: PoolId::derive(&[b"script-region-pool"]),
        scope_id,
        generation: 1,
    };
    let gang = runmat_execution::GangHandle {
        id: runmat_execution::GangId::derive(&[b"script-region-gang"]),
        scope_id,
        generation: 1,
        pool,
        labs: runmat_types::LabCount(2),
    };
    let mut spmd = request();
    spmd.callable = ProgramCallable::spmd_region(region);
    spmd.context = ProgramInvocationContext::SpmdTask {
        task: runmat_execution::SpmdTaskContext {
            gang,
            region,
            rank: runmat_types::LabRank(1),
        },
    };
    spmd.recipe.entrypoint = spmd.callable.recipe_entrypoint();
    spmd.artifact = ProgramArtifact::materialize(
        &spmd.recipe,
        ExecutableForm::InterpreterScriptV2,
        b"compiler-bound-spmd-region".to_vec(),
    )
    .unwrap();
    spmd.arguments = vec![inline_argument("spmd capture")];
    spmd.validate().unwrap();

    let mut top_level = request();
    top_level.recipe.entrypoint = "script".into();
    top_level.callable = ProgramCallable::semantic(ProgramFunctionId(0), None);
    top_level.artifact = ProgramArtifact::materialize(
        &top_level.recipe,
        ExecutableForm::InterpreterScriptV2,
        b"top-level-script".to_vec(),
    )
    .unwrap();
    top_level.validate().unwrap();
    top_level.arguments = vec![inline_argument("unexpected")];
    assert!(top_level.validate().is_err());
}

#[test]
fn externalized_response_requires_a_complete_unique_result_inventory() {
    let request = request();
    let root = result_reference(b"root");
    let response = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![ValuePayload::Object(Box::new(root.clone()))],
        result_objects: vec![root.clone(), result_reference(b"chunk")],
    };
    response.validate_against(&request).unwrap();

    let missing_root = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![ValuePayload::Object(Box::new(root.clone()))],
        result_objects: vec![result_reference(b"chunk")],
    };
    assert!(missing_root.validate_against(&request).is_err());

    let duplicate = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![ValuePayload::Object(Box::new(root.clone()))],
        result_objects: vec![root.clone(), root],
    };
    assert!(duplicate.validate_against(&request).is_err());
}

#[test]
fn inline_response_uses_a_bounded_output_vector() {
    let request = request();
    let response = ProgramExecutionResponse::Success {
        outputs: vec![ValuePayload::Inline(Box::new(
            runmat_execution::value::InlineValue::F64Bits(3.0_f64.to_bits()),
        ))],
    };
    response.validate_against(&request).unwrap();

    let wrong_count = ProgramExecutionResponse::Success {
        outputs: Vec::new(),
    };
    assert!(wrong_count.validate_against(&request).is_err());

    let frozen_v5 = r#"{"outcome":"success","value":{"form":"inline","value":{"type":"logical","value":true}}}"#;
    assert!(serde_json::from_str::<ProgramExecutionResponse>(frozen_v5).is_err());
}

#[test]
fn spmd_response_preserves_unassigned_outputs_as_protocol_state() {
    let ordinary_request = request();
    let mut spmd_request = ordinary_request.clone();
    let region = runmat_types::ParallelRegionId(runmat_types::RegionId {
        function: runmat_types::ProgramFunctionId(7),
        ordinal: 3,
    });
    let scope_id = ExecutionScopeId::derive(&[b"spmd-response-scope"]);
    let pool = runmat_execution::PoolHandle {
        id: PoolId::derive(&[b"spmd-response-pool"]),
        scope_id,
        generation: 1,
    };
    let gang = runmat_execution::GangHandle {
        id: runmat_execution::GangId::derive(&[b"spmd-response-gang"]),
        scope_id,
        generation: 1,
        pool,
        labs: runmat_types::LabCount(2),
    };
    spmd_request.callable = ProgramCallable::spmd_region(region);
    spmd_request.context = ProgramInvocationContext::SpmdTask {
        task: runmat_execution::SpmdTaskContext {
            gang,
            region,
            rank: runmat_types::LabRank(1),
        },
    };
    spmd_request.recipe.entrypoint = spmd_request.callable.recipe_entrypoint();
    spmd_request.recipe.outputs.requested_outputs = 2;
    spmd_request.requested_outputs = 2;
    spmd_request.artifact = ProgramArtifact::materialize(
        &spmd_request.recipe,
        ExecutableForm::InterpreterBytecodeV2,
        b"spmd-program".to_vec(),
    )
    .unwrap();

    let response = ProgramExecutionResponse::SpmdSuccess {
        outputs: vec![
            Some(runmat_execution::SpmdOutputValue::Value(
                ValuePayload::Inline(Box::new(runmat_execution::value::InlineValue::U64(
                    9_007_199_254_740_993,
                ))),
            )),
            None,
        ],
    };
    response.validate_against(&spmd_request).unwrap();
    let encoded = serde_json::to_vec(&response).unwrap();
    let decoded: ProgramExecutionResponse = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, response);

    let wrong_callable = ProgramExecutionResponse::SpmdSuccess {
        outputs: vec![None],
    };
    assert!(wrong_callable.validate_against(&ordinary_request).is_err());
}

#[test]
fn canonical_request_digest_is_exact_without_a_small_envelope_limit() {
    let mut large = request();
    large.arguments.push(ValuePayload::Inline(Box::new(
        runmat_execution::value::InlineValue::String("x".repeat(300_000).into()),
    )));
    let digest = large.canonical_digest().unwrap();
    let mut changed = large;
    changed.arguments[0] = ValuePayload::Inline(Box::new(
        runmat_execution::value::InlineValue::String(format!("{}y", "x".repeat(299_999)).into()),
    ));
    assert_ne!(digest, changed.canonical_digest().unwrap());
}

#[test]
fn borrowed_request_preserves_original_payload_wire_identity_and_cold_decode() {
    let mut owned = request();
    owned.arguments.push(ValuePayload::Inline(Box::new(
        runmat_execution::value::InlineValue::String("λ\\\"\n".repeat(25_000).into()),
    )));
    let borrowed = owned.as_borrowed();
    assert!(std::ptr::eq(borrowed.recipe, &owned.recipe));
    assert!(std::ptr::eq(borrowed.artifact, &owned.artifact));
    assert_eq!(borrowed.arguments.as_ptr(), owned.arguments.as_ptr());
    borrowed.validate_for_portable_host().unwrap();
    let bytes = serde_json::to_vec(&borrowed).unwrap();
    assert_eq!(bytes, serde_json::to_vec(&owned).unwrap());
    assert_eq!(
        borrowed.canonical_digest().unwrap(),
        owned.canonical_digest().unwrap()
    );
    let cold: ProgramExecutionRequest = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(cold, owned);
    let response = ProgramExecutionResponse::from_success_outputs(
        vec![ValuePayload::Inline(Box::new(
            runmat_execution::value::InlineValue::Logical(true),
        ))],
        vec![],
    );
    response.validate_against(&borrowed).unwrap();
    response.validate_against(&cold).unwrap();
    let argument_owner = owned.arguments.as_ptr();
    let worker = WorkerProgramRequest::direct(owned);
    let WorkerProgramRequest::Direct(direct) = &worker else {
        unreachable!("Direct request constructor");
    };
    assert_eq!(direct.arguments.as_ptr(), argument_owner);
    assert_eq!(serde_json::to_vec(&worker).unwrap(), bytes);
    let cold_worker: WorkerProgramRequest = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(cold_worker, worker);
}

#[test]
fn task_success_projection_preserves_inline_delivery_and_request_arity() {
    let request = request();
    let value = ValuePayload::Inline(Box::new(runmat_execution::value::InlineValue::Logical(
        true,
    )));
    let response = ProgramExecutionResponse::from_success_outputs(vec![value.clone()], vec![]);
    assert_eq!(
        response,
        ProgramExecutionResponse::Success {
            outputs: vec![value.clone()]
        }
    );
    response.validate_against(&request).unwrap();
    for outputs in [vec![], vec![value.clone(), value]] {
        assert!(
            ProgramExecutionResponse::from_success_outputs(outputs, vec![])
                .validate_against(&request)
                .is_err()
        );
    }
}
