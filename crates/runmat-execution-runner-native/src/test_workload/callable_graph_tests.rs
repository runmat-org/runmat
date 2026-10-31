use std::collections::BTreeSet;
use std::rc::Rc;

use runmat_execution::value::{
    GraphNode, GraphProgramUnitId, InlineValue, ValueLimits, ValuePayload,
};
use runmat_execution_artifact::{
    ExecutableForm, GraphExecutionRequest, GraphInvocationRoots, ProgramArtifact,
    ProgramBuildRecipe, ProgramExecutionResponse, ProgramTarget,
    GRAPH_EXECUTION_REQUEST_SCHEMA_V10, GRAPH_EXECUTION_REQUEST_SCHEMA_V11,
    GRAPH_EXECUTION_REQUEST_SCHEMA_V12, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
};
use runmat_runtime::execution::value_codec::GraphCaptureProof;
use runmat_value::{BoundFunctionHandle, Closure, Value};

use super::execute_host_graph_request;

#[tokio::test]
async fn selected_graph_worker_preserves_callable_argument_layouts() {
    let ambient_path_name = std::env::vars_os()
        .find_map(|(name, value)| {
            let name = name.to_str()?;
            (name.eq_ignore_ascii_case("PATH") && !value.is_empty()).then(|| name.to_owned())
        })
        .expect("worker fixture has an ambient executable search path");
    let source = format!(
        "function result = graph_callable_arguments(x)\nc = 11;\nh = @(z) c + z + double(~isempty(getenv('{ambient_path_name}')));\nresult = h(x);\nend\n"
    );
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    session.set_compat_mode(runmat_core::CompatMode::RunMat);
    let unit = session
        .compile_executable_unit(
            runmat_core::ExecutableSource::new(
                "graph-callable-arguments@1",
                "graph_callable_arguments.m",
                &source,
            ),
            None,
        )
        .await
        .unwrap();
    let envelope = unit
        .portable_envelope_for(Some("graph_callable_arguments"))
        .unwrap();
    let bytes = envelope.canonical_bytes().unwrap();
    let admitted = runmat_vm::admit_executable_unit(&bytes).unwrap();
    let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let bound = admitted.bind_runtime_sources(runtime).unwrap();
    let (_, proofs, bindings) =
        runmat_vm::selected_graph_callable_facts(bound.registry(), GraphProgramUnitId(0))
            .unwrap()
            .into_parts();
    let (owner, proof) = proofs
        .iter()
        .find(|(_, proof)| matches!(proof.captures, GraphCaptureProof::Exact(1)))
        .expect("compiled anonymous body has one admitted capture");
    let binding = bindings.get(owner).unwrap();
    let codec = bound.graph_codec_admission().unwrap();
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: unit.revision().program_revision.clone(),
        entrypoint: envelope.manifest.identity.entrypoint.clone(),
        outputs: runmat_execution::OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: "interpreter".into(),
        target: ProgramTarget::portable("graph-callable-arguments"),
        interop: runmat_types::InteropManifest::empty(),
        foreign_policy: None,
        accelerators: Vec::new(),
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let artifact =
        ProgramArtifact::materialize(&recipe, ExecutableForm::ExecutableUnitV4, bytes).unwrap();
    let legacy_values = vec![
        Value::BoundFunctionHandle(BoundFunctionHandle::new(
            proof.qualified_name.clone(),
            binding.function,
            binding.authority.clone(),
            proof.source_role,
            proof.issued_source_file.clone(),
        )),
        Value::Num(11.0),
        Value::Num(17.0),
    ];
    let requests = [
        (
            GRAPH_EXECUTION_REQUEST_SCHEMA_V12,
            vec![
                Value::Closure(Closure::new(
                    proof.qualified_name.clone(),
                    Some(binding.clone()),
                    proof.source_role,
                    proof.issued_source_file.clone(),
                    vec![Value::Num(11.0)],
                    None,
                )),
                Value::Num(17.0),
            ],
        ),
        (GRAPH_EXECUTION_REQUEST_SCHEMA_V11, legacy_values.clone()),
        (GRAPH_EXECUTION_REQUEST_SCHEMA_V10, legacy_values),
    ];
    for (schema_version, values) in requests {
        let graph = codec.snapshot(&values, ValueLimits::default()).unwrap();
        let roots = GraphInvocationRoots::Callable {
            target: graph.roots[0],
            arguments: graph.roots[1..].to_vec(),
        };
        let request = GraphExecutionRequest {
            invocation_controls: None,
            project_runtime_policy: None,
            schema_version,
            recipe: recipe.clone(),
            artifact: artifact.clone(),
            retained_units: Vec::new(),
            class_dependencies: Vec::new(),
            class_method_dependencies: Vec::new(),
            static_dependencies: Vec::new(),
            selected_bundle_archives: vec![None],
            selected_foreign_policies: Vec::new(),
            graph,
            roots,
            context: runmat_execution::ProgramInvocationContext::Direct,
            assignment: None,
            job_id: None,
            requested_outputs: 1,
        };
        request.validate().unwrap();
        if schema_version == GRAPH_EXECUTION_REQUEST_SCHEMA_V12 {
            let policy = runmat_config::runtime::ClusterRuntimePolicy::built_in();
            let cluster =
                runmat_core::cluster_settings::prepare_graph_for_cluster(request.clone(), &policy)
                    .unwrap();
            assert_eq!(
                cluster.schema_version,
                runmat_execution_artifact::GRAPH_EXECUTION_REQUEST_SCHEMA_V13
            );
            assert!(cluster.project_runtime_policy.is_some());
            let encoded = serde_json::to_vec(&cluster).unwrap();
            runmat_execution_artifact::admit_remote_program_object_bytes(&encoded).unwrap();
            // Configured remote Graph work has no invocation environment;
            // the host's PATH must not enter its callable or nested callbacks.
            assert_single_graph_scalar(
                execute_host_graph_request(cluster.clone(), None).await,
                28.0,
            );
            let mut captured = cluster.clone();
            captured.schema_version = runmat_execution_artifact::GRAPH_EXECUTION_REQUEST_SCHEMA_V14;
            captured.invocation_controls =
                Some(runmat_execution_artifact::GraphInvocationControls {
                    environment: runmat_execution::EnvironmentSnapshot::new(vec![(
                        ambient_path_name.clone(),
                        "captured-invocation-only".into(),
                    )])
                    .unwrap(),
                    console: None,
                });
            captured.validate().unwrap();
            assert_single_graph_scalar(execute_host_graph_request(captured, None).await, 29.0);
            let mut wrong_language = policy.clone();
            wrong_language.language.compat = runmat_config::runtime::LanguageCompatMode::Matlab;
            assert!(runmat_core::cluster_settings::prepare_graph_for_cluster(
                request.clone(),
                &wrong_language
            )
            .unwrap_err()
            .contains("recompile"));
            let mut missing = cluster;
            missing.project_runtime_policy = None;
            assert!(missing.validate().is_err());
        }
        let response = execute_host_graph_request(request, None).await;
        if schema_version == GRAPH_EXECUTION_REQUEST_SCHEMA_V10 {
            let ProgramExecutionResponse::Failure { message } = response else {
                panic!("V10 executed selected arithmetic despite its V11 code requirement");
            };
            assert!(
                message.contains("selected graph executable requires a newer schema"),
                "{message}"
            );
            continue;
        }
        // Legacy Graph schemas carry no invocation environment. They cannot
        // recover ambient host authority by taking the standalone VM route.
        assert_single_graph_scalar(response, 28.0);
    }
}

fn assert_single_graph_scalar(response: ProgramExecutionResponse, expected: f64) {
    let ProgramExecutionResponse::GraphSuccess { graph } = response else {
        panic!("selected callable worker failed: {response:?}");
    };
    let [root] = graph.roots.as_slice() else {
        panic!("selected callable returned an invalid output count");
    };
    assert!(
        matches!(
            graph.nodes.get(root.0 as usize),
            Some(GraphNode::Leaf {
                value: ValuePayload::Inline(value)
            }) if **value == InlineValue::F64Bits(expected.to_bits())
        ),
        "expected scalar {expected}, received {:?}",
        graph.nodes.get(root.0 as usize)
    );
}
