//! Actual shared-closure capture and the existing selected-value wire route.
use super::*;
use runmat_execution::value::{GraphInvocationRoots, GraphNode, PayloadGraph};
use runmat_execution_artifact::{GraphExecutionRequest, WorkerProgramRequest};

pub(super) fn graph_descriptor(
    revision: &ProgramRevision,
    owner: &str,
    source: &str,
    function: &runmat_mir::analysis::FunctionAnalysis,
    executable: &runmat_execution::ExecutableUnitEnvelope,
    graph: PayloadGraph,
) -> GraphExecutionRequest {
    let callable = BundleCallable {
        owner_identity: owner.into(),
        qualified_name: "loads.pulse".into(),
        source_digest: Digest::sha256(source),
    };
    let binding =
        runmat_analysis_execution::named_callable::CompiledCallableBinding::from_selected_identity(
            &executable.manifest.identity,
            &callable.qualified_name,
        )
        .unwrap();
    let descriptor = runmat_analysis_execution::named_callable::prepare_named_callable_descriptor(
        runmat_analysis_execution::named_callable::NamedCallableAdmissionRef {
            program_revision: revision,
            callable: &callable,
            compiled_binding: &binding,
            executable_source_digest: executable.manifest.identity.source_digest,
            function: function.function,
            required_capabilities: &function.capabilities.0,
        },
        executable.clone(),
        runmat_analysis_execution::named_callable::AnalysisCallablePurpose::Generator,
    )
    .unwrap();
    assert_eq!(
        descriptor.recipe.interop,
        runmat_types::InteropManifest::empty(),
        "this real program has no foreign policy/archive"
    );
    let [target] = graph.roots.as_slice() else {
        panic!("one actual closure root");
    };
    let roots = GraphInvocationRoots::Callable {
        target: *target,
        arguments: Vec::new(),
    };
    let request = GraphExecutionRequest {
        invocation_controls: None,
        project_runtime_policy: None,
        schema_version: runmat_execution_artifact::GRAPH_EXECUTION_REQUEST_SCHEMA_V12,
        recipe: descriptor.recipe,
        artifact: descriptor.artifact,
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
    // Bind the exact authored project policy before generator admission and
    // chunk identity construction, not after the original task is granted.
    let request = runmat_core::cluster_settings::prepare_graph_for_cluster(
        request,
        &compilation::project_policy(),
    )
    .unwrap();
    let node_owner = request.graph.nodes.as_ptr();
    let bytes = serde_json::to_vec(&request).unwrap();
    let worker = WorkerProgramRequest::graph(request);
    let WorkerProgramRequest::Graph(request) = &worker else {
        unreachable!("Graph request constructor");
    };
    assert_eq!(request.graph.nodes.as_ptr(), node_owner);
    assert_eq!(serde_json::to_vec(&worker).unwrap(), bytes);
    let cold: WorkerProgramRequest = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(cold, worker);
    let WorkerProgramRequest::Graph(request) = worker else {
        unreachable!("Graph request constructor");
    };
    *request
}

pub(super) fn capabilities(request: &WorkerProgramRequest) -> BTreeSet<Capability> {
    let mut capabilities = BTreeSet::from([Capability::ProcessIsolation]);
    if let WorkerProgramRequest::Graph(request) = request {
        let schema =
            runmat_execution::value::GraphRequiredSchema::from_version(request.schema_version)
                .unwrap();
        capabilities.extend(schema.required_capability());
    }
    capabilities
}

pub(super) fn assert_capture_source_controls(
    signal: &PhysicalSignal,
    revision: &ProgramRevision,
    owner: &str,
    source: &str,
    function: &runmat_mir::analysis::FunctionAnalysis,
    executable: &runmat_execution::ExecutableUnitEnvelope,
    request: &GraphExecutionRequest,
) {
    let admit = |request| {
        admit_graph_callable_signal_generator(
            signal,
            revision.clone(),
            BundleCallable {
                owner_identity: owner.into(),
                qualified_name: "loads.pulse".into(),
                source_digest: Digest::sha256(source),
            },
            runmat_analysis_execution::named_callable::CompiledCallableBinding::from_selected_identity(
                &executable.manifest.identity,
                "loads.pulse",
            ).unwrap(),
            function,
            request,
            &BTreeSet::new(),
        )
    };
    let mut zero_copy = request.clone();
    let GraphInvocationRoots::Callable { target, .. } = zero_copy.roots else {
        unreachable!()
    };
    let GraphNode::Callable(callable) = &mut zero_copy.graph.nodes[target.0 as usize] else {
        unreachable!()
    };
    assert!(callable.activation.is_some());
    callable.captures.clear();
    assert!(
        admit(zero_copy).is_ok(),
        "actual shared cells need no copied ABI captures"
    );

    let mut changed = request.clone();
    let binding = changed
        .graph
        .nodes
        .iter()
        .find_map(|node| match node {
            GraphNode::Activation(value) => value.bindings.iter().find_map(|binding| binding.value),
            _ => None,
        })
        .unwrap();
    let GraphNode::Leaf { value } = &mut changed.graph.nodes[binding.0 as usize] else {
        unreachable!()
    };
    *value = ValuePayload::Inline(Box::new(runmat_execution::value::InlineValue::String(
        "not the numeric scale cell".into(),
    )));
    assert!(
        admit(changed).is_err(),
        "shared BindingId value, not a stale copied numeric capture, governs fresh admission"
    );
}

pub(super) fn assert_frozen_capture_controls(prepared: &PreparedSignalMaterialization) {
    prepared.validate().unwrap();
    let reopened: runmat_analysis_execution::AdmittedSignalGenerator =
        serde_json::from_slice(&serde_json::to_vec(&prepared.admitted).unwrap()).unwrap();
    assert_eq!(
        reopened, prepared.admitted,
        "canonical graph state survives a cold admission round trip"
    );
    let mut changed = prepared.clone();
    let runmat_analysis_execution::FrozenGeneratorInputs::Graph(request) =
        &mut changed.admitted.frozen_inputs
    else {
        panic!("actual shared closure mode");
    };
    let graph = &mut request.graph;
    let binding = graph
        .nodes
        .iter()
        .find_map(|node| match node {
            GraphNode::Activation(value) => value.bindings.iter().find_map(|binding| binding.value),
            _ => None,
        })
        .expect("actual factory-owned scale cell");
    let GraphNode::Leaf { value } = &mut graph.nodes[binding.0 as usize] else {
        panic!("numeric scale cell");
    };
    let ValuePayload::Inline(value) = value else {
        panic!("closed scale value");
    };
    let runmat_execution::value::InlineValue::Dense(value) = value.as_mut() else {
        panic!("dense scale value");
    };
    value.little_endian_data[0] ^= 1;
    assert!(changed.admitted.frozen_inputs.direct_captures().is_empty());
    assert!(
        changed.validate().is_err(),
        "activation values are bound by the full frozen graph identity"
    );
    let mut demoted = prepared.clone();
    let plan = &demoted.program.chunks[0].plan;
    let direct = runmat_execution_artifact::ProgramExecutionRequest::from_parts(
        demoted.program.descriptor.clone(),
        runmat_execution_artifact::ProgramExecutionInputs {
            schema_version: runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            context: runmat_execution::ProgramInvocationContext::Direct,
            arguments: plan.arguments.clone(),
            objects: Vec::new(),
        },
    )
    .unwrap();
    demoted.program.chunks[0].request = WorkerProgramRequest::direct(direct);
    assert!(
        demoted.validate().is_err(),
        "shared activation cannot be demoted to flat capture ABI"
    );
}

pub(super) fn host(
    request: &WorkerProgramRequest,
) -> runmat_execution::host::ExecutionHostRequirement {
    match request {
        WorkerProgramRequest::Direct(request) => {
            super::super::phase_execution_support::host_requirement(request)
        }
        WorkerProgramRequest::Graph(request) => request
            .execution_host_requirement(BTreeSet::from([
                super::super::phase_execution_support::worker_host_inventory().trust_tier,
            ]))
            .unwrap(),
    }
}

pub(super) fn value(graph: PayloadGraph) -> ValuePayload {
    let [root] = graph.roots.as_slice() else {
        panic!("one numeric generator graph result");
    };
    let root = root.0 as usize;
    let GraphNode::Leaf { value } = graph
        .nodes
        .into_iter()
        .nth(root)
        .expect("admitted output root")
    else {
        panic!("analysis generator must return one portable numeric leaf");
    };
    value
        .validate(runmat_execution::value::ValueLimits::default())
        .unwrap();
    value
}

pub(super) fn success_output(
    identity: Digest,
    success: runmat_execution_runner::AttemptSuccess,
) -> GeneratorChunkOutput {
    let value = match success {
        runmat_execution_runner::AttemptSuccess::Graph { graph, .. } => value(graph),
        success => {
            let (mut values, objects) = success
                .into_values()
                .expect("numeric generator value result");
            assert!(objects.is_empty());
            assert_eq!(values.len(), 1);
            values.remove(0)
        }
    };
    GeneratorChunkOutput {
        chunk_identity: identity,
        value,
    }
}
