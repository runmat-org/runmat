//! Shared-activation callables retain the canonical selected graph. Legacy
//! by-value captures are not used to manufacture a lexical activation.
use super::*;
use runmat_execution::value::{GraphCallableOwner, GraphInvocationRoots, GraphNode};
use runmat_execution_artifact::GraphExecutionRequest;

/// The runtime snapshots the actual compiler-constructed closure through its
/// selected graph codec. This boundary retains the resulting closed graph;
/// worker admission still proves every selected body and lexical-layout edge.
pub fn admit_graph_callable_signal_generator(
    signal: &PhysicalSignal,
    program_revision: ProgramRevision,
    callable: BundleCallable,
    compiled_binding: CompiledCallableBinding,
    function: &FunctionAnalysis,
    frozen_graph: GraphExecutionRequest,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> AnalysisBridgeResult<AdmittedSignalGenerator> {
    frozen_graph.validate()?;
    let generator = signal
        .generator()
        .ok_or_else(|| AnalysisBridgeError::Invalid("signal is not callable-backed".into()))?;
    if !matches!(&generator.invocation.callable,
        runmat_analysis_core::GeneratorCallable::BoundFunction{function:bound,..} if *bound==function.function)
    {
        return Err(AnalysisBridgeError::Identity(
            "graph callable differs from authored bound function",
        ));
    }
    captures(&frozen_graph, &program_revision, function)?;
    let executable =
        frozen_graph
            .artifact
            .executable_unit()?
            .ok_or(AnalysisBridgeError::Identity(
                "graph generator is not an executable unit",
            ))?;
    compiled_binding.validate_candidate(
        &executable.manifest.identity,
        &program_revision,
        &callable,
    )?;
    let graph_digest = contract_digest("frozen-callable-graph", &frozen_graph)?;
    admit_signal_generator(
        signal,
        CallableAdmission {
            program_revision,
            executable_source_digest: compiled_binding.source_digest(),
            callable,
            compiled_binding,
            function,
            available_capabilities,
        },
        FrozenCallableBinding {
            inputs: FrozenGeneratorInputs::Graph(Box::new(frozen_graph)),
            callable_value_digest: Some(graph_digest),
        },
    )
}

pub(crate) fn captures<'a>(
    request: &'a GraphExecutionRequest,
    revision: &ProgramRevision,
    function: &FunctionAnalysis,
) -> AnalysisBridgeResult<Vec<&'a ValuePayload>> {
    request.validate()?;
    let GraphInvocationRoots::Callable { target, arguments } = &request.roots else {
        return Err(AnalysisBridgeError::Identity(
            "generator graph root is not a callable",
        ));
    };
    if !arguments.is_empty()
        || request.graph.roots != [*target]
        || request.assignment.is_some()
        || request.job_id.is_some()
        || request.requested_outputs != 1
        || request.recipe.program_revision != *revision
    {
        return Err(AnalysisBridgeError::Identity(
            "frozen generator graph has mutable invocation state or a foreign revision",
        ));
    }
    let Some(GraphNode::Callable(callable)) = request.graph.nodes.get(target.0 as usize) else {
        return Err(AnalysisBridgeError::Identity(
            "generator target is not a canonical callable node",
        ));
    };
    let GraphCallableOwner::Bound(owner) = callable.owner else {
        return Err(AnalysisBridgeError::Identity(
            "generator graph target has no original bound owner",
        ));
    };
    let expected =
        runmat_execution::FunctionBindingOrigin::from_revision(revision, function.function)?;
    if owner.unit.0 != 0 || owner.origin != expected {
        return Err(AnalysisBridgeError::Identity(
            "generator graph target belongs to a different selected executable function",
        ));
    }
    if function.captures.len() != function.callable.captures.len()
        || function
            .captures
            .iter()
            .zip(&function.callable.captures)
            .any(|(capture, value)| capture.value != *value)
    {
        return Err(AnalysisBridgeError::Identity(
            "graph capture source evidence is incomplete or stale",
        ));
    }
    if !callable.captures.is_empty() && callable.captures.len() != function.captures.len() {
        return Err(AnalysisBridgeError::Identity(
            "graph copied capture ABI is incomplete",
        ));
    }
    function.captures.iter().enumerate().map(|(index,capture)| {
        let reference=match capture.source {
            runmat_mir::analysis::FunctionCaptureSource::ByValue=>*callable.captures.get(index)
                .ok_or(AnalysisBridgeError::Identity("by-value graph capture is absent"))?,
            runmat_mir::analysis::FunctionCaptureSource::Lexical(binding)=>lexical_value(request,callable.activation,binding)?,
            runmat_mir::analysis::FunctionCaptureSource::CanonicalStorage(_)=>return Err(AnalysisBridgeError::Invalid("analysis generators cannot freeze ambient canonical storage as a lexical capture".into())),
            runmat_mir::analysis::FunctionCaptureSource::Unknown=>return Err(AnalysisBridgeError::Identity("graph capture has no exact compiler source evidence")),
        };
        match request.graph.nodes.get(reference.0 as usize) {
                Some(GraphNode::Leaf { value }) => {
                    value.validate(ValueLimits::default())?;
                    Ok(value)
                }
                _ => Err(AnalysisBridgeError::Invalid(
                    "generator capture contract requires closed portable leaf captures".into(),
                )),
            }
    })
        .collect()
}

fn lexical_value(
    request: &GraphExecutionRequest,
    mut activation: Option<runmat_execution::value::GraphNodeRef>,
    binding: runmat_types::BindingId,
) -> AnalysisBridgeResult<runmat_execution::value::GraphNodeRef> {
    let mut visited = std::collections::HashSet::new();
    let mut found = None;
    while let Some(reference) = activation {
        if !visited.insert(reference) {
            return Err(AnalysisBridgeError::Identity(
                "graph activation parent chain is cyclic",
            ));
        }
        let Some(GraphNode::Activation(value)) = request.graph.nodes.get(reference.0 as usize)
        else {
            return Err(AnalysisBridgeError::Identity(
                "shared capture has no canonical activation",
            ));
        };
        if let Some(value) = value.bindings.iter().find(|value| value.binding == binding) {
            if found.is_some() {
                return Err(AnalysisBridgeError::Identity(
                    "shared capture has multiple owned cells",
                ));
            }
            found = Some(value.value.ok_or(AnalysisBridgeError::Identity(
                "shared capture is unassigned",
            ))?);
        }
        activation = value.parent;
    }
    found.ok_or(AnalysisBridgeError::Identity(
        "compiler shared binding is absent from graph activation ancestry",
    ))
}

pub(crate) fn validate_identity(
    admission: crate::signal_program::GeneratorProgramAdmissionRef<'_>,
    request: &GraphExecutionRequest,
) -> AnalysisBridgeResult<()> {
    if admission.frozen_inputs.graph().is_none() {
        return Err(AnalysisBridgeError::Identity(
            "graph invocation has no matching frozen input mode",
        ));
    }
    let graph_digest = contract_digest("frozen-callable-graph", request)?;
    let callable_digest = contract_digest(
        "callable",
        &(
            admission.program_revision,
            &admission.callable.owner_identity,
            &admission.callable.qualified_name,
            admission.callable.source_digest,
            admission.executable_source_digest,
            admission.compiled_binding,
            admission.function,
            Some(graph_digest),
        ),
    )?;
    if callable_digest.to_string() != admission.provenance.callable_id {
        return Err(AnalysisBridgeError::Identity(
            "frozen graph differs from original generator admission identity",
        ));
    }
    Ok(())
}
