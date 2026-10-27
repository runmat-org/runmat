//! One invocation-scoped value/activation codec. The selected executable
//! supplies source and lexical-layout facts; the payload never certifies them.

use std::collections::HashMap;

use runmat_execution::value::GraphInvocationRoots;
use runmat_execution::value::{
    CallableForm, GraphBoundOwner, GraphCallable, GraphCallableOwner, GraphLexicalLayout,
    PayloadGraph, ValueLimits,
};
use runmat_execution::ExecutableSourceMapV2;
use runmat_value::{BoundFunctionBinding, Value};

use super::ValueCodecError;

mod issuer;
mod reconstruct;
mod route;
mod snapshot;
pub use issuer::{GraphSourceIssuer, GraphSourceUnitFacts};
pub use route::deferred_invocation_requires_graph;
pub use snapshot::{prepare_callable_targets, GraphFrameSourceRequirement, PreparedPayloadGraph};
#[cfg(test)]
mod tests;

/// Exact selected-body facts issued by an admitted executable, never accepted
/// from graph bytes. By-value capture arity is distinct from shared cells.
#[derive(Clone, Debug)]
pub struct GraphCallableProof {
    pub qualified_name: String,
    pub source_role: runmat_types::FunctionSourceRole,
    /// Exact issuing-session path for locally constructed Values. This is
    /// checked before a graph is made portable and restored only after the
    /// destination admits the graph against its selected executable.
    pub issued_source_file: Option<String>,
    /// Portable path from the admitted V3 source record. Wire callable
    /// identity never depends on a client's absolute filesystem path.
    pub source_file: Option<String>,
    pub captures: GraphCaptureProof,
}

/// The admitted producer, not payload bytes, chooses a closure's capture ABI.
#[derive(Clone, Copy, Debug)]
pub enum GraphCaptureProof {
    Exact(usize),
    /// VM copies admitted slots; Native may carry the same nested binding via
    /// admitted shared activation or delegated canonical storage without the
    /// redundant by-value vector. The selected body issues the zero-copy basis.
    NestedShared {
        vm_copied: usize,
        zero_copy: GraphNestedZeroCopy,
    },
    /// Runtime's bound instance-method closure owns exactly one receiver.
    InstanceMethodReceiver,
}

/// Why a nested producer may omit VM's copied capture vector. This is issued
/// from admitted lexical/storage metadata, never from a payload assertion.
#[derive(Clone, Copy, Debug)]
pub enum GraphNestedZeroCopy {
    RequiresCopies,
    SharedActivation,
    CanonicalStorage,
}

impl GraphCallableProof {
    fn matches(&self, callable: &GraphCallable) -> bool {
        self.matches_with_source(callable, &self.source_file)
    }

    fn matches_issued(&self, callable: &GraphCallable) -> bool {
        self.matches_with_source(callable, &self.issued_source_file)
    }

    fn matches_with_source(&self, callable: &GraphCallable, source_file: &Option<String>) -> bool {
        self.qualified_name == callable.qualified_name
            && self.source_role == callable.source_role
            && source_file == &callable.source_file
            && match callable.form {
                CallableForm::Closure => match self.captures {
                    GraphCaptureProof::Exact(count) => callable.captures.len() == count,
                    GraphCaptureProof::NestedShared {
                        vm_copied,
                        zero_copy,
                    } => {
                        callable.captures.len() == vm_copied
                            || (callable.captures.is_empty()
                                && match zero_copy {
                                    GraphNestedZeroCopy::RequiresCopies => false,
                                    GraphNestedZeroCopy::SharedActivation => {
                                        callable.activation.is_some()
                                    }
                                    GraphNestedZeroCopy::CanonicalStorage => true,
                                })
                    }
                    GraphCaptureProof::InstanceMethodReceiver => {
                        callable.captures.len() == 1 && callable.activation.is_none()
                    }
                },
                CallableForm::Handle => {
                    !matches!(self.captures, GraphCaptureProof::NestedShared { .. })
                        && callable.captures.is_empty()
                        && callable.activation.is_none()
                }
            }
    }
}

fn validate_bound_callable_proofs(
    graph: &PayloadGraph,
    proofs: &HashMap<GraphBoundOwner, GraphCallableProof>,
) -> Result<(), ValueCodecError> {
    for node in &graph.nodes {
        if let runmat_execution::value::GraphNode::Callable(callable) = node {
            if let GraphCallableOwner::Bound(origin) = callable.owner {
                if !proofs
                    .get(&origin)
                    .is_some_and(|proof| proof.matches(callable))
                {
                    return Err(invalid("bound callable differs from its admitted body"));
                }
            }
        }
    }
    Ok(())
}

/// Snapshot one caller-selected value batch under one GC-root lease and budget.
/// Artifact wrappers assign target/argument roles; graph ordinals do not.
/// Layouts must come from the selected admitted executable, not the wire.
pub fn encode_payload_graph_in_context(
    values: &[Value],
    runtime: &crate::context::RuntimeContext,
    source_map: &ExecutableSourceMapV2,
    layouts: &HashMap<GraphBoundOwner, GraphLexicalLayout>,
    proofs: &HashMap<GraphBoundOwner, GraphCallableProof>,
    limits: ValueLimits,
) -> Result<PayloadGraph, ValueCodecError> {
    snapshot::encode(values, runtime, source_map, layouts, proofs, limits)
}

/// Snapshot with source and session authority captured at Future creation.
/// The issuer retains no execution service and must still be live at spawn.
pub fn encode_payload_graph_with_issuer(
    values: &[Value],
    issuer: &GraphSourceIssuer,
    source_map: &ExecutableSourceMapV2,
    layouts: &HashMap<GraphBoundOwner, GraphLexicalLayout>,
    proofs: &HashMap<GraphBoundOwner, GraphCallableProof>,
    limits: ValueLimits,
    workspace: Option<&super::ValidationWorkspaceOwner>,
) -> Result<PayloadGraph, ValueCodecError> {
    snapshot::encode_with_issuer(
        values, issuer, source_map, layouts, proofs, limits, workspace,
    )
}

/// Traverse and root a graph once, leaving leaves unencoded until the exact
/// admitted V3 unit set and its deterministic source projection are selected.
pub fn prepare_payload_graph<'a>(
    values: &'a [Value],
    issuer: &'a GraphSourceIssuer,
    limits: ValueLimits,
    workspace: Option<&super::ValidationWorkspaceOwner>,
) -> Result<PreparedPayloadGraph<'a>, ValueCodecError> {
    snapshot::prepare(values, issuer, limits, workspace)
}

/// Prepare one typed deferred call, preserving its selected callable target
/// and root roles without reconstructing a name-based Value handle.
pub fn prepare_deferred_invocation_graph<'a>(
    invocation: &'a crate::execution::DeferredInvocation,
    issuer: &'a GraphSourceIssuer,
    limits: ValueLimits,
    workspace: Option<&super::ValidationWorkspaceOwner>,
) -> Result<(PreparedPayloadGraph<'a>, GraphInvocationRoots), ValueCodecError> {
    snapshot::prepare_invocation(invocation, issuer, limits, workspace)
}

/// Validate before allocating, reconstruct fresh local cells, then publish a
/// rooted batch. Exact source resolution is selected from `runtime`, never a
/// later drainer's ambient context.
pub fn decode_payload_graph_in_context(
    graph: &PayloadGraph,
    runtime: &crate::context::RuntimeContext,
    source_map: &ExecutableSourceMapV2,
    layouts: &HashMap<GraphBoundOwner, GraphLexicalLayout>,
    proofs: &HashMap<GraphBoundOwner, GraphCallableProof>,
    bindings: &HashMap<GraphBoundOwner, BoundFunctionBinding>,
    limits: ValueLimits,
) -> Result<crate::execution::RootedValues, ValueCodecError> {
    reconstruct::decode(
        graph,
        runtime,
        source_map,
        None,
        layouts,
        proofs,
        bindings,
        &HashMap::new(),
        limits,
    )
}

/// Decode an admitted worker graph with its exact selected source-unit basis.
/// Portable frame ordinals are localized only after this issuer and the
/// authenticated union source map have both been selected.
pub fn decode_payload_graph_with_source_issuer_in_context(
    graph: &PayloadGraph,
    runtime: &crate::context::RuntimeContext,
    source_map: &ExecutableSourceMapV2,
    issuer: &GraphSourceIssuer,
    layouts: &HashMap<GraphBoundOwner, GraphLexicalLayout>,
    proofs: &HashMap<GraphBoundOwner, GraphCallableProof>,
    bindings: &HashMap<GraphBoundOwner, BoundFunctionBinding>,
    limits: ValueLimits,
) -> Result<crate::execution::RootedValues, ValueCodecError> {
    decode_payload_graph_with_selected_classes_in_context(
        graph,
        runtime,
        source_map,
        issuer,
        layouts,
        proofs,
        bindings,
        &HashMap::new(),
        limits,
    )
}

/// Selected class receivers require a VM/Core-issued constructor product
/// projection in addition to the ordinary callable/source admission. The
/// payload cannot certify its own class ancestry or method bindings.
pub fn decode_payload_graph_with_selected_classes_in_context(
    graph: &PayloadGraph,
    runtime: &crate::context::RuntimeContext,
    source_map: &ExecutableSourceMapV2,
    issuer: &GraphSourceIssuer,
    layouts: &HashMap<GraphBoundOwner, GraphLexicalLayout>,
    proofs: &HashMap<GraphBoundOwner, GraphCallableProof>,
    bindings: &HashMap<GraphBoundOwner, BoundFunctionBinding>,
    selected_classes: &HashMap<
        GraphBoundOwner,
        std::rc::Rc<runmat_value::SelectedObjectClassFacts>,
    >,
    limits: ValueLimits,
) -> Result<crate::execution::RootedValues, ValueCodecError> {
    reconstruct::decode(
        graph,
        runtime,
        source_map,
        Some(issuer),
        layouts,
        proofs,
        bindings,
        selected_classes,
        limits,
    )
}

fn invalid(message: impl Into<String>) -> ValueCodecError {
    ValueCodecError::invalid("$", message)
}
