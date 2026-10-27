use std::collections::{HashMap, HashSet};

use runmat_execution::value::GraphInvocationRoots;
use runmat_execution::value::{
    GraphActivation, GraphBinding, GraphBindingKind, GraphBoundOwner, GraphCallableOwner,
    GraphLexicalLayout, GraphNode, GraphNodeRef, GraphProgramUnitId, PayloadGraph, ValueLimits,
};
use runmat_execution::{ExecutableSourceMapV2, FunctionBindingOrigin};
use runmat_gc::GcHandle;
use runmat_types::SourceId;
use runmat_value::{BoundFunctionAuthority, FunctionDefinitionLease, LexicalEnvironment, Value};

use super::{invalid, validate_bound_callable_proofs, GraphCallableProof, ValueCodecError};
use crate::call::descriptor::CallableTarget;
use crate::execution::value_codec::encode::InlineSourceAuthority;
use crate::execution::DeferredInvocation;

mod activation;
mod targets;
pub use targets::prepare_callable_targets;
mod value;

pub(super) fn encode(
    values: &[Value],
    runtime: &crate::context::RuntimeContext,
    source_map: &ExecutableSourceMapV2,
    layouts: &HashMap<GraphBoundOwner, GraphLexicalLayout>,
    proofs: &HashMap<GraphBoundOwner, GraphCallableProof>,
    limits: ValueLimits,
) -> Result<PayloadGraph, ValueCodecError> {
    let workspace = crate::execution::value_codec::ValidationWorkspaceOwner::capture(runtime);
    let prepared = prepare_with_authority(values, runtime, limits, Some(&workspace))?;
    let units = prepared.single_unit_nodes();
    prepared.finish_with_authority(runtime, source_map, layouts, proofs, &units, 1)
}

pub(super) fn encode_with_issuer(
    values: &[Value],
    issuer: &super::GraphSourceIssuer,
    source_map: &ExecutableSourceMapV2,
    layouts: &HashMap<GraphBoundOwner, GraphLexicalLayout>,
    proofs: &HashMap<GraphBoundOwner, GraphCallableProof>,
    limits: ValueLimits,
    workspace: Option<&crate::execution::value_codec::ValidationWorkspaceOwner>,
) -> Result<PayloadGraph, ValueCodecError> {
    issuer.ensure_live()?;
    if issuer.selected_unit_count() != 1 {
        return Err(invalid(
            "multi-unit graph snapshot requires exact per-node bound authority",
        ));
    }
    let prepared = prepare_with_authority(values, issuer, limits, workspace)?;
    let units = prepared.single_unit_nodes();
    prepared.finish(issuer, source_map, layouts, proofs, &units)
}

pub(super) fn prepare<'a>(
    values: &'a [Value],
    issuer: &'a super::GraphSourceIssuer,
    limits: ValueLimits,
    workspace: Option<&crate::execution::value_codec::ValidationWorkspaceOwner>,
) -> Result<PreparedPayloadGraph<'a>, ValueCodecError> {
    issuer.ensure_live()?;
    prepare_with_authority(values, issuer, limits, workspace)
}

/// Prepare the actual deferred invocation as one rooted graph. A typed bound
/// target does not become a textual FunctionHandle or acquire execution
/// authority from its display name; its selected proof fills wire metadata
/// only after the exact executable products have been chosen.
pub fn prepare_invocation<'a>(
    invocation: &'a DeferredInvocation,
    issuer: &'a super::GraphSourceIssuer,
    limits: ValueLimits,
    workspace: Option<&crate::execution::value_codec::ValidationWorkspaceOwner>,
) -> Result<(PreparedPayloadGraph<'a>, GraphInvocationRoots), ValueCodecError> {
    enum Role {
        Callable,
        Builtin(runmat_types::BuiltinId),
        ParallelRegion(runmat_types::ParallelRegionId),
    }
    issuer.ensure_live()?;
    let (inputs, role) = match invocation {
        DeferredInvocation::Callable(descriptor) => {
            let target = targets::selected_input(&descriptor.target)?;
            let mut inputs = Vec::with_capacity(descriptor.args.len() + 1);
            inputs.push(target);
            inputs.extend(descriptor.args.iter().map(Input::Borrowed));
            (inputs, Role::Callable)
        }
        DeferredInvocation::Program {
            callable: runmat_execution::ProgramCallable::Builtin { name },
            context: runmat_execution::ProgramInvocationContext::Direct,
            arguments,
            ..
        } => (
            arguments.iter().map(Input::Borrowed).collect(),
            Role::Builtin(name.clone()),
        ),
        DeferredInvocation::Program {
            callable: runmat_execution::ProgramCallable::ParallelRegion { region },
            context: runmat_execution::ProgramInvocationContext::ParallelTask { task },
            arguments,
            ..
        } if task.region == *region => (
            arguments.iter().map(Input::Borrowed).collect(),
            Role::ParallelRegion(*region),
        ),
        DeferredInvocation::Program { .. } => {
            return Err(ValueCodecError::unsupported(
                "$",
                "deferred program does not have a V8 graph invocation role",
            ));
        }
    };
    let prepared = prepare_inputs(inputs, issuer, limits, workspace)?;
    let roots = match role {
        Role::ParallelRegion(region) => GraphInvocationRoots::ParallelRegion {
            region,
            captures: prepared.roots.clone(),
        },
        Role::Builtin(name) => GraphInvocationRoots::Builtin {
            name,
            arguments: prepared.roots.clone(),
        },
        Role::Callable => {
            let (target, arguments) = prepared
                .roots
                .split_first()
                .ok_or_else(|| invalid("callable graph has no target root"))?;
            GraphInvocationRoots::Callable {
                target: *target,
                arguments: arguments.to_vec(),
            }
        }
    };
    Ok((prepared, roots))
}

fn prepare_with_authority<'a>(
    values: &'a [Value],
    issuer: &'a dyn InlineSourceAuthority,
    limits: ValueLimits,
    workspace: Option<&crate::execution::value_codec::ValidationWorkspaceOwner>,
) -> Result<PreparedPayloadGraph<'a>, ValueCodecError> {
    prepare_inputs(
        values.iter().map(Input::Borrowed).collect(),
        issuer,
        limits,
        workspace,
    )
}

fn prepare_inputs<'a>(
    inputs: Vec<Input<'a>>,
    issuer: &'a dyn InlineSourceAuthority,
    limits: ValueLimits,
    workspace: Option<&crate::execution::value_codec::ValidationWorkspaceOwner>,
) -> Result<PreparedPayloadGraph<'a>, ValueCodecError> {
    let inputs_root = crate::execution::RootedValueLease::for_values(
        inputs.iter().filter_map(Input::rooted_value),
    )
    .map_err(|error| invalid(error.to_string()))?;
    let mut encoder = Encoder {
        issuer,
        workspace: workspace.cloned(),
        limits,
        nodes: Vec::new(),
        activations: HashMap::new(),
        provisional_activations: HashMap::new(),
        pending_leaves: Vec::new(),
        pending_bound_targets: Vec::new(),
        pending_bound_callables: Vec::new(),
        pending_selected_objects: Vec::new(),
        selected_handle_targets: HashMap::new(),
        selected_handle_nodes: HashSet::new(),
        callable_instances: HashMap::new(),
        bound_owners: Vec::new(),
        bound_authorities: Vec::new(),
    };
    let mut work = Vec::new();
    let roots = encoder.push_values(inputs, &mut work)?;
    while let Some(task) = work.pop() {
        encoder.process(task, &mut work)?;
    }
    Ok(PreparedPayloadGraph {
        encoder,
        roots,
        _inputs: inputs_root,
    })
}

/// The one rooted Value/activation traversal. Leaves remain borrowed or owned
/// here until the selected unit set determines their portable source ordinals.
pub struct PreparedPayloadGraph<'a> {
    encoder: Encoder<'a>,
    roots: Vec<GraphNodeRef>,
    _inputs: crate::execution::RootedValueLease,
}

/// Whether an exception frame must name admitted code, or may be carried as
/// presentation-only metadata when its original definition has retired.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphFrameSourceRequirement {
    RequiredAdmitted,
    MayDetach,
}

impl PreparedPayloadGraph<'_> {
    fn single_unit_nodes(&self) -> HashMap<GraphNodeRef, GraphProgramUnitId> {
        self.encoder
            .bound_authorities
            .iter()
            .map(|(reference, _, _)| (*reference, GraphProgramUnitId(0)))
            .collect()
    }

    /// Exact in-session definition owners observed during the one rooted
    /// traversal. Origin and equal V3 bytes alone cannot distinguish separate
    /// imports that issued different live callable handles.
    pub fn bound_owner_leases(&self) -> &[(FunctionBindingOrigin, FunctionDefinitionLease)] {
        &self.encoder.bound_owners
    }

    /// Exact issued dependency graphs observed in the rooted traversal.
    /// Selecting an inherited lexical parent by numeric ID in a later
    /// published registry would silently substitute a different definition.
    pub fn bound_authorities(
        &self,
    ) -> &[(GraphNodeRef, FunctionBindingOrigin, BoundFunctionAuthority)] {
        &self.encoder.bound_authorities
    }

    /// Direct bound callable origins discovered without trusting an activation
    /// owner claim. Admitted layouts may add ancestor origins after selection.
    pub fn direct_bound_origins(&self) -> HashSet<FunctionBindingOrigin> {
        let mut origins = HashSet::new();
        origins.extend(
            self.encoder
                .pending_bound_targets
                .iter()
                .map(|(_, origin)| *origin),
        );
        origins.extend(
            self.encoder
                .pending_selected_objects
                .iter()
                .map(|object| object.origin),
        );
        origins.extend(
            self.encoder
                .bound_authorities
                .iter()
                .map(|(_, origin, _)| *origin),
        );
        origins
    }

    pub fn exception_sources(&self) -> HashSet<SourceId> {
        let mut sources = HashSet::new();
        for (_, input) in &self.encoder.pending_leaves {
            if let Value::MException(exception) = input.value() {
                if let Some(error) = exception.original() {
                    sources.extend(
                        error
                            .context
                            .call_frames
                            .iter()
                            .filter_map(|frame| frame.source_id.map(SourceId)),
                    );
                }
            }
        }
        sources
    }

    /// Exact local frame-source requirements discovered in the same rooted
    /// traversal. A source ID alone is insufficient when multiple selected
    /// admitted units share a durable SourcePool alias.
    pub fn exception_frame_sources(
        &self,
    ) -> Vec<(
        SourceId,
        Option<crate::CallFrameSourceOwner>,
        GraphFrameSourceRequirement,
    )> {
        let mut frames = Vec::new();
        for (_, input) in &self.encoder.pending_leaves {
            let Value::MException(exception) = input.value() else {
                continue;
            };
            let Some(error) = exception.original() else {
                continue;
            };
            for frame in &error.context.call_frames {
                if frame
                    .diagnostic_source
                    .as_ref()
                    .is_some_and(crate::CallFrameDiagnosticSource::is_detached)
                {
                    // Presentation-only retained diagnostics do not select
                    // executable units or grant callable/source authority.
                    continue;
                }
                let Some(source) = frame.source_id.map(SourceId) else {
                    continue;
                };
                let requirement = if frame.diagnostic_source.is_some() {
                    GraphFrameSourceRequirement::MayDetach
                } else {
                    GraphFrameSourceRequirement::RequiredAdmitted
                };
                let selected = (source, frame.source_owner.clone(), requirement);
                if !frames.contains(&selected) {
                    frames.push(selected);
                }
            }
        }
        frames
    }

    pub fn finish(
        self,
        issuer: &super::GraphSourceIssuer,
        source_map: &ExecutableSourceMapV2,
        layouts: &HashMap<GraphBoundOwner, GraphLexicalLayout>,
        proofs: &HashMap<GraphBoundOwner, GraphCallableProof>,
        node_units: &HashMap<GraphNodeRef, GraphProgramUnitId>,
    ) -> Result<PayloadGraph, ValueCodecError> {
        issuer.ensure_live()?;
        self.finish_with_authority(
            issuer,
            source_map,
            layouts,
            proofs,
            node_units,
            issuer.selected_unit_count(),
        )
    }

    fn finish_with_authority(
        mut self,
        issuer: &dyn InlineSourceAuthority,
        source_map: &ExecutableSourceMapV2,
        layouts: &HashMap<GraphBoundOwner, GraphLexicalLayout>,
        proofs: &HashMap<GraphBoundOwner, GraphCallableProof>,
        node_units: &HashMap<GraphNodeRef, GraphProgramUnitId>,
        selected_unit_count: usize,
    ) -> Result<PayloadGraph, ValueCodecError> {
        if node_units.len() != self.encoder.bound_authorities.len() {
            return Err(invalid("selected graph unit mapping is incomplete"));
        }
        for (reference, _, _) in &self.encoder.bound_authorities {
            let unit = node_units
                .get(reference)
                .copied()
                .ok_or_else(|| invalid("bound callable has no selected graph unit"))?;
            if usize::from(unit.0) >= selected_unit_count {
                return Err(invalid("bound callable selects an unadmitted graph unit"));
            }
        }
        for pending in self.encoder.pending_bound_callables.drain(..) {
            let unit = node_units[&pending.index];
            self.encoder.nodes[pending.index.0 as usize] = Some(GraphNode::Callable(
                runmat_execution::value::GraphCallable {
                    owner: GraphCallableOwner::Bound(GraphBoundOwner {
                        unit,
                        origin: pending.origin,
                    }),
                    qualified_name: pending.qualified_name,
                    callable_digest: pending.callable_digest,
                    form: pending.form,
                    source_role: pending.source_role,
                    source_file: pending.source_file,
                    captures: pending.captures,
                    activation: pending.activation,
                },
            ));
        }
        for pending in self.encoder.pending_selected_objects.drain(..) {
            let unit = node_units[&pending.index];
            self.encoder.nodes[pending.index.0 as usize] = Some(GraphNode::SelectedObject(
                runmat_execution::value::GraphSelectedObject {
                    constructor: GraphBoundOwner {
                        unit,
                        origin: pending.origin,
                    },
                    class: pending.class,
                    kind: pending.kind,
                    fields: pending.fields,
                    dynamic_properties: pending.dynamic_properties,
                    dynamic_metadata: pending.dynamic_metadata,
                },
            ));
        }
        self.encoder.resolve_activation_owners(layouts)?;
        for (index, origin) in self.encoder.pending_bound_targets.drain(..) {
            let unit = node_units
                .get(&index)
                .copied()
                .ok_or_else(|| invalid("bound target has no selected graph unit"))?;
            let selected = GraphBoundOwner { unit, origin };
            let proof = proofs
                .get(&selected)
                .ok_or_else(|| invalid("bound target has no selected source proof"))?;
            let owner = GraphCallableOwner::Bound(selected);
            let form = runmat_execution::value::CallableForm::Handle;
            self.encoder.nodes[index.0 as usize] = Some(GraphNode::Callable(
                runmat_execution::value::GraphCallable {
                    owner,
                    qualified_name: proof.qualified_name.clone(),
                    callable_digest: runmat_execution::value::CallableValue::identity_digest(
                        owner.portable_owner(),
                        &proof.qualified_name,
                        form,
                        proof.source_role,
                        proof.issued_source_file.as_deref(),
                    ),
                    form,
                    source_role: proof.source_role,
                    source_file: proof.issued_source_file.clone(),
                    captures: Vec::new(),
                    activation: None,
                },
            ));
        }
        for (index, input) in self.encoder.pending_leaves.drain(..) {
            self.encoder.nodes[index.0 as usize] = Some(GraphNode::Leaf {
                value: crate::execution::value_codec::encode::encode_inline_value_with_authority(
                    input.value(),
                    issuer,
                    self.encoder.workspace.as_ref(),
                )?,
            });
        }
        let mut graph = PayloadGraph {
            roots: self.roots,
            nodes: self
                .encoder
                .nodes
                .into_iter()
                .map(|node| node.ok_or_else(|| invalid("prepared graph node was not finalized")))
                .collect::<Result<_, _>>()?,
        };
        for node in &mut graph.nodes {
            let GraphNode::Callable(callable) = node else {
                continue;
            };
            let GraphCallableOwner::Bound(origin) = callable.owner else {
                continue;
            };
            let proof = proofs
                .get(&origin)
                .ok_or_else(|| invalid("bound callable has no selected source proof"))?;
            if !proof.matches_issued(callable) {
                return Err(invalid("bound callable differs from its issuing source"));
            }
            callable.source_file.clone_from(&proof.source_file);
            callable.callable_digest = runmat_execution::value::CallableValue::identity_digest(
                callable.owner.portable_owner(),
                &callable.qualified_name,
                callable.form,
                callable.source_role,
                callable.source_file.as_deref(),
            );
        }
        graph
            .validate_for_program(
                self.encoder.limits,
                Some(source_map),
                layouts,
                selected_unit_count,
            )
            .map_err(|error| invalid(error.to_string()))?;
        validate_bound_callable_proofs(&graph, proofs)?;
        Ok(graph)
    }
}

enum Input<'a> {
    Borrowed(&'a Value),
    Owned(Value),
    /// The original shared Cell backing retains this child; `prepare_inputs`
    /// separately leases all managed handles reached by the rooted traversal.
    SharedCellChild(runmat_value::RetainedArrayValue),
    BoundTarget(&'a runmat_value::BoundFunctionBinding),
    SelectedAnchor(runmat_value::BoundFunctionBinding),
}

impl<'a> Input<'a> {
    fn rooted_value(&self) -> Option<&Value> {
        match self {
            Self::Borrowed(value) => Some(value),
            Self::Owned(value) => Some(value),
            Self::SharedCellChild(child) => Some(child.value()),
            Self::BoundTarget(_) | Self::SelectedAnchor(_) => None,
        }
    }

    fn value(&self) -> &Value {
        match self {
            Self::Borrowed(value) => value,
            Self::Owned(value) => value,
            Self::SharedCellChild(child) => child.value(),
            Self::BoundTarget(_) | Self::SelectedAnchor(_) => {
                unreachable!("typed bound target is not a Value")
            }
        }
    }

    fn cell_children(self) -> Vec<Self> {
        match self {
            Self::Borrowed(Value::Cell(cell)) => cell.data().iter().map(Self::Borrowed).collect(),
            Self::Owned(Value::Cell(cell)) => match cell.into_owned_data() {
                Ok(values) => values.into_iter().map(Self::Owned).collect(),
                Err(cell) => {
                    let owner = cell.data_owner();
                    (0..owner.len())
                        .map(|index| {
                            Self::SharedCellChild(
                                owner.retain_value_at(index).expect("shared cell child"),
                            )
                        })
                        .collect()
                }
            },
            Self::SharedCellChild(parent) => {
                let Value::Cell(cell) = parent.value() else {
                    unreachable!("selected cell input")
                };
                (0..cell.data().len())
                    .map(|index| {
                        Self::SharedCellChild(
                            parent
                                .project(|value| match value {
                                    Value::Cell(cell) => cell.data().get(index),
                                    _ => None,
                                })
                                .expect("retained cell child"),
                        )
                    })
                    .collect()
            }
            _ => unreachable!("selected cell input"),
        }
    }

    fn object_array_children(self) -> Vec<Self> {
        match self {
            Self::Borrowed(Value::ObjectArray(array)) => {
                array.data().iter().map(Self::Borrowed).collect()
            }
            Self::Owned(Value::ObjectArray(array)) => {
                array.data().iter().cloned().map(Self::Owned).collect()
            }
            Self::SharedCellChild(parent) => {
                let Value::ObjectArray(array) = parent.value() else {
                    unreachable!("selected object array input")
                };
                (0..array.data().len())
                    .map(|index| {
                        Self::SharedCellChild(
                            parent
                                .project(|value| match value {
                                    Value::ObjectArray(array) => array.data().get(index),
                                    _ => None,
                                })
                                .expect("retained object-array child"),
                        )
                    })
                    .collect()
            }
            _ => unreachable!("selected object array input"),
        }
    }

    fn named_children(self) -> Vec<(String, Self)> {
        let mut fields: Vec<(String, Self)> = match self {
            Self::Borrowed(Value::Struct(value)) => value
                .fields
                .iter()
                .map(|(name, value)| (name.to_owned(), Self::Borrowed(value)))
                .collect(),
            Self::Owned(Value::Struct(value)) => value
                .fields
                .iter()
                .map(|(name, value)| (name.to_owned(), Self::Owned(value.clone())))
                .collect(),
            Self::Borrowed(Value::Object(value)) => value
                .properties()
                .iter()
                .map(|(name, value)| (name.to_owned(), Self::Borrowed(value)))
                .collect(),
            Self::Owned(Value::Object(value)) => value
                .properties()
                .iter()
                .map(|(name, value)| (name.to_owned(), Self::Owned(value.clone())))
                .collect(),
            Self::SharedCellChild(parent) => match parent.value() {
                Value::Struct(value) => value
                    .fields
                    .iter()
                    .map(|(name, _)| {
                        (
                            name.to_owned(),
                            Self::SharedCellChild(
                                parent
                                    .project(|value| match value {
                                        Value::Struct(value) => value.fields.get(name),
                                        _ => None,
                                    })
                                    .expect("retained struct child"),
                            ),
                        )
                    })
                    .collect(),
                Value::Object(value) => value
                    .properties()
                    .keys()
                    .map(|name| {
                        (
                            name.to_owned(),
                            Self::SharedCellChild(
                                parent
                                    .project(|value| match value {
                                        Value::Object(value) => value.properties().get(name),
                                        _ => None,
                                    })
                                    .expect("retained object child"),
                            ),
                        )
                    })
                    .collect(),
                _ => unreachable!("selected named input"),
            },
            _ => unreachable!("selected named input"),
        };
        fields.sort_by(|left, right| left.0.cmp(&right.0));
        fields
    }

    fn capture_children(self) -> Vec<Self> {
        match self {
            Self::Borrowed(Value::Closure(value)) => {
                value.captures.iter().map(Self::Borrowed).collect()
            }
            Self::Owned(Value::Closure(value)) => {
                let captures = value.into_data().captures;
                match captures.try_into_owned() {
                    Ok(values) => values.into_iter().map(Self::Owned).collect(),
                    Err(owner) => (0..owner.len())
                        .map(|index| {
                            Self::SharedCellChild(
                                owner
                                    .retain_value_at(index)
                                    .expect("shared closure capture"),
                            )
                        })
                        .collect(),
                }
            }
            Self::SharedCellChild(parent) => {
                let Value::Closure(closure) = parent.value() else {
                    unreachable!("selected closure input")
                };
                (0..closure.captures.len())
                    .map(|index| {
                        Self::SharedCellChild(
                            parent
                                .project(|value| match value {
                                    Value::Closure(closure) => closure.captures.get(index),
                                    _ => None,
                                })
                                .expect("retained closure capture"),
                        )
                    })
                    .collect()
            }
            _ => unreachable!("selected closure input"),
        }
    }
}

enum Task<'a> {
    Value {
        input: Input<'a>,
        index: GraphNodeRef,
    },
    Activation {
        environment: LexicalEnvironment,
        index: GraphNodeRef,
    },
}

struct ProvisionalActivation {
    parent: Option<GraphNodeRef>,
    bindings: Vec<GraphBinding>,
}

struct PreparedBoundCallable {
    index: GraphNodeRef,
    origin: FunctionBindingOrigin,
    qualified_name: String,
    callable_digest: runmat_execution::Digest,
    form: runmat_execution::value::CallableForm,
    source_role: runmat_types::FunctionSourceRole,
    source_file: Option<String>,
    captures: Vec<GraphNodeRef>,
    activation: Option<GraphNodeRef>,
}

struct PreparedSelectedObject {
    index: GraphNodeRef,
    origin: FunctionBindingOrigin,
    class: runmat_types::ClassIdentity,
    kind: runmat_execution::value::GraphSelectedObjectKind,
    fields: Vec<runmat_execution::value::GraphField>,
    dynamic_properties: Vec<runmat_execution::value::DurableDynamicProperty>,
    dynamic_metadata: Vec<GraphNodeRef>,
}

struct Encoder<'a> {
    issuer: &'a dyn InlineSourceAuthority,
    workspace: Option<crate::execution::value_codec::ValidationWorkspaceOwner>,
    limits: ValueLimits,
    nodes: Vec<Option<GraphNode>>,
    activations: HashMap<GcHandle, GraphNodeRef>,
    provisional_activations: HashMap<GraphNodeRef, ProvisionalActivation>,
    pending_leaves: Vec<(GraphNodeRef, Input<'a>)>,
    pending_bound_targets: Vec<(GraphNodeRef, FunctionBindingOrigin)>,
    pending_bound_callables: Vec<PreparedBoundCallable>,
    pending_selected_objects: Vec<PreparedSelectedObject>,
    selected_handle_targets: HashMap<GcHandle, GraphNodeRef>,
    selected_handle_nodes: HashSet<GraphNodeRef>,
    callable_instances: HashMap<runmat_value::CallableInstanceKey, GraphNodeRef>,
    bound_owners: Vec<(FunctionBindingOrigin, FunctionDefinitionLease)>,
    bound_authorities: Vec<(GraphNodeRef, FunctionBindingOrigin, BoundFunctionAuthority)>,
}

impl<'a> Encoder<'a> {
    fn resolve_activation_owners(
        &mut self,
        layouts: &HashMap<GraphBoundOwner, GraphLexicalLayout>,
    ) -> Result<(), ValueCodecError> {
        let mut owners = HashMap::<GraphNodeRef, GraphBoundOwner>::new();
        let mut pending = Vec::new();
        for node in &self.nodes {
            let Some(GraphNode::Callable(callable)) = node else {
                continue;
            };
            if let Some(activation) = callable.activation {
                let GraphCallableOwner::Bound(origin) = callable.owner else {
                    return Err(invalid("unbound callable carries a lexical activation"));
                };
                let parent = layouts
                    .get(&origin)
                    .and_then(|layout| layout.activation_parent)
                    .ok_or_else(|| invalid("closure lacks an admitted lexical parent"))?;
                pending.push((activation, parent));
            }
        }
        while let Some((reference, owner)) = pending.pop() {
            if let Some(known) = owners.insert(reference, owner) {
                if known != owner {
                    return Err(invalid("one activation has conflicting admitted owners"));
                }
                continue;
            }
            let provisional = self
                .provisional_activations
                .get(&reference)
                .ok_or_else(|| invalid("callable activation was not snapshotted"))?;
            let layout = layouts
                .get(&owner)
                .ok_or_else(|| invalid("activation owner is absent from admitted layouts"))?;
            let expected = layout
                .bindings
                .iter()
                .filter_map(|(binding, kind)| {
                    (*kind == GraphBindingKind::Owned).then_some(*binding)
                })
                .collect::<Vec<_>>();
            if provisional
                .bindings
                .iter()
                .map(|binding| binding.binding)
                .collect::<Vec<_>>()
                != expected
            {
                return Err(invalid(
                    "activation reservations disagree with admitted layout",
                ));
            }
            match (provisional.parent, layout.activation_parent) {
                (Some(parent), Some(parent_owner)) => pending.push((parent, parent_owner)),
                (None, None) => {}
                _ => return Err(invalid("activation parent disagrees with admitted layout")),
            }
        }
        if owners.len() != self.provisional_activations.len() {
            return Err(invalid("unclaimed activation is present in the payload"));
        }
        for (reference, provisional) in self.provisional_activations.drain() {
            let owner = owners[&reference];
            self.nodes[reference.0 as usize] = Some(GraphNode::Activation(GraphActivation {
                owner,
                parent: provisional.parent,
                bindings: provisional.bindings,
            }));
        }
        Ok(())
    }

    fn reserve(&mut self) -> Result<GraphNodeRef, ValueCodecError> {
        if self.nodes.len() as u64 >= self.limits.max_nodes {
            return Err(invalid("payload graph exceeds its node limit"));
        }
        let index = u32::try_from(self.nodes.len())
            .map_err(|_| invalid("payload graph has too many local ordinals"))?;
        self.nodes.push(None);
        Ok(GraphNodeRef(index))
    }

    fn push_values(
        &mut self,
        inputs: Vec<Input<'a>>,
        work: &mut Vec<Task<'a>>,
    ) -> Result<Vec<GraphNodeRef>, ValueCodecError> {
        let mut refs = Vec::with_capacity(inputs.len());
        for input in inputs {
            let callable = input
                .rooted_value()
                .and_then(runmat_value::Value::callable_instance_key);
            if let Some(instance) = callable {
                if let Some(index) = self.callable_instances.get(&instance).copied() {
                    refs.push(index);
                    continue;
                }
            }
            let handle = match &input {
                Input::Borrowed(Value::HandleObject(handle))
                | Input::Owned(Value::HandleObject(handle))
                    if handle.valid =>
                {
                    Some(handle.target)
                }
                Input::SharedCellChild(_) => match input.value() {
                    Value::HandleObject(handle) if handle.valid => Some(handle.target),
                    _ => None,
                },
                _ => None,
            };
            if let Some(handle) = handle {
                if let Some(index) = self.selected_handle_targets.get(&handle).copied() {
                    refs.push(index);
                    continue;
                }
            }
            let index = self.reserve()?;
            if let Some(handle) = handle {
                self.selected_handle_targets.insert(handle, index);
                self.selected_handle_nodes.insert(index);
            }
            if let Some(instance) = callable {
                self.callable_instances.insert(instance, index);
            }
            refs.push(index);
            work.push(Task::Value { input, index });
        }
        Ok(refs)
    }

    fn push_child_values(
        &mut self,
        inputs: Vec<Input<'a>>,
        work: &mut Vec<Task<'a>>,
    ) -> Result<Vec<GraphNodeRef>, ValueCodecError> {
        self.push_values(inputs, work)
    }

    fn push_activation(
        &mut self,
        environment: LexicalEnvironment,
        work: &mut Vec<Task<'a>>,
    ) -> Result<GraphNodeRef, ValueCodecError> {
        if let Some(index) = self.activations.get(&environment.handle()) {
            return Ok(*index);
        }
        let index = self.reserve()?;
        self.activations.insert(environment.handle(), index);
        work.push(Task::Activation { environment, index });
        Ok(index)
    }

    fn process(&mut self, task: Task<'a>, work: &mut Vec<Task<'a>>) -> Result<(), ValueCodecError> {
        match task {
            Task::Value { input, index } => self.process_value(input, index, work),
            Task::Activation { environment, index } => {
                self.process_activation(environment, index, work)
            }
        }
    }
}
