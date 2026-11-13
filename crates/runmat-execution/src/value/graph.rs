//! One invocation-scoped graph for portable values and shared lexical cells.
//! Ordinals are local to this payload; no runtime handle or session identity is
//! part of the wire contract. The runtime codec owns snapshot and reconstruction.

use std::collections::HashMap;

use runmat_types::BindingId;
use runmat_types::{BuiltinId, ParallelRegionId};
use serde::{Deserialize, Serialize};

use super::{CallableForm, CallableOwner, InlineValue, ValueLimits, ValuePayload};
use crate::{ContractError, ExecutableSourceMapV2, FunctionBindingOrigin};

#[cfg(test)]
mod tests;
mod validation;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GraphNodeRef(pub u32);

/// Typed roles of roots in one invocation graph. Artifact admission validates
/// these roles against its requested execution context.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GraphInvocationRoots {
    Callable {
        target: GraphNodeRef,
        arguments: Vec<GraphNodeRef>,
    },
    Builtin {
        name: BuiltinId,
        arguments: Vec<GraphNodeRef>,
    },
    ParallelRegion {
        region: ParallelRegionId,
        captures: Vec<GraphNodeRef>,
    },
}

/// Position in the enclosing request's ordered, admitted V3 unit bundle.
/// This ordinal is meaningful only together with the exact admitted unit.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GraphProgramUnitId(pub u16);

/// Coordinate representability only. The artifact and scheduler enforce
/// aggregate selected-code and resource budgets for the actual deployment.
pub const GRAPH_MAX_SELECTED_UNITS: usize = u16::MAX as usize + 1;

pub const GRAPH_REQUEST_SCHEMA_V10: u16 = 10;
pub const GRAPH_REQUEST_SCHEMA_V11: u16 = 11;
/// Callable roots carry only explicitly supplied arguments. Captures remain
/// on the callable target and are not repeated in invocation arguments.
pub const GRAPH_REQUEST_SCHEMA_V12: u16 = 12;
/// Digest-bound project-only runtime policy on a remotely executed Graph.
pub const GRAPH_REQUEST_SCHEMA_V13: u16 = 13;
/// Exact captured invocation environment and bounded console controls.
pub const GRAPH_REQUEST_SCHEMA_V14: u16 = 14;

/// Minimum schema required by actual graph nodes. A producer may require V11
/// for selected code capable of returning V11 nodes even when inputs are V10.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum GraphRequiredSchema {
    V10,
    SelectedValuesV11,
    CallableArgumentsV12,
    ProjectRuntimePolicyV13,
    InvocationControlsV14,
}

impl GraphRequiredSchema {
    pub fn from_version(version: u16) -> Option<Self> {
        match version {
            GRAPH_REQUEST_SCHEMA_V10 => Some(Self::V10),
            GRAPH_REQUEST_SCHEMA_V11 => Some(Self::SelectedValuesV11),
            GRAPH_REQUEST_SCHEMA_V12 => Some(Self::CallableArgumentsV12),
            GRAPH_REQUEST_SCHEMA_V13 => Some(Self::ProjectRuntimePolicyV13),
            GRAPH_REQUEST_SCHEMA_V14 => Some(Self::InvocationControlsV14),
            _ => None,
        }
    }

    pub fn version(self) -> u16 {
        match self {
            Self::V10 => GRAPH_REQUEST_SCHEMA_V10,
            Self::SelectedValuesV11 => GRAPH_REQUEST_SCHEMA_V11,
            Self::CallableArgumentsV12 => GRAPH_REQUEST_SCHEMA_V12,
            Self::ProjectRuntimePolicyV13 => GRAPH_REQUEST_SCHEMA_V13,
            Self::InvocationControlsV14 => GRAPH_REQUEST_SCHEMA_V14,
        }
    }

    pub fn required_capability(self) -> Option<crate::resource::Capability> {
        match self {
            Self::V10 => None,
            Self::SelectedValuesV11 => Some(crate::resource::Capability::GraphSelectedValuesV11),
            Self::CallableArgumentsV12 => {
                Some(crate::resource::Capability::GraphCallableArgumentsV12)
            }
            Self::ProjectRuntimePolicyV13 => {
                Some(crate::resource::Capability::GraphProjectRuntimePolicyV13)
            }
            Self::InvocationControlsV14 => {
                Some(crate::resource::Capability::GraphInvocationControlsV14)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphBoundOwner {
    pub unit: GraphProgramUnitId,
    pub origin: FunctionBindingOrigin,
}

/// A declaration-only source product selected by one exact callable unit.
/// It carries no callable origin or fabricated function ordinal.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphClassDependency {
    pub caller: GraphBoundOwner,
    pub declaration_unit: GraphProgramUnitId,
    pub declaration_digest: crate::Digest,
}

/// One exact ordinary method body selected by a constructor's checked
/// definition graph and class declaration. Unit-qualified target identity is
/// required even when two revisions expose the same ClassMethodOwner.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphClassMethodDependency {
    pub caller: GraphBoundOwner,
    pub method: runmat_types::ClassMethodOwner,
    pub target: GraphBoundOwner,
    pub target_unit_digest: crate::Digest,
    pub target_body_digest: crate::executable::SelectedBodySemanticDigest,
}

/// One compiler-bound static reference in an admitted caller body. The
/// instruction and occurrence ordinals are local to that exact body, not
/// session FunctionIds. Admission checks the referenced instruction before
/// using the target's independently admitted unit and digest.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphStaticReferenceSite {
    pub instruction: u32,
    pub occurrence: u16,
}

/// Which selected executable stream owns a sealed static reference. A
/// parallel-region request executes its primary script; ordinary callback
/// bodies use an exact unit-qualified definition instead.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "owner",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum GraphStaticCaller {
    Script(GraphProgramUnitId),
    Function(GraphBoundOwner),
}

/// A static call/handle edge crossing two exact selected code products.
/// Identical origins may appear in separate units from separate revisions;
/// neither an origin-only nor a numeric-ID lookup is sufficient authority.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphStaticDependency {
    pub caller: GraphStaticCaller,
    pub site: GraphStaticReferenceSite,
    pub target: GraphBoundOwner,
    /// SHA-256 of the target's canonical admitted V3 bytes. Artifact
    /// admission can check this without importing VM bytecode semantics.
    pub target_unit_digest: crate::Digest,
    /// Semantic fingerprint of the target body inside that admitted unit.
    /// VM admission checks the selected function and this fingerprint.
    pub target_body_digest: crate::executable::SelectedBodySemanticDigest,
}

/// Graph-specific callable ownership retains the selected unit coordinate;
/// ordinary InlineValue callable identity remains unchanged.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "binding")]
pub enum GraphCallableOwner {
    Workspace,
    External,
    Method,
    Bound(GraphBoundOwner),
}

impl GraphCallableOwner {
    pub fn portable_owner(self) -> CallableOwner {
        match self {
            Self::Workspace => CallableOwner::Workspace,
            Self::External => CallableOwner::External,
            Self::Method => CallableOwner::Method,
            Self::Bound(owner) => CallableOwner::Bound(owner.origin),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadGraph {
    /// Roles and ordering belong to the enclosing artifact request or response,
    /// not to this generic graph.
    pub roots: Vec<GraphNodeRef>,
    pub nodes: Vec<GraphNode>,
}

impl PayloadGraph {
    pub fn required_schema(&self) -> GraphRequiredSchema {
        if self.nodes.iter().any(|node| {
            matches!(
                node,
                GraphNode::SelectedObjectArray(_)
                    | GraphNode::SelectedDynamicPropertyMetadata(_)
            ) || matches!(node, GraphNode::SelectedObject(object) if !object.dynamic_metadata.is_empty())
        }) {
            GraphRequiredSchema::SelectedValuesV11
        } else {
            GraphRequiredSchema::V10
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "node", rename_all = "snake_case", deny_unknown_fields)]
pub enum GraphNode {
    /// A canonical ValuePayload with no nested values or callable. Numeric,
    /// sparse, exception and fenced-reference rules remain in its codec.
    Leaf {
        value: ValuePayload,
    },
    Cell {
        shape: Vec<u64>,
        values: Vec<GraphNodeRef>,
    },
    Struct {
        fields: Vec<GraphField>,
    },
    Registered(GraphRegistered),
    SelectedObject(GraphSelectedObject),
    SelectedObjectArray(GraphSelectedObjectArray),
    SelectedDynamicPropertyMetadata(GraphSelectedDynamicPropertyMetadata),
    Callable(GraphCallable),
    Activation(GraphActivation),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphField {
    pub name: String,
    pub value: GraphNodeRef,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphRegisteredKind {
    Symbolic,
    ImmutableValueClass,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphRegistered {
    pub kind: GraphRegisteredKind,
    pub type_identity: String,
    pub schema_version: u32,
    pub fields: Vec<GraphField>,
}

/// An object of one exact selected class product. Its constructor owner
/// selects admitted code and ancestry; the class spelling alone has no
/// execution authority. Repeated references to a Handle node share one GC
/// target when reconstructed.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphSelectedObject {
    pub constructor: GraphBoundOwner,
    pub class: runmat_types::ClassIdentity,
    pub kind: GraphSelectedObjectKind,
    pub fields: Vec<GraphField>,
    pub dynamic_properties: Vec<super::DurableDynamicProperty>,
    /// Metadata handles retain this receiver through graph-local references.
    #[serde(default)]
    pub dynamic_metadata: Vec<GraphNodeRef>,
}

/// A metadata handle selected for one exact receiver and dynamic property.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphSelectedDynamicPropertyMetadata {
    pub receiver: GraphNodeRef,
    pub property: String,
}

/// Selected array provenance, including an empty array with no element from
/// which an exact old/new constructor product could otherwise be recovered.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphSelectedObjectArray {
    pub class: runmat_types::ClassIdentity,
    pub kind: GraphSelectedObjectKind,
    pub shape: Vec<u64>,
    pub values: Vec<GraphNodeRef>,
    /// Exact constructor callable anchors, not a class-name lookup.
    pub empty_provenance: Vec<GraphNodeRef>,
    pub includes_unselected: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphSelectedObjectKind {
    Value,
    Handle,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphCallable {
    pub owner: GraphCallableOwner,
    pub qualified_name: String,
    pub callable_digest: crate::Digest,
    pub form: CallableForm,
    pub source_role: runmat_types::FunctionSourceRole,
    pub source_file: Option<String>,
    pub captures: Vec<GraphNodeRef>,
    pub activation: Option<GraphNodeRef>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphBindingKind {
    Owned,
    Inherited,
    Transit,
}

/// An admitted bytecode owner supplies this exact lexical layout. A payload
/// does not get to certify its own layout by carrying a digest or display name.
/// Global/persistent inherited storage is not a lexical cell: worker service
/// admission must handle it separately, without implicit host writeback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphLexicalLayout {
    /// Nearest ancestor which actually created an activation, derived from
    /// admitted lexical ancestry rather than the graph's claimed parent.
    pub activation_parent: Option<GraphBoundOwner>,
    pub bindings: Vec<(BindingId, GraphBindingKind)>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphBinding {
    pub binding: BindingId,
    /// None is an owned but not-yet-assigned cell; it is not a missing owner.
    pub value: Option<GraphNodeRef>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphActivation {
    pub owner: GraphBoundOwner,
    pub parent: Option<GraphNodeRef>,
    /// Only owned cells live here. Inherited/transit bindings resolve through
    /// the admitted parent chain; their numeric BindingId alone is not proof.
    pub bindings: Vec<GraphBinding>,
}

impl PayloadGraph {
    /// Pre-admission range check for graph-local unit coordinates. This is
    /// only a structural check: the VM must still prove each unit's exact
    /// selected body, lexical layout, and source before constructing Values.
    pub fn validate_selected_unit_references(
        &self,
        selected_unit_count: usize,
    ) -> Result<(), ContractError> {
        if selected_unit_count == 0 || selected_unit_count > GRAPH_MAX_SELECTED_UNITS {
            return Err(ContractError::invalid(
                "payload graph",
                "selected graph unit count is invalid",
            ));
        }
        for node in &self.nodes {
            let selected = match node {
                GraphNode::Callable(GraphCallable {
                    owner: GraphCallableOwner::Bound(owner),
                    ..
                }) => Some(owner.unit),
                GraphNode::Activation(activation) => Some(activation.owner.unit),
                GraphNode::SelectedObject(object) => Some(object.constructor.unit),
                _ => None,
            };
            if selected.is_some_and(|unit| usize::from(unit.0) >= selected_unit_count) {
                return Err(ContractError::invalid(
                    "payload graph",
                    "graph owner selects a unit outside the admitted bundle",
                ));
            }
        }
        Ok(())
    }

    /// Conservative numeric allocation demand after by-value fanout. This
    /// checks the Value-edge DAG without interpreting activation links or
    /// trusting payload claims about executable ownership. The caller still
    /// needs full selected-program validation before reconstruction.
    pub fn expanded_numeric_allocation_bytes(&self) -> Result<u64, ContractError> {
        validation::expanded_numeric_allocation_bytes(self)
    }

    /// Structural, source and exact admitted-layout validation before effects.
    /// `layouts` must be issued from the selected executable's authenticated
    /// function bodies, never copied from this payload or a caller assertion.
    pub fn validate_for_program(
        &self,
        limits: ValueLimits,
        source_map: Option<&ExecutableSourceMapV2>,
        layouts: &HashMap<GraphBoundOwner, GraphLexicalLayout>,
        selected_unit_count: usize,
    ) -> Result<(), ContractError> {
        self.validate_for_program_and_value_order(limits, source_map, layouts, selected_unit_count)
            .map(|_| ())
    }

    /// Perform the same full contract validation and return Value nodes in
    /// child-before-parent order for an allocating consumer. Activation shells
    /// are excluded: only typed activation links may form a cycle. This paired
    /// result prevents a decoder from walking or trusting a separate graph.
    pub fn validate_for_program_and_value_order(
        &self,
        limits: ValueLimits,
        source_map: Option<&ExecutableSourceMapV2>,
        layouts: &HashMap<GraphBoundOwner, GraphLexicalLayout>,
        selected_unit_count: usize,
    ) -> Result<Vec<GraphNodeRef>, ContractError> {
        validation::validate(self, limits, source_map, layouts, selected_unit_count)
    }

    /// Exact immutable function owners reachable through callable and
    /// activation nodes. The admitted leaf grammar contains no callables.
    pub fn binding_origins(
        &self,
        limits: ValueLimits,
        source_map: Option<&ExecutableSourceMapV2>,
        layouts: &HashMap<GraphBoundOwner, GraphLexicalLayout>,
        selected_unit_count: usize,
    ) -> Result<std::collections::HashSet<GraphBoundOwner>, ContractError> {
        self.validate_for_program(limits, source_map, layouts, selected_unit_count)?;
        let mut origins = std::collections::HashSet::new();
        for node in &self.nodes {
            match node {
                GraphNode::Callable(GraphCallable {
                    owner: GraphCallableOwner::Bound(origin),
                    ..
                }) => {
                    origins.insert(*origin);
                }
                GraphNode::Activation(activation) => {
                    origins.insert(activation.owner);
                }
                GraphNode::SelectedObject(object) => {
                    origins.insert(object.constructor);
                }
                _ => {}
            }
        }
        Ok(origins)
    }
}

fn is_leaf(value: &ValuePayload) -> bool {
    match value {
        ValuePayload::Inline(value) => match value.as_ref() {
            InlineValue::Null
            | InlineValue::Logical(_)
            | InlineValue::F64Bits(_)
            | InlineValue::I8(_)
            | InlineValue::I16(_)
            | InlineValue::I32(_)
            | InlineValue::I64(_)
            | InlineValue::U8(_)
            | InlineValue::U16(_)
            | InlineValue::U32(_)
            | InlineValue::U64(_)
            | InlineValue::ComplexF64Bits { .. }
            | InlineValue::String(_)
            | InlineValue::Char { .. }
            | InlineValue::StringArray { .. }
            | InlineValue::Dense(_)
            | InlineValue::Sparse(_)
            | InlineValue::Exception(_) => true,
            InlineValue::Cell { .. }
            | InlineValue::Struct(_)
            | InlineValue::Symbolic(_)
            | InlineValue::ImmutableValueClass(_)
            | InlineValue::Callable(_)
            | InlineValue::DurableOrderedStruct(_)
            | InlineValue::DurablePlainObject(_)
            | InlineValue::DurableStructArray(_)
            | InlineValue::DurableObjectArray(_) => false,
        },
        ValuePayload::Object(_) | ValuePayload::Distributed(_) | ValuePayload::Composite(_) => true,
    }
}
