#[path = "graph/policy.rs"]
mod policy;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::Cursor;
use std::sync::Arc;

use crate::archive::{read_bundle, ArchiveLimits};
use crate::ExecutionBundle;
use runmat_execution::executable::{
    decode_static_imports, selected_unit_content_digest, ExecutableStaticCaller,
    STATIC_IMPORT_SECTION,
};
use runmat_execution::host::{ExecutionHostRequirement, SelectedUnitHostRequirement};
use runmat_execution::security::ExecutionTrustTier;
use runmat_execution::value::{
    GraphCallableOwner, GraphClassMethodDependency, GraphNode, GraphNodeRef, GraphRequiredSchema,
    GraphStaticCaller, GraphStaticDependency, GraphStaticReferenceSite, PayloadGraph, ValueLimits,
};
use runmat_execution::{
    ExecutableComponentKind, ExecutableSourceMapV2, ExecutableUnitEnvelope, ProgramCallable,
    ProgramExecutionAssignment, ProgramInvocationContext,
};
use serde::{Deserialize, Serialize};

pub use runmat_execution::value::GraphInvocationRoots;

use super::{
    ArtifactError, ArtifactResult, ExecutableForm, JobId, ProgramArtifact, ProgramBuildRecipe,
    ProgramExecutionResponse, MAX_PROGRAM_EXECUTION_ARGUMENTS,
};

fn ordered_roots(roots: &GraphInvocationRoots) -> Vec<GraphNodeRef> {
    match roots {
        GraphInvocationRoots::Callable { target, arguments } => {
            let mut ordered = Vec::with_capacity(arguments.len() + 1);
            ordered.push(*target);
            ordered.extend(arguments);
            ordered
        }
        GraphInvocationRoots::Builtin { arguments, .. } => arguments.clone(),
        GraphInvocationRoots::ParallelRegion { captures, .. } => captures.clone(),
    }
}

fn validate_root_context(
    roots: &GraphInvocationRoots,
    context: &ProgramInvocationContext,
) -> ArtifactResult<()> {
    let valid = match (roots, context) {
        (GraphInvocationRoots::Callable { .. }, ProgramInvocationContext::Direct) => true,
        (GraphInvocationRoots::Builtin { .. }, ProgramInvocationContext::Direct) => true,
        (
            GraphInvocationRoots::ParallelRegion { region, .. },
            ProgramInvocationContext::ParallelTask { task },
        ) => task.validate().is_ok() && task.region == *region,
        _ => false,
    };
    valid.then_some(()).ok_or_else(|| {
        ArtifactError::Invalid("graph invocation context disagrees with its root roles".into())
    })
}

/// V10/V11 worker-only graph route. The direct V7 request remains byte-for-byte
/// independent and retains its manifest-entrypoint rule.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphExecutionRequest {
    pub schema_version: u16,
    pub recipe: ProgramBuildRecipe,
    pub artifact: ProgramArtifact,
    /// Additional canonical admitted V3 envelopes in graph-unit order. Unit
    /// zero is `artifact`; callable units are selected by exact definition
    /// lease, while declaration-only units are retained by their caller's
    /// selected class product. The worker admits every envelope and source.
    pub retained_units: Vec<Vec<u8>>,
    /// Exact caller-to-declaration selections; no declaration is callable.
    pub class_dependencies: Vec<runmat_execution::value::GraphClassDependency>,
    /// Exact selected unit for each constructor-owned ordinary method. This
    /// layer checks coordinates and full target V3 bytes; VM admission must
    /// prove the constructor, method body and semantic digest before effects.
    #[serde(default)]
    pub class_method_dependencies: Vec<GraphClassMethodDependency>,
    /// One exact selected target for each sealed cross-unit static reference.
    /// This layer authenticates the sealed caller claim and full target V3;
    /// it cannot inspect VM bytecode. VM admission must prove the target body
    /// exists with the declared semantic fingerprint before any effect.
    #[serde(default)]
    pub static_dependencies: Vec<GraphStaticDependency>,
    /// Canonical archives in exact selected-unit order, including unit zero.
    /// Truly pure units use `None`; policy-bearing units retain `Some` even
    /// when their selected interop is empty.
    /// These bytes are part of the request's stored/remote identity, never a
    /// later sidecar or ambient project lookup.
    #[serde(default)]
    pub selected_bundle_archives: Vec<Option<Vec<u8>>>,
    /// Ordered compact commitments from the exact issued receipt for each
    /// selected unit. An archive cannot be omitted merely because a retained
    /// V3 unit declares empty interop. Pure legacy requests omit this field.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub selected_foreign_policies: Vec<Option<runmat_execution::ForeignPolicyRef>>,
    pub graph: PayloadGraph,
    pub roots: GraphInvocationRoots,
    pub context: ProgramInvocationContext,
    pub assignment: Option<ProgramExecutionAssignment>,
    pub job_id: Option<JobId>,
    pub requested_outputs: u16,
}

/// One selected Graph request with its ordered foreign bundles admitted exactly once.
///
/// A host can borrow the bundles to create private materialized roots, then
/// pass the same owned receipt to the VM. Its private constructor prevents a
/// raw request and unrelated decoded archives from being paired.
#[derive(Debug)]
pub struct ValidatedGraphExecutionRequest {
    request: GraphExecutionRequest,
    bundles: Vec<Option<ExecutionBundle>>,
    admission: Arc<()>,
}

/// The same admitted Graph request after the host has materialized its private
/// foreign-object roots and released the decoded bundle objects. The archive
/// admission fact remains typed through VM execution and response validation.
#[derive(Debug)]
pub struct ArchiveAdmittedGraphExecutionRequest {
    request: GraphExecutionRequest,
    admission: Arc<()>,
}

/// One decoded bundle paired with its exact admitted Graph request and unit.
/// Neither the selected unit index nor the admission identity is caller-made.
#[derive(Debug)]
pub struct AdmittedGraphUnitBundle {
    unit_index: usize,
    bundle: Option<ExecutionBundle>,
    admission: Arc<()>,
}

/// Compact response authority issued only after the complete Graph request is
/// admitted. A local process parent can retain it across attempts without
/// re-reading or cloning the archive-bearing request for each response.
#[derive(Debug)]
pub struct GraphResponseAuthority {
    source_map: ExecutableSourceMapV2,
    selected_unit_count: usize,
    requested_outputs: u16,
    required_schema: GraphRequiredSchema,
}

impl ValidatedGraphExecutionRequest {
    pub fn request(&self) -> &GraphExecutionRequest {
        &self.request
    }

    pub fn selected_bundles(&self) -> &[Option<ExecutionBundle>] {
        &self.bundles
    }

    /// The request was admitted by this type's private constructor. Check the
    /// response against that same selected-unit contract without decoding its
    /// bundle archives a second time.
    pub fn validate_response(&self, response: &ProgramExecutionResponse) -> ArtifactResult<()> {
        self.request.validate_admitted_response(response)
    }

    pub fn response_authority(&self) -> ArtifactResult<GraphResponseAuthority> {
        Ok(GraphResponseAuthority {
            source_map: self.request.selected_source_map()?,
            selected_unit_count: self.request.retained_units.len() + 1,
            requested_outputs: self.request.requested_outputs,
            required_schema: GraphRequiredSchema::from_version(self.request.schema_version)
                .ok_or_else(|| ArtifactError::Invalid("unsupported graph schema".into()))?,
        })
    }

    /// Release decoded logical objects after the host has taken ownership of
    /// its private materialized roots, without falling back to a raw request.
    pub fn release_bundles(self) -> ArchiveAdmittedGraphExecutionRequest {
        ArchiveAdmittedGraphExecutionRequest {
            request: self.request,
            admission: self.admission,
        }
    }

    /// Transfer the already-admitted bundles to the VM after the host has
    /// borrowed them for private materialization. The VM can bind each unit
    /// with its exact bundle without decoding the request archives again;
    /// only the request marker needs to survive execution.
    pub fn into_admitted_with_bundles(
        self,
    ) -> (
        ArchiveAdmittedGraphExecutionRequest,
        Vec<AdmittedGraphUnitBundle>,
    ) {
        let bundles = self
            .bundles
            .into_iter()
            .enumerate()
            .map(|(unit_index, bundle)| AdmittedGraphUnitBundle {
                unit_index,
                bundle,
                admission: Arc::clone(&self.admission),
            })
            .collect();
        (
            ArchiveAdmittedGraphExecutionRequest {
                request: self.request,
                admission: self.admission,
            },
            bundles,
        )
    }
}

impl ArchiveAdmittedGraphExecutionRequest {
    pub fn request(&self) -> &GraphExecutionRequest {
        &self.request
    }

    pub fn validate_response(&self, response: &ProgramExecutionResponse) -> ArtifactResult<()> {
        self.request.validate_admitted_response(response)
    }

    /// Consume only the bundle issued for this precise request and unit. A
    /// same-code bundle from another admission cannot substitute its foreign
    /// objects, and a caller cannot silently reorder retained-unit payloads.
    pub fn consume_bundle_for(
        &self,
        unit_index: usize,
        admitted: AdmittedGraphUnitBundle,
    ) -> ArtifactResult<Option<ExecutionBundle>> {
        if unit_index > self.request.retained_units.len()
            || admitted.unit_index != unit_index
            || !Arc::ptr_eq(&self.admission, &admitted.admission)
        {
            return Err(ArtifactError::Invalid(
                "selected Graph bundle differs from its admitted unit".into(),
            ));
        }
        Ok(admitted.bundle)
    }
}

impl GraphResponseAuthority {
    pub fn validate_response(&self, response: &ProgramExecutionResponse) -> ArtifactResult<()> {
        validate_graph_response_against(
            response,
            &self.source_map,
            self.selected_unit_count,
            self.requested_outputs,
            self.required_schema,
        )
    }
}

impl GraphExecutionRequest {
    pub fn into_validated(self) -> ArtifactResult<ValidatedGraphExecutionRequest> {
        let bundles = self.validate_and_admit_bundles()?;
        Ok(ValidatedGraphExecutionRequest {
            request: self,
            bundles,
            admission: Arc::new(()),
        })
    }

    /// A placement identity, not a dispatch authority. The VM executes the
    /// reconstructed root after admitting the exact selected V3 products.
    pub fn scheduler_callable(&self) -> ArtifactResult<ProgramCallable> {
        match &self.roots {
            GraphInvocationRoots::Callable { target, .. } => {
                let Some(GraphNode::Callable(callable)) = self.graph.nodes.get(target.0 as usize)
                else {
                    return Err(ArtifactError::Invalid(
                        "graph scheduler target is not a callable node".into(),
                    ));
                };
                Ok(ProgramCallable::graph_target(callable.callable_digest))
            }
            GraphInvocationRoots::Builtin { name, .. } => {
                let builtin = ProgramCallable::Builtin { name: name.clone() };
                builtin.validate().map_err(super::contract_error)?;
                Ok(builtin)
            }
            GraphInvocationRoots::ParallelRegion { region, .. } => {
                Ok(ProgramCallable::parallel_region(*region))
            }
        }
    }

    /// A scheduler floor for the actual graph materialization, independent of
    /// `TaskRequest.inputs` (which is empty to avoid copying the graph). This
    /// counts the stored and in-flight wire buffers, selected executable bytes,
    /// graph-node allocation, and canonical numeric leaf buffers. It is a
    /// reservation floor, not an estimate of total execution memory.
    pub fn input_allocation_floor_bytes(&self, wire_bytes: usize) -> ArtifactResult<u64> {
        let numeric_bytes = self
            .graph
            .expanded_numeric_allocation_bytes()
            .map_err(super::contract_error)?;
        let selected_code_bytes = self.retained_units.iter().try_fold(
            self.artifact.executable_bytes.len(),
            |total, bytes| {
                total.checked_add(bytes.len()).ok_or_else(|| {
                    ArtifactError::Limit("graph selected-code allocation overflows".into())
                })
            },
        )?;
        let archive_bytes =
            self.selected_bundle_archives
                .iter()
                .try_fold(0usize, |total, archive| {
                    total
                        .checked_add(archive.as_ref().map_or(0, Vec::len))
                        .ok_or_else(|| ArtifactError::Limit("graph bundle bytes overflow".into()))
                })?;
        let node_storage = self
            .graph
            .nodes
            .len()
            .checked_mul(std::mem::size_of::<GraphNode>())
            .ok_or_else(|| ArtifactError::Limit("graph node allocation overflows".into()))?;
        let total =
            (1024usize * 1024)
                .checked_add(wire_bytes.checked_mul(2).ok_or_else(|| {
                    ArtifactError::Limit("graph wire allocation overflows".into())
                })?)
                .and_then(|total| total.checked_add(selected_code_bytes))
                // The wire buffer and decoded logical-object buffers coexist
                // during archive admission; materialization retains the
                // authenticated archive until the selected host retires.
                .and_then(|total| {
                    archive_bytes
                        .checked_mul(2)
                        .and_then(|bytes| total.checked_add(bytes))
                })
                .and_then(|total| total.checked_add(node_storage))
                .ok_or_else(|| ArtifactError::Limit("graph input allocation overflows".into()))?;
        u64::try_from(total)
            .ok()
            .and_then(|total| total.checked_add(numeric_bytes))
            .ok_or_else(|| ArtifactError::Limit("graph input allocation exceeds u64".into()))
    }

    pub fn validate(&self) -> ArtifactResult<()> {
        self.validate_and_admit_bundles().map(|_| ())
    }

    /// Validate the whole Graph request and return its already-decoded selected
    /// bundles to the worker host. This avoids a second archive decode before
    /// materialization while keeping `validate` as the public check API.
    pub fn validate_and_admit_bundles(&self) -> ArtifactResult<Vec<Option<ExecutionBundle>>> {
        if GraphRequiredSchema::from_version(self.schema_version)
            .is_none_or(|schema| schema < self.graph.required_schema())
        {
            return Err(ArtifactError::Invalid(
                "graph request schema cannot admit its selected value features".into(),
            ));
        }
        let admitted = self.artifact.validate_context(&self.recipe)?;
        self.artifact.target.validate_for_portable_host()?;
        if self.artifact.form != ExecutableForm::ExecutableUnitV3
            || admitted.unit_admission.is_none()
            || self.requested_outputs != self.recipe.outputs.requested_outputs
            || self
                .assignment
                .as_ref()
                .is_some_and(|assignment| assignment.validate().is_err())
            || ordered_roots(&self.roots) != self.graph.roots
            || self.graph.roots.len() > MAX_PROGRAM_EXECUTION_ARGUMENTS + 1
        {
            return Err(ArtifactError::Invalid(
                "graph request has an inconsistent selected unit, roots, or output contract".into(),
            ));
        }
        validate_root_context(&self.roots, &self.context)?;
        if let GraphInvocationRoots::Callable { target, .. } = &self.roots {
            if !matches!(
                self.graph.nodes.get(target.0 as usize),
                Some(GraphNode::Callable(_))
            ) {
                return Err(ArtifactError::Invalid(
                    "graph callable target is not a callable node".into(),
                ));
            }
        }
        let _source_map = self.selected_source_map()?;
        // The recipe authenticates the primary unit. Validate each retained
        // interop contract independently; equal logical names in distinct
        // selected revisions are not one foreign binding namespace.
        let _selected = self.selected_unit_host_requirements()?;
        let bundles = self.selected_bundle_admission()?;
        self.graph
            .validate_selected_unit_references(self.retained_units.len() + 1)
            .map_err(super::contract_error)?;
        self.validate_class_dependencies()?;
        self.validate_static_dependencies()?;
        // The VM validates every graph edge and lexical layout against the
        // admitted selected registry before allocating cells or invoking code.
        // The transport owner rejects unbounded size and malformed root roles.
        if self.graph.nodes.len() as u64 > ValueLimits::default().max_nodes {
            return Err(ArtifactError::Limit(
                "graph request exceeds its node bound".into(),
            ));
        }
        Ok(bundles)
    }

    fn validate_class_dependencies(&self) -> ArtifactResult<()> {
        if self.class_dependencies.len() > runmat_execution::value::GRAPH_MAX_SELECTED_UNITS {
            return Err(ArtifactError::Limit(
                "graph class dependencies exceed their shared bound".into(),
            ));
        }
        let units = std::iter::once(self.artifact.executable_bytes.as_slice())
            .chain(self.retained_units.iter().map(Vec::as_slice))
            .collect::<Vec<_>>();
        let callers = self
            .graph
            .nodes
            .iter()
            .filter_map(|node| match node {
                GraphNode::Callable(callable) => match callable.owner {
                    GraphCallableOwner::Bound(owner) => Some(owner),
                    _ => None,
                },
                GraphNode::SelectedObject(object) => Some(object.constructor),
                _ => None,
            })
            .collect::<HashSet<_>>();
        let mut declarations = HashMap::new();
        let mut selected = HashSet::new();
        for edge in &self.class_dependencies {
            let caller = usize::from(edge.caller.unit.0);
            let declaration = usize::from(edge.declaration_unit.0);
            if caller >= units.len() || declaration == 0 || declaration >= units.len() {
                return Err(ArtifactError::Invalid(
                    "graph class dependency selects an absent unit".into(),
                ));
            }
            if !callers.contains(&edge.caller) {
                return Err(ArtifactError::Invalid(
                    "graph class dependency has no selected callable".into(),
                ));
            }
            let (digest, class) = match declarations.entry(declaration) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let bytes = units[declaration];
                    let digest = runmat_execution::Digest::sha256(bytes);
                    let unit = ExecutableUnitEnvelope::from_canonical_bytes(bytes)
                        .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
                    let class = match unit.manifest.identity.validated_entrypoint() {
                        Ok(runmat_execution::ValidatedExecutableEntrypoint::DeclarationOnly(
                            class,
                        )) => class,
                        _ => {
                            return Err(ArtifactError::Invalid(
                                "graph class dependency is not a declaration-only product".into(),
                            ));
                        }
                    };
                    entry.insert((digest, class))
                }
            };
            if *digest != edge.declaration_digest {
                return Err(ArtifactError::Invalid(
                    "graph class dependency differs from its admitted bytes".into(),
                ));
            }
            if !selected.insert((edge.caller, class.clone())) {
                return Err(ArtifactError::Invalid(
                    "graph callable repeats a class declaration dependency".into(),
                ));
            }
        }
        Ok(())
    }

    /// Prove unit-qualified structural coverage of sealed static imports.
    /// `target_body_digest` is a *claim* from the admitted caller section at
    /// this layer; the VM must compare it with the actual admitted target body.
    fn validate_static_dependencies(&self) -> ArtifactResult<()> {
        let bytes = std::iter::once(self.artifact.executable_bytes.as_slice())
            .chain(self.retained_units.iter().map(Vec::as_slice))
            .collect::<Vec<_>>();
        let units = bytes
            .iter()
            .map(|bytes| {
                ExecutableUnitEnvelope::from_canonical_bytes(bytes)
                    .map_err(|error| ArtifactError::Invalid(error.to_string()))
            })
            .collect::<ArtifactResult<Vec<_>>>()?;
        let revisions = units
            .iter()
            .map(|unit| {
                unit.manifest
                    .identity
                    .program
                    .identity_digest()
                    .map_err(|error| ArtifactError::Invalid(error.to_string()))
            })
            .collect::<ArtifactResult<Vec<_>>>()?;
        let digests = bytes
            .iter()
            .map(runmat_execution::Digest::sha256)
            .collect::<Vec<_>>();
        let selected_code_bytes = bytes.iter().try_fold(0_usize, |total, bytes| {
            total
                .checked_add(bytes.len())
                .ok_or_else(|| ArtifactError::Limit("graph selected code bytes overflow".into()))
        })?;
        self.validate_class_method_dependencies(&revisions, &digests, selected_code_bytes)?;
        let sealed_bytes = units
            .iter()
            .flat_map(|unit| &unit.manifest.optional_sections)
            .filter(|section| section.name == STATIC_IMPORT_SECTION)
            .try_fold(0_usize, |total, section| {
                total.checked_add(section.payload.len()).ok_or_else(|| {
                    ArtifactError::Limit("graph sealed static section bytes overflow".into())
                })
            })?;
        if self.static_dependencies.len() > sealed_bytes {
            return Err(ArtifactError::Limit(
                "graph static edges exceed their sealed section bound".into(),
            ));
        }
        let mut content_digests = vec![None; units.len()];
        let mut edges = HashMap::<(GraphStaticCaller, GraphStaticReferenceSite), _>::new();
        for edge in &self.static_dependencies {
            let caller_index = match edge.caller {
                GraphStaticCaller::Script(unit) => usize::from(unit.0),
                GraphStaticCaller::Function(owner) => usize::from(owner.unit.0),
            };
            let target_index = usize::from(edge.target.unit.0);
            if caller_index >= units.len() || target_index >= units.len() {
                return Err(ArtifactError::Invalid(
                    "graph static dependency selects an absent unit".into(),
                ));
            }
            if let GraphStaticCaller::Function(owner) = edge.caller {
                if owner.origin.program != revisions[caller_index] {
                    return Err(ArtifactError::Invalid(
                        "graph static caller differs from its admitted revision".into(),
                    ));
                }
            }
            if edge.target.origin.program != revisions[target_index]
                || edge.target_unit_digest != digests[target_index]
            {
                return Err(ArtifactError::Invalid(
                    "graph static target differs from its admitted unit".into(),
                ));
            }
            if edges.insert((edge.caller, edge.site), edge).is_some() {
                return Err(ArtifactError::Invalid(
                    "graph static dependency repeats a caller site".into(),
                ));
            }
        }
        for (index, unit) in units.iter().enumerate() {
            let unit_id = runmat_execution::value::GraphProgramUnitId(
                u16::try_from(index).map_err(|_| {
                    ArtifactError::Limit("graph static caller unit exceeds its ordinal ABI".into())
                })?,
            );
            let Some(section) = unit
                .manifest
                .optional_sections
                .iter()
                .find(|section| section.name == STATIC_IMPORT_SECTION)
            else {
                continue;
            };
            let imports = decode_static_imports(section)
                .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
            for import in imports {
                let caller = match import.caller {
                    ExecutableStaticCaller::Script => GraphStaticCaller::Script(unit_id),
                    ExecutableStaticCaller::Function(origin) => {
                        if origin.program != revisions[index] {
                            return Err(ArtifactError::Invalid(
                                "sealed static caller differs from its admitted revision".into(),
                            ));
                        }
                        GraphStaticCaller::Function(runmat_execution::value::GraphBoundOwner {
                            unit: unit_id,
                            origin,
                        })
                    }
                };
                let edge = edges.remove(&(caller, import.site)).ok_or_else(|| {
                    ArtifactError::Invalid("sealed static import has no selected graph edge".into())
                })?;
                let target_index = usize::from(edge.target.unit.0);
                let content = match content_digests[target_index] {
                    Some(content) => content,
                    None => {
                        let content = selected_unit_content_digest(&units[target_index])
                            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
                        content_digests[target_index] = Some(content);
                        content
                    }
                };
                if edge.target.origin != import.target
                    || edge.target_body_digest != import.target_body
                    || content != import.target_content
                {
                    return Err(ArtifactError::Invalid(
                        "graph static edge differs from its sealed target".into(),
                    ));
                }
            }
        }
        if !edges.is_empty() {
            return Err(ArtifactError::Invalid(
                "graph static edge has no sealed caller site".into(),
            ));
        }
        Ok(())
    }

    /// Structural coordinates only: the VM owns the exact constructor role,
    /// selected method body and semantic-body fingerprint proof.
    fn validate_class_method_dependencies(
        &self,
        revisions: &[runmat_execution::Digest],
        unit_digests: &[runmat_execution::Digest],
        selected_code_bytes: usize,
    ) -> ArtifactResult<()> {
        if self.class_method_dependencies.len() > selected_code_bytes {
            return Err(ArtifactError::Limit(
                "graph class method edges exceed their selected code bound".into(),
            ));
        }
        let mut seen = HashSet::new();
        for edge in &self.class_method_dependencies {
            let caller = usize::from(edge.caller.unit.0);
            let target = usize::from(edge.target.unit.0);
            if caller >= revisions.len() || target >= revisions.len() {
                return Err(ArtifactError::Invalid(
                    "graph class method edge selects an absent unit".into(),
                ));
            }
            if edge.caller.origin.program != revisions[caller]
                || edge.target.origin.program != revisions[target]
                || edge.target_unit_digest != unit_digests[target]
            {
                return Err(ArtifactError::Invalid(
                    "graph class method edge differs from its admitted V3 unit".into(),
                ));
            }
            if edge.method.role != runmat_types::ClassMethodRole::Ordinary {
                return Err(ArtifactError::Invalid(
                    "graph class method edge does not name an ordinary method".into(),
                ));
            }
            if !seen.insert((edge.caller, edge.method.clone())) {
                return Err(ArtifactError::Invalid(
                    "graph class method edge repeats a constructor method".into(),
                ));
            }
        }
        Ok(())
    }

    /// Exact interop contracts in selected V3 unit order. A worker admits
    /// each unit against its own materialized foreign catalog. Placement may
    /// check all requirements, but must never merge their binding identities.
    pub fn selected_unit_host_requirements(
        &self,
    ) -> ArtifactResult<Vec<SelectedUnitHostRequirement>> {
        self.validate_selected_unit_bytes()?;
        std::iter::once(&self.artifact.executable_bytes)
            .chain(&self.retained_units)
            .map(|bytes| {
                let unit = ExecutableUnitEnvelope::from_canonical_bytes(bytes)
                    .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
                unit.manifest
                    .interop
                    .validate()
                    .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
                Ok(SelectedUnitHostRequirement {
                    environment: unit.manifest.identity.program.environment(),
                    interop: unit.manifest.interop.clone(),
                })
            })
            .collect()
    }

    /// Admit one canonical foreign-object bundle for each selected V3 unit.
    /// The returned bundles are the only objects a worker may materialize;
    /// admission proves their embedded V3 bytes, revision, and interop before
    /// any graph leaf decode or foreign effect.
    pub fn selected_bundle_admission(&self) -> ArtifactResult<Vec<Option<ExecutionBundle>>> {
        self.validate_selected_unit_bytes()?;
        let selected_count = self.retained_units.len() + 1;
        if self.selected_bundle_archives.len() != selected_count {
            return Err(ArtifactError::Invalid(
                "graph bundle archive count differs from selected V3 units".into(),
            ));
        }
        policy::validate_refs(self, selected_count)?;
        let limits = ArchiveLimits::default();
        let mut total_archive_bytes = 0_u64;
        std::iter::once(&self.artifact.executable_bytes)
            .chain(&self.retained_units)
            .zip(&self.selected_bundle_archives)
            .enumerate()
            .map(|(index, (unit_bytes, archive))| {
                let unit = ExecutableUnitEnvelope::from_canonical_bytes(unit_bytes)
                    .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
                let has_interop = !unit.manifest.interop.adapters.is_empty()
                    || !unit.manifest.interop.foreign_types.is_empty()
                    || !unit.manifest.interop.adapter_contracts.is_empty();
                let policy_ref = self.selected_foreign_policies.get(index).copied().flatten();
                match (policy_ref, archive) {
                    (None, None) if !has_interop => Ok(None),
                    (Some(expected_policy), Some(bytes)) => {
                        let length = u64::try_from(bytes.len()).map_err(|_| {
                            ArtifactError::Limit("graph bundle archive cannot fit its ABI".into())
                        })?;
                        total_archive_bytes =
                            total_archive_bytes.checked_add(length).ok_or_else(|| {
                                ArtifactError::Limit("graph bundle archive lengths overflow".into())
                            })?;
                        if length > limits.max_total_bytes
                            || total_archive_bytes > limits.max_total_bytes
                        {
                            return Err(ArtifactError::Limit(
                                "graph bundle archives exceed their aggregate bound".into(),
                            ));
                        }
                        let bundle = read_bundle(Cursor::new(bytes), limits)?;
                        if bundle.manifest.program_revision != unit.manifest.identity.program {
                            return Err(ArtifactError::Identity(
                                "graph bundle revision differs from selected V3".into(),
                            ));
                        }
                        if bundle.manifest.recipes.len() != 1
                            || bundle.manifest.artifacts.len() != 1
                        {
                            return Err(ArtifactError::Identity(
                                "graph bundle must contain one exact selected V3 recipe and artifact"
                                    .into(),
                            ));
                        }
                        let mut matching = bundle.manifest.artifacts.iter().filter(|artifact| {
                            artifact.form == ExecutableForm::ExecutableUnitV3
                                && artifact.executable_bytes.as_slice() == unit_bytes.as_slice()
                        });
                        let Some(matched) = matching.next() else {
                            return Err(ArtifactError::Identity(
                                "graph bundle has no exact selected V3 artifact".into(),
                            ));
                        };
                        if matching.next().is_some() {
                            return Err(ArtifactError::Identity(
                                "graph bundle does not contain one exact selected V3 artifact"
                                    .into(),
                            ));
                        }
                        let recipe = bundle
                            .manifest
                            .recipes
                            .iter()
                            .find(|recipe| recipe.id().ok() == Some(matched.recipe_id))
                            .ok_or_else(|| {
                                ArtifactError::Identity(
                                    "graph bundle selected V3 recipe is absent".into(),
                                )
                            })?;
                        if recipe.interop != unit.manifest.interop {
                            return Err(ArtifactError::Identity(
                                "graph bundle interop differs from selected V3".into(),
                            ));
                        }
                        policy::validate_archive(self, index, recipe, expected_policy)?;
                        if bundle
                            .manifest
                            .recipes
                            .iter()
                            .any(|recipe| recipe.interop != unit.manifest.interop)
                        {
                            return Err(ArtifactError::Identity(
                                "graph bundle contains foreign artifacts outside its selected V3 interop"
                                    .into(),
                            ));
                        }
                        Ok(Some(bundle))
                    }
                    _ => Err(ArtifactError::Invalid(
                        "graph bundle presence differs from its issued policy ref or selected V3 interop".into(),
                    )),
                }
            })
            .collect()
    }

    /// Scheduler admission for every selected product, while the build recipe
    /// continues to authenticate the primary V3 unit only.
    pub fn execution_host_requirement(
        &self,
        permitted_trust_tiers: BTreeSet<ExecutionTrustTier>,
    ) -> ArtifactResult<ExecutionHostRequirement> {
        let mut requirement = self
            .artifact
            .execution_host_requirement(&self.recipe, permitted_trust_tiers)?;
        for bytes in &self.retained_units {
            let unit = ExecutableUnitEnvelope::from_canonical_bytes(bytes)
                .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
            requirement
                .capabilities
                .0
                .extend(unit.manifest.capabilities.0);
        }
        let mut selected = self.selected_unit_host_requirements()?;
        let primary = selected.remove(0);
        if primary.interop != requirement.interop || primary.environment != requirement.environment
        {
            return Err(ArtifactError::Identity(
                "graph primary placement facts differ from its authenticated recipe".into(),
            ));
        }
        if selected.iter().any(|unit| {
            !unit.interop.adapters.is_empty()
                || !unit.interop.foreign_types.is_empty()
                || !unit.interop.adapter_contracts.is_empty()
        }) {
            requirement
                .capabilities
                .0
                .insert(runmat_types::CapabilityRequirement::ForeignRuntime);
        }
        let requirement = requirement
            .with_selected_unit_requirements(selected)
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        Ok(requirement)
    }

    /// Both client and worker derive this same map from admitted V3 bytes.
    /// Program-local ordinals never enter the graph as unproved sidecar facts.
    pub fn selected_source_map(&self) -> ArtifactResult<ExecutableSourceMapV2> {
        self.validate_selected_unit_bytes()?;
        // The ordered unit position is part of the selected source basis.
        // Two separately issued copies of identical V3 bytes may coexist in
        // one invocation; coalescing by program digest would make a returned
        // exception ordinal ambiguous to the client's exact result receipt.
        let mut selected = Vec::new();
        for bytes in std::iter::once(&self.artifact.executable_bytes).chain(&self.retained_units) {
            let unit = ExecutableUnitEnvelope::from_canonical_bytes(bytes)
                .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
            let component = unit
                .component(ExecutableComponentKind::SourceMap)
                .ok_or_else(|| ArtifactError::Invalid("selected V3 source map is absent".into()))?;
            let map = ExecutableSourceMapV2::decode(&component.bytes)
                .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
            for mut record in map.entries {
                record.source_id =
                    runmat_types::ProgramSourceId(u32::try_from(selected.len()).map_err(|_| {
                        ArtifactError::Limit("graph source ordinals exceed their ABI".into())
                    })?);
                selected.push(record);
            }
        }
        ExecutableSourceMapV2::new(selected)
            .map_err(|error| ArtifactError::Invalid(error.to_string()))
    }

    fn validate_selected_unit_bytes(&self) -> ArtifactResult<()> {
        const MAX_SELECTED_UNITS: usize = runmat_execution::value::GRAPH_MAX_SELECTED_UNITS;
        const MAX_SELECTED_BYTES: usize = runmat_execution::EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES;
        if self.retained_units.len() >= MAX_SELECTED_UNITS {
            return Err(ArtifactError::Limit(
                "graph request has too many selected units".into(),
            ));
        }
        let total = self.retained_units.iter().try_fold(
            self.artifact.executable_bytes.len(),
            |total, unit| {
                total.checked_add(unit.len()).ok_or_else(|| {
                    ArtifactError::Limit("graph selected-unit bytes overflow".into())
                })
            },
        )?;
        if total > MAX_SELECTED_BYTES {
            return Err(ArtifactError::Limit(
                "graph selected-unit bytes exceed their shared bound".into(),
            ));
        }
        Ok(())
    }

    pub fn validate_response(&self, response: &ProgramExecutionResponse) -> ArtifactResult<()> {
        self.validate()?;
        self.validate_admitted_response(response)
    }

    fn validate_admitted_response(
        &self,
        response: &ProgramExecutionResponse,
    ) -> ArtifactResult<()> {
        let source_map = self.selected_source_map()?;
        validate_graph_response_against(
            response,
            &source_map,
            self.retained_units.len() + 1,
            self.requested_outputs,
            GraphRequiredSchema::from_version(self.schema_version)
                .ok_or_else(|| ArtifactError::Invalid("unsupported graph schema".into()))?,
        )
    }
}

fn validate_graph_response_against(
    response: &ProgramExecutionResponse,
    source_map: &ExecutableSourceMapV2,
    selected_unit_count: usize,
    requested_outputs: u16,
    required_schema: GraphRequiredSchema,
) -> ArtifactResult<()> {
    match response {
        ProgramExecutionResponse::GraphSuccess { graph }
            if graph.roots.len() == usize::from(requested_outputs)
                && graph.nodes.len() as u64 <= ValueLimits::default().max_nodes =>
        {
            if graph.required_schema() > required_schema {
                return Err(ArtifactError::Invalid(
                    "graph response requires a newer selected-value schema".into(),
                ));
            }
            graph
                .validate_selected_unit_references(selected_unit_count)
                .map_err(super::contract_error)?;
            // Selected VM layout/source proof is required before decoding;
            // this transport check cannot promote wire-claimed bindings.
            for node in &graph.nodes {
                if let GraphNode::Leaf { value } = node {
                    value
                        .validate_for_graph_sources(ValueLimits::default(), Some(source_map))
                        .map_err(super::contract_error)?;
                }
            }
            Ok(())
        }
        ProgramExecutionResponse::GraphSuccess { .. } => Err(ArtifactError::Invalid(
            "graph response differs from its output contract".into(),
        )),
        ProgramExecutionResponse::Failure { message }
            if !message.is_empty() && message.len() <= 1024 * 1024 =>
        {
            Ok(())
        }
        ProgramExecutionResponse::RuntimeFailure { failure } => failure
            .validate_for_program_sources(Some(source_map))
            .map_err(super::contract_error),
        _ => Err(ArtifactError::Invalid(
            "graph request received an incompatible response".into(),
        )),
    }
}
