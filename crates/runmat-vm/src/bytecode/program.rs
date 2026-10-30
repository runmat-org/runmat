#[cfg(feature = "native-accel")]
use crate::accel::graph::build_accel_graph;
#[cfg(feature = "native-accel")]
use crate::accel::stack_layout::annotate_fusion_groups_with_stack_layout;
use crate::bytecode::instr::Instr;
use crate::layout::VmAssemblyLayout;
#[cfg(feature = "native-accel")]
use runmat_accelerate::graph::AccelGraph;
#[cfg(feature = "native-accel")]
use runmat_accelerate::FusionGroup;
use runmat_builtins::Type;
use runmat_hir::FunctionId;
use runmat_types::{FunctionArgDefaultValue, FunctionArgSizeSpec, FunctionArgValidator};
use runmat_value::Value;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use super::{
    parallel::{BytecodeParforRegion, BytecodeSpmdRegion},
    region::BytecodeRegion,
    source_ref::BytecodeSourceRef,
    statement_pc::{project_statement_pcs, StatementPcProjectionError, StatementPcRange},
};

#[path = "program/constructor_handles.rs"]
mod constructor_handles;
#[path = "program/definition_dependencies.rs"]
mod definition_dependencies;
#[path = "program/owner_index.rs"]
mod owner_index;
#[path = "program/selected_class_product.rs"]
mod selected_class_product;
#[path = "program/selected_executable_product.rs"]
mod selected_executable_product;
pub use selected_executable_product::{
    SelectedExecutableAttachmentError, SelectedExecutableCode, SelectedExecutableProduct,
    SelectedLinkedNativeSite, SelectedStaticCapture, SelectedStaticProductDependency,
    SelectedStaticProductTarget, SelectedStaticSiteOwner,
};
#[path = "program/selected_registry.rs"]
mod selected_registry;
#[path = "program/super_constructor_dependencies.rs"]
mod super_constructor_dependencies;
pub(crate) use super_constructor_dependencies::{instruction_classes, same_class_registration};
pub use super_constructor_dependencies::{
    RetainedImplicitSuperConstructorTarget, SuperConstructorDependencyError,
};
#[cfg(feature = "native-accel")]
#[path = "program/function_fusion.rs"]
mod function_fusion;
#[path = "program/function_remap.rs"]
pub(crate) mod function_remap;
#[path = "program/private_resolution.rs"]
mod private_resolution;
#[cfg(feature = "native-accel")]
pub use function_fusion::FunctionFusionScaffold;
#[cfg(feature = "native-accel")]
pub(crate) use function_fusion::{
    selected_function_windows, FunctionFusionCertificateIndex, FunctionPrivateSlotLayout,
};
pub use private_resolution::PublishedNameResolutionView;
#[cfg(feature = "native-accel")]
#[path = "program/vm_fusion_plan.rs"]
mod vm_fusion_plan;
#[cfg(feature = "native-accel")]
#[path = "program/vm_image_plan.rs"]
mod vm_image_plan;
pub use function_remap::{
    instruction_reference_sites, remap_reference_sites, FunctionReferenceKind,
};
pub use function_remap::{
    rebind_session_source, remap_closed_session_function, remap_decorated_session_definitions,
    remap_session_entrypoint, remap_session_function, SessionCallableBinding,
};
#[cfg(feature = "native-accel")]
pub(crate) use vm_fusion_plan::VmFusionPlanInstallation;
#[cfg(feature = "native-accel")]
pub(crate) use vm_image_plan::SealedFunctionImage;
#[cfg(feature = "native-accel")]
pub(crate) use vm_image_plan::{VmImagePlanInstallation, VmImageSelected};
#[path = "program/bound_binding.rs"]
mod bound_binding;
#[path = "program/retained_artifacts.rs"]
pub(crate) mod retained_artifacts;
#[path = "program/semantic_fingerprint.rs"]
mod semantic_fingerprint;
#[path = "program/slot_set_serde.rs"]
mod slot_set_serde;
#[path = "program/source_facts.rs"]
mod source_facts;
#[path = "program/source_projection.rs"]
mod source_projection;
pub use source_projection::FunctionSourceProjectionError;

#[cfg(feature = "native-accel")]
pub(crate) use semantic_fingerprint::fusion_stage_source_fingerprint;
pub(crate) use semantic_fingerprint::local_bytecode_semantic_fingerprint;
pub use semantic_fingerprint::semantic_definition_fingerprint;

#[derive(Debug, Clone)]
pub struct CallFrame {
    pub function_name: String,
    pub return_address: usize,
    pub locals_start: usize,
    pub locals_count: usize,
    pub expected_outputs: usize,
}

#[derive(Debug)]
pub struct ExecutionContext {
    pub call_stack: Vec<CallFrame>,
    pub locals: Vec<Value>,
    pub instruction_pointer: usize,
    pub runtime: runmat_runtime::context::RuntimeContext,
    /// Explicit activation selected by this frame, never inferred from a name
    /// or the ambient runtime context.
    pub lexical_environment: Option<runmat_value::LexicalEnvironment>,
    pub lexical_binding_slots: Vec<LexicalBindingSlot>,
    pub inherited_storage_slots: Vec<InheritedStorageSlot>,
    /// Exact worker product selected at the owning invocation. Bare VM
    /// execution cannot infer it from ambient or latest-session state.
    pub(crate) graph_program: Option<runmat_runtime::execution::DeferredGraphProgram>,
    pub(crate) process_driver: Option<crate::interpreter::process_driver::ProcessDriverHandle>,
    /// The exact active compiler product, retained for queued child VM work.
    pub(crate) bytecode_owner: Option<crate::interpreter::bytecode_owner::BytecodeLease>,
    /// Only an explicitly admitted Core invocation may install this port.
    pub(crate) dynamic_eval_admission:
        Option<crate::interpreter::dynamic_admission::DynamicEvalAdmissionPort>,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            call_stack: Vec::new(),
            locals: Vec::new(),
            instruction_pointer: 0,
            runtime: runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            )),
            lexical_environment: None,
            lexical_binding_slots: Vec::new(),
            inherited_storage_slots: Vec::new(),
            graph_program: None,
            process_driver: None,
            bytecode_owner: None,
            dynamic_eval_admission: None,
        }
    }
}

fn fresh_definition_owner() -> std::sync::Arc<runmat_value::FunctionDefinitionOwner> {
    std::sync::Arc::new(runmat_value::FunctionDefinitionOwner::default())
}

/// Path-free logical directory name in the authenticated source map.
#[derive(Debug, Clone, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct PrivateSourceAlias {
    pub owner_identity: String,
    pub relative_directory: String,
}

impl PrivateSourceAlias {
    pub fn from_source_path(owner_identity: &str, relative_path: &str) -> Self {
        let normalized = relative_path.replace('\\', "/");
        let source_directory = std::path::Path::new(&normalized)
            .parent()
            .unwrap_or(std::path::Path::new(""));
        let directory = if source_directory
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case("private"))
        {
            source_directory
                .parent()
                .unwrap_or(std::path::Path::new(""))
        } else {
            source_directory
        };
        Self {
            owner_identity: owner_identity.to_owned(),
            relative_directory: directory.to_string_lossy().into_owned(),
        }
    }
}

/// Private-source directory equivalence captured by the publishing owner.
/// Portable aliases record the source graph's equivalence partition without
/// serializing the native filesystem location.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct PrivateSourceDirectory {
    aliases: Vec<PrivateSourceAlias>,
    #[serde(skip)]
    host_path: Option<std::path::PathBuf>,
}

impl PrivateSourceDirectory {
    pub fn new(host_path: std::path::PathBuf) -> Self {
        Self {
            aliases: Vec::new(),
            host_path: Some(host_path),
        }
    }

    pub fn captured(
        host_path: Option<std::path::PathBuf>,
        mut aliases: Vec<PrivateSourceAlias>,
    ) -> Self {
        aliases.sort();
        aliases.dedup();
        Self { aliases, host_path }
    }

    pub fn aliases(&self) -> &[PrivateSourceAlias] {
        &self.aliases
    }

    pub fn is_known(&self) -> bool {
        self.host_path.is_some() || !self.aliases.is_empty()
    }

    pub fn same_owner(&self, other: &Self) -> bool {
        match (&self.host_path, &other.host_path) {
            (Some(left), Some(right)) => left == right,
            _ => self
                .aliases
                .iter()
                .any(|alias| other.aliases.contains(alias)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LexicalBindingKind {
    Owned,
    Inherited,
    /// Carry the activation to nested descendants while preserving this
    /// anonymous frame's own captured value as a snapshot.
    Transit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalBindingSlot {
    pub binding: runmat_types::BindingId,
    pub slot: usize,
    pub kind: LexicalBindingKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InheritedStorageKind {
    Global,
    Persistent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InheritedStorageSlot {
    pub binding: runmat_types::BindingId,
    pub slot: usize,
    /// Selected function identity and its compiler ordinal, both required
    /// before an imported child can delegate to ancestor session storage.
    pub owner: FunctionId,
    pub portable_owner: runmat_types::ProgramFunctionId,
    pub name: String,
    pub owner_function: String,
    pub kind: InheritedStorageKind,
}

#[path = "program/future_spawn.rs"]
mod future_spawn;
pub use future_spawn::CompilerSpawnContract;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionBytecode {
    /// Compiler-owned metadata travels with this body, like its capabilities.
    /// Portable admission renews it from accepted MIR, never serialized facts.
    #[serde(skip)]
    pub spawn_contract: Option<CompilerSpawnContract>,
    pub function: FunctionId,
    /// Process-local identity of this immutable definition. Serialized
    /// artifacts hydrate a fresh owner; they never import a local lease.
    #[serde(skip, default = "fresh_definition_owner")]
    pub definition_owner: std::sync::Arc<runmat_value::FunctionDefinitionOwner>,
    pub display_name: String,
    pub class_method_owner: Option<runmat_types::ClassMethodOwner>,
    #[serde(default)]
    pub private_owner_scope: String,
    #[serde(default)]
    pub source_id: Option<BytecodeSourceRef>,
    /// Exact issued source leases retained with this definition after its
    /// publishing invocation ends. This local owner never enters wire code.
    #[serde(skip)]
    pub selected_source_owner: Option<runmat_runtime::source_context::SelectedProgramSourceOwner>,
    /// Checked unit-owned foreign archive; never serialized as bytecode.
    #[serde(skip)]
    pub selected_interop_receipt: Option<std::rc::Rc<crate::SelectedUnitInteropReceipt>>,
    /// Invocation-local selected class ancestry retained with this exact
    /// definition owner, never serialized as source or registry authority.
    #[serde(skip)]
    pub selected_class_product: Option<std::rc::Rc<crate::SelectedClassProduct>>,
    /// Exact code/source product retained by this definition. Its optional
    /// local admission port is serde-skipped and has no strong body or registry backedge.
    #[serde(skip)]
    pub selected_executable_product: Option<std::rc::Rc<SelectedExecutableProduct>>,
    /// Immutable site-qualified import closure captured by this exact body.
    #[serde(skip)]
    pub selected_static_capture: Option<std::rc::Rc<SelectedStaticCapture>>,
    /// Original display file attached by the publishing source-map owner.
    /// Retired callable values never reconstruct it from a reused SourceId.
    #[serde(default)]
    pub source_file: Option<String>,
    /// Source owner's captured directory identity for private-name admission.
    /// Core captures native symlink equivalence and portable aliases at
    /// publication; observers do not consult the filesystem.
    #[serde(default)]
    pub private_source_owner: Option<PrivateSourceDirectory>,
    #[serde(default)]
    pub source_role: runmat_types::FunctionSourceRole,
    /// Session identity of the containing function, remapped with the body.
    #[serde(default)]
    pub lexical_parent: Option<FunctionId>,
    /// Compiler ordinal before Core assigns a session-local function ID.
    pub portable_function: runmat_types::ProgramFunctionId,
    /// Stable compiler ordinal of the containing function. Definition
    /// fingerprints use this rather than the session-local parent ID.
    #[serde(default)]
    pub portable_lexical_parent: Option<runmat_types::ProgramFunctionId>,
    /// Immutable origin set by the publishing owner before session-ID remap.
    /// None is a legacy or unpublished function, never portable binding proof.
    pub binding_origin: Option<runmat_execution::FunctionBindingOrigin>,
    /// Whole-function semantic requirements, including reachable static
    /// callees, captured by MIR analysis before bytecode serialization.
    #[serde(default)]
    pub capabilities: runmat_types::CapabilitySet,
    #[cfg(feature = "native-accel")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fusion_stage_projection: Option<std::sync::Arc<crate::bytecode::FusionStageProjection>>,
    #[cfg(feature = "native-accel")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fusion_scaffold: Option<FunctionFusionScaffold>,
    pub instructions: Vec<Instr>,
    /// Innermost lexical catch at each instruction, derived from MIR.
    pub catch_scopes: Vec<Option<runmat_mir::CatchScopeId>>,
    #[serde(default)]
    pub instr_spans: Vec<runmat_hir::Span>,
    #[serde(default)]
    pub call_arg_spans: Vec<Option<Vec<runmat_hir::Span>>>,
    #[serde(default)]
    pub coverage_sites: Vec<Vec<u64>>,
    pub var_count: usize,
    pub input_slots: Vec<usize>,
    #[serde(default)]
    pub varargin_slot: Option<usize>,
    #[serde(default)]
    pub implicit_nargin_slot: Option<usize>,
    pub output_slots: Vec<usize>,
    #[serde(default)]
    pub varargout_slot: Option<usize>,
    #[serde(default)]
    pub implicit_nargout_slot: Option<usize>,
    pub capture_slots: Vec<usize>,
    /// Only bindings shared with a nested descendant or inherited from an
    /// ancestor. Child-local bindings remain private to each invocation.
    #[serde(default)]
    pub lexical_binding_slots: Vec<LexicalBindingSlot>,
    /// Nested access to an ancestor's canonical global/persistent storage.
    #[serde(default)]
    pub inherited_storage_slots: Vec<InheritedStorageSlot>,
    #[serde(default)]
    pub creates_lexical_environment: bool,
    #[serde(default)]
    pub var_names: HashMap<usize, String>,
    #[serde(default, with = "slot_set_serde")]
    pub initially_unassigned_slots: HashSet<usize>,
    #[serde(default)]
    pub argument_validations: Vec<FunctionArgumentValidation>,
    #[serde(default, with = "crate::layout::resume_point_map_serde")]
    pub resume_points: std::collections::BTreeMap<runmat_types::ProgramPointId, usize>,
    #[serde(default)]
    pub regions: Vec<BytecodeRegion>,
    #[serde(default)]
    pub parfor_regions: Vec<BytecodeParforRegion>,
    #[serde(default)]
    pub spmd_regions: Vec<BytecodeSpmdRegion>,
    #[serde(default)]
    pub distributed_values: Vec<runmat_types::DistributedValueContract>,
    #[serde(default)]
    pub collective_contracts: Vec<runmat_types::CollectiveContract>,
}

impl Default for FunctionBytecode {
    fn default() -> Self {
        Self {
            spawn_contract: None,
            function: FunctionId(0),
            definition_owner: fresh_definition_owner(),
            display_name: String::new(),
            class_method_owner: None,
            private_owner_scope: String::new(),
            source_id: None,
            selected_source_owner: None,
            selected_interop_receipt: None,
            selected_class_product: None,
            selected_executable_product: None,
            selected_static_capture: None,
            source_file: None,
            private_source_owner: None,
            source_role: runmat_types::FunctionSourceRole::Unknown,
            lexical_parent: None,
            portable_function: runmat_types::ProgramFunctionId(0),
            portable_lexical_parent: None,
            binding_origin: None,
            capabilities: Default::default(),
            #[cfg(feature = "native-accel")]
            fusion_stage_projection: None,
            #[cfg(feature = "native-accel")]
            fusion_scaffold: None,
            instructions: Vec::new(),
            catch_scopes: Vec::new(),
            instr_spans: Vec::new(),
            call_arg_spans: Vec::new(),
            coverage_sites: Vec::new(),
            var_count: 0,
            input_slots: Vec::new(),
            varargin_slot: None,
            implicit_nargin_slot: None,
            output_slots: Vec::new(),
            varargout_slot: None,
            implicit_nargout_slot: None,
            capture_slots: Vec::new(),
            lexical_binding_slots: Vec::new(),
            inherited_storage_slots: Vec::new(),
            creates_lexical_environment: false,
            var_names: HashMap::new(),
            initially_unassigned_slots: HashSet::new(),
            argument_validations: Vec::new(),
            resume_points: std::collections::BTreeMap::new(),
            regions: Vec::new(),
            parfor_regions: Vec::new(),
            spmd_regions: Vec::new(),
            distributed_values: Vec::new(),
            collective_contracts: Vec::new(),
        }
    }
}

impl FunctionBytecode {
    /// Copied captures may be omitted only when every slot has a compiled
    /// alternative owner. An activation cannot substitute for a by-value
    /// snapshot (including anonymous transit captures).
    pub(crate) fn captures_owned_without_copies(&self, has_activation: bool) -> bool {
        self.capture_slots.iter().all(|slot| {
            self.inherited_storage_slots
                .iter()
                .any(|entry| entry.slot == *slot)
                || (has_activation
                    && self.lexical_binding_slots.iter().any(|entry| {
                        entry.slot == *slot && entry.kind == LexicalBindingKind::Inherited
                    }))
        })
    }

    /// Preserve the point owner's own identity domain: closed-unit and
    /// session rebinding may intentionally assign different FunctionId and
    /// ProgramFunctionId values. The later MIR bridge must match its selected
    /// body against `StatementPcRange::statement.function` explicitly.
    pub fn statement_pc_ranges(&self) -> Result<Vec<StatementPcRange>, StatementPcProjectionError> {
        project_statement_pcs(&self.resume_points, self.instructions.len())
    }

    pub(crate) fn validate_catch_scope_metadata(&self) -> Result<(), String> {
        validate_catch_scope_metadata(&self.instructions, &self.catch_scopes)
    }

    /// Materializes the function-local executable view consumed by the VM.
    /// All compiler-owned metadata travels with the instruction stream here so
    /// function dispatch cannot accidentally discard executable contracts.
    pub(crate) fn execution_bytecode(&self, registry: &FunctionRegistry) -> Bytecode {
        let mut bytecode = Bytecode::with_instructions(self.instructions.clone(), self.var_count);
        bytecode.catch_scopes = self.catch_scopes.clone();
        bytecode.instr_spans = self.instr_spans.clone();
        bytecode.call_arg_spans = self.call_arg_spans.clone();
        bytecode.coverage_sites = self.coverage_sites.clone();
        bytecode.resume_points = self.resume_points.clone();
        #[cfg(feature = "native-accel")]
        {
            bytecode.fusion_metadata.stage_projection = self.fusion_stage_projection.clone();
            if let Some(scaffold) = registry.admitted_fusion_scaffold(self) {
                bytecode.function_private_layout = scaffold.private_layout.clone();
                bytecode.fusion_metadata.mir_fusion_signal_count = scaffold.signal_count;
                bytecode.fusion_metadata.mir_fusion_candidate_group_count =
                    scaffold.candidates.len();
                bytecode.fusion_metadata.mir_fusion_candidate_groups = scaffold.candidates.clone();
                bytecode.fusion_metadata.instruction_window_count = scaffold.windows.len();
                bytecode.fusion_metadata.instruction_windows = scaffold.windows.clone();
            }
        }
        bytecode.source_id = self.source_id;
        bytecode.active_function = Some(self.function);
        bytecode.active_class_method_owner = self.class_method_owner.clone();
        bytecode.var_names = self.var_names.clone();
        bytecode.initially_unassigned_slots = self.initially_unassigned_slots.clone();
        bytecode.regions = self.regions.clone();
        bytecode.parfor_regions = self.parfor_regions.clone();
        bytecode.spmd_regions = self.spmd_regions.clone();
        bytecode.distributed_values = self.distributed_values.clone();
        bytecode.collective_contracts = self.collective_contracts.clone();
        // The authoritative registry below is already a shared immutable
        // execution snapshot. Duplicating every body into the legacy fallback
        // map on each call makes invocation scale with the whole program.
        bytecode.function_registry = registry.clone();
        #[cfg(feature = "native-accel")]
        if registry
            .get(self.function)
            .is_some_and(|selected| std::ptr::eq(selected, self))
            && registry.static_overlay.is_none()
            && (registry
                .data
                .selected_image_views
                .borrow()
                .contains_key(&self.function)
                || self.validate_fusion_stage_projection().is_ok())
        {
            if let Some(layout) = self
                .fusion_stage_projection
                .as_ref()
                .and_then(|projection| projection.selected_function_layout.as_ref())
                .filter(|layout| {
                    layout.function == self.function
                        && layout.resume_points == self.resume_points
                        && layout.local_count == self.var_count
                })
            {
                bytecode.layout = Some(VmAssemblyLayout {
                    functions: HashMap::from([(self.function, layout.clone())]),
                    entrypoints: HashMap::new(),
                    storage_bindings: HashMap::new(),
                });
            }
        }
        #[cfg(feature = "native-accel")]
        {
            bytecode.sealed_function_image = registry.finalize_selected_image_for(self, &bytecode);
        }
        bytecode
    }

    /// Rebinds every compiler-owned identity when a unit-local function is
    /// installed into a session registry. Control-flow and parallel metadata
    /// must move atomically with the function instruction stream.
    pub fn rebind_identity(&mut self, function: FunctionId) {
        self.rebind_identity_inner(function, true);
    }

    pub(crate) fn rebind_identity_unsealed(&mut self, function: FunctionId) {
        self.rebind_identity_inner(function, false);
    }

    fn rebind_identity_inner(&mut self, function: FunctionId, reseal_stage: bool) {
        let program_function = self.portable_function;
        #[cfg(not(feature = "native-accel"))]
        let _ = reseal_stage;
        #[cfg(feature = "native-accel")]
        let prior_stage_valid = self.fusion_stage_projection.is_some()
            && self.validate_fusion_stage_projection().is_ok();
        self.function = function;
        #[cfg(feature = "native-accel")]
        if let Some(scaffold) = &mut self.fusion_scaffold {
            for candidate in &mut scaffold.candidates {
                candidate.function = function;
            }
            if let Some(layout) = &mut scaffold.private_layout {
                layout.rebind(function, program_function);
            }
        }
        self.resume_points = std::mem::take(&mut self.resume_points)
            .into_iter()
            .map(|(mut point, pc)| {
                point.function = program_function;
                (point, pc)
            })
            .collect();
        for region in &mut self.regions {
            region.rebind_owner(program_function);
        }
        for region in &mut self.parfor_regions {
            region.rebind_owner(program_function);
        }
        for region in &mut self.spmd_regions {
            region.rebind_owner(program_function);
        }
        for distributed in &mut self.distributed_values {
            distributed.id.function = program_function;
            match &mut distributed.owner {
                runmat_types::DistributedOwner::Client(function) => *function = program_function,
                runmat_types::DistributedOwner::Region(region) => {
                    region.0.function = program_function
                }
            }
        }
        for collective in &mut self.collective_contracts {
            collective.id.region.0.function = program_function;
        }
        for instruction in &mut self.instructions {
            match instruction {
                Instr::ExecuteParfor { region, .. } | Instr::ExecuteSpmd { region, .. } => {
                    region.0.function = program_function;
                }
                _ => {}
            }
        }
        #[cfg(feature = "native-accel")]
        {
            if prior_stage_valid {
                let valid_after_rebind =
                    self.fusion_stage_projection
                        .as_mut()
                        .is_some_and(|projection| {
                            let projection = std::sync::Arc::make_mut(projection);
                            if let Some(layout) = &mut projection.selected_function_layout {
                                layout.function = function;
                                layout.resume_points = self.resume_points.clone();
                                layout.parent = self.lexical_parent;
                            }
                            if reseal_stage {
                                projection.rebind_owner(
                                    program_function,
                                    crate::bytecode::fusion_stage::FusionStageStream {
                                        instructions: &self.instructions,
                                        catch_scopes: &self.catch_scopes,
                                        resume_points: &self.resume_points,
                                        source_id: self.source_id,
                                        var_count: self.var_count,
                                    },
                                )
                            } else {
                                projection.rebind_owner_unsealed(program_function);
                                true
                            }
                        });
                if !valid_after_rebind {
                    self.fusion_stage_projection = None;
                }
            } else {
                self.fusion_stage_projection = None;
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionArgumentValidation {
    pub input_slot: usize,
    pub size: Option<FunctionArgSizeSpec>,
    pub class_name: Option<String>,
    #[serde(default)]
    pub validators: Vec<FunctionArgValidator>,
    #[serde(default)]
    pub default_value: Option<FunctionArgDefaultValue>,
}

/// Cheap immutable execution snapshots; authoring mutations copy on write.
/// The serialized representation remains the prior registry record, not an
/// Rc or a process-local owner token.
#[derive(Debug, Clone, Default)]
pub struct FunctionRegistry {
    data: std::rc::Rc<FunctionRegistryData>,
    static_overlay: Option<std::rc::Rc<StaticDefinitionOverlay>>,
}

/// Exact bodies reachable through a retained bound value. Dynamic name calls
/// use the published registry, never these old definitions.
#[derive(Debug, Clone)]
struct StaticDefinitionOverlay {
    authority: runmat_value::BoundFunctionAuthority,
    execution_stacks: OnceLock<HashMap<FunctionId, runmat_types::ExecutionStackRequirement>>,
}

/// A source snapshot is code storage, not definition authority. The counted
/// root lease lives on an escaped value; this raw owner only permits pruning
/// snapshots after that lease has retired.
#[derive(Debug, Clone)]
struct SelectedSiteView {
    registry: FunctionRegistry,
    root_owner: std::sync::Arc<runmat_value::FunctionDefinitionOwner>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunctionRegistryData {
    pub functions: HashMap<FunctionId, FunctionBytecode>,
    #[serde(default)]
    pub names: HashMap<String, FunctionId>,
    #[serde(skip)]
    pub source_functions: HashMap<BytecodeSourceRef, Vec<FunctionId>>,
    /// Session-only old bodies that have live callable leases. They are never
    /// name-resolvable or serialized into a portable executable artifact.
    #[serde(skip)]
    retired: HashMap<FunctionId, Vec<FunctionBytecode>>,
    /// Unnamed immutable source views for site-qualified selected targets.
    /// Each view shares its code storage; only an exact owner lease may open a
    /// body from it, and no source/name lookup consults this collection.
    #[serde(skip)]
    selected_site_views: Vec<SelectedSiteView>,
    /// Rebuilt from current and retired bodies after unrestricted mutation;
    /// never serialized as an independent definition authority.
    #[serde(skip)]
    owner_index: std::cell::RefCell<Option<std::sync::Arc<owner_index::OwnerIndex>>>,
    #[serde(skip)]
    execution_stacks: OnceLock<HashMap<FunctionId, runmat_types::ExecutionStackRequirement>>,
    #[cfg(feature = "native-accel")]
    #[serde(skip)]
    admitted_fusion_scaffolds: OnceLock<HashMap<usize, std::sync::Arc<FunctionFusionScaffold>>>,
    /// Derived immutable image views belong to this exact copy-on-write
    /// registry generation, not to a session-wide function ordinal.
    #[cfg(feature = "native-accel")]
    #[serde(skip)]
    selected_image_views: std::cell::RefCell<vm_image_plan::SelectedImageViews>,
    #[cfg(feature = "native-accel")]
    #[serde(skip)]
    selected_image_generation: u64,
}

#[cfg(feature = "native-accel")]
impl FunctionRegistryData {
    fn invalidate_selected_image_views(&mut self) {
        self.selected_image_views.get_mut().clear();
        self.selected_image_generation = self
            .selected_image_generation
            .checked_add(1)
            .expect("selected image registry generation exhausted");
    }
}

impl std::ops::Deref for FunctionRegistry {
    type Target = FunctionRegistryData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::DerefMut for FunctionRegistry {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.static_overlay = None;
        let data = std::rc::Rc::make_mut(&mut self.data);
        data.owner_index.get_mut().take();
        data.execution_stacks.take();
        #[cfg(feature = "native-accel")]
        data.admitted_fusion_scaffolds.take();
        #[cfg(feature = "native-accel")]
        data.invalidate_selected_image_views();
        data
    }
}

impl Serialize for FunctionRegistry {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.data.as_ref().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FunctionRegistry {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        FunctionRegistryData::deserialize(deserializer).map(|mut data| {
            data.source_functions = source_index(&data.functions);
            let functions = &data.functions;
            data.names.retain(|_, function| {
                functions.get(function).is_some_and(|body| {
                    body.source_role != runmat_types::FunctionSourceRole::Nested
                })
            });
            Self {
                data: std::rc::Rc::new(data),
                static_overlay: None,
            }
        })
    }
}

fn source_index(
    functions: &HashMap<FunctionId, FunctionBytecode>,
) -> HashMap<BytecodeSourceRef, Vec<FunctionId>> {
    let mut index: HashMap<BytecodeSourceRef, Vec<FunctionId>> = HashMap::new();
    let mut ids = functions.keys().copied().collect::<Vec<_>>();
    ids.sort_by_key(|id| id.0);
    for id in ids {
        if let Some(source) = functions[&id].source_id {
            index.entry(source).or_default().push(id);
        }
    }
    index
}

impl FunctionRegistry {
    #[cfg(not(feature = "native-accel"))]
    pub fn finalize_selected_image_views(&self) {}

    /// Exact immutable execution-view identity, including any site-qualified
    /// static overlay. Derived lookup caches are intentionally excluded.
    pub fn same_immutable_execution_view(&self, other: &Self) -> bool {
        std::rc::Rc::ptr_eq(&self.data, &other.data)
            && match (&self.static_overlay, &other.static_overlay) {
                (None, None) => true,
                (Some(left), Some(right)) => left.authority.shares_local_graph(&right.authority),
                _ => false,
            }
    }

    pub fn selected_runtime_origin_fingerprints(
        &self,
    ) -> Result<
        std::collections::HashMap<runmat_execution::FunctionBindingOrigin, String>,
        retained_artifacts::RetainedCallableStageError,
    > {
        retained_artifacts::selected_runtime_origin_fingerprints(self)
    }

    pub fn selected_runtime_body_fingerprint(
        &self,
        function: FunctionId,
    ) -> Result<
        (runmat_execution::FunctionBindingOrigin, String),
        retained_artifacts::RetainedCallableStageError,
    > {
        retained_artifacts::selected_runtime_body_fingerprint(self, function)
    }

    pub fn selected_runtime_body_fingerprints(
        &self,
        functions: &[FunctionId],
    ) -> Result<
        std::collections::HashMap<FunctionId, (runmat_execution::FunctionBindingOrigin, String)>,
        retained_artifacts::RetainedCallableStageError,
    > {
        retained_artifacts::selected_runtime_body_fingerprints(self, functions)
    }

    pub fn selected_runtime_body_fingerprints_with_imports(
        &self,
        functions: &[FunctionId],
        imports: &[runmat_execution::executable::ExecutableStaticImport],
    ) -> Result<
        std::collections::HashMap<FunctionId, (runmat_execution::FunctionBindingOrigin, String)>,
        retained_artifacts::RetainedCallableStageError,
    > {
        retained_artifacts::selected_runtime_body_fingerprints_with_imports(
            self, functions, imports,
        )
    }

    pub fn selected_runtime_body_fingerprints_by_product(
        &self,
        inputs: &[(
            FunctionId,
            &[runmat_execution::executable::ExecutableStaticImport],
        )],
    ) -> Result<
        std::collections::HashMap<FunctionId, (runmat_execution::FunctionBindingOrigin, String)>,
        retained_artifacts::RetainedCallableStageError,
    > {
        retained_artifacts::selected_runtime_body_fingerprints_by_product(self, inputs)
    }

    /// Retain one already-issued source table on selected, immutable bodies.
    /// Validate every member before changing the registry so a mismatched
    /// product cannot leave a partially source-owned publication behind.
    pub fn attach_selected_program_source_owner(
        &mut self,
        ids: impl IntoIterator<Item = FunctionId>,
        program: runmat_execution::Digest,
        owner: &runmat_runtime::source_context::SelectedProgramSourceOwner,
    ) -> Result<(), &'static str> {
        let ids = ids.into_iter().collect::<Vec<_>>();
        for id in &ids {
            let body = self
                .functions
                .get(id)
                .ok_or("selected source owner has no matching function body")?;
            if body.binding_origin.map(|origin| origin.program) != Some(program) {
                return Err("selected source owner differs from the function product");
            }
            let Some(BytecodeSourceRef::Runtime(source)) = body.source_id else {
                return Err("selected source owner requires an issued runtime source");
            };
            if !owner.owns_source(source) {
                return Err("selected source owner differs from the function source lease");
            }
            if body
                .selected_source_owner
                .as_ref()
                .is_some_and(|previous| !previous.same_issuance(owner))
            {
                return Err("selected source owner differs from the previous issuance");
            }
        }
        for id in ids {
            self.functions
                .get_mut(&id)
                .expect("selected source owner was validated above")
                .selected_source_owner = Some(owner.clone());
        }
        Ok(())
    }

    pub fn validate_selected_source_owners(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<(), &'static str> {
        if self.functions.values().any(|body| {
            matches!(body.source_id, Some(BytecodeSourceRef::Runtime(reference)) if reference.resolve_in(runtime).is_none())
        }) {
            return Err("function registry carries a source issued by another runtime owner");
        }
        Ok(())
    }

    pub fn validate_portable_source_origins(&self) -> Result<(), &'static str> {
        if self.functions.values().any(|body| {
            body.source_id
                .is_some_and(|source| !matches!(source, BytecodeSourceRef::Program(_)))
        }) {
            return Err("portable function registry contains a session source identity");
        }
        Ok(())
    }

    /// Consume the authoring registry at publication boundaries. Executing
    /// snapshots retain their own immutable Rc; only shared publication
    /// inputs require a copy of the map here.
    pub fn into_functions(self) -> HashMap<FunctionId, FunctionBytecode> {
        std::rc::Rc::try_unwrap(self.data)
            .unwrap_or_else(|data| (*data).clone())
            .functions
    }

    pub fn new(functions: HashMap<FunctionId, FunctionBytecode>) -> Self {
        let mut names = HashMap::new();
        let mut source_functions: HashMap<BytecodeSourceRef, Vec<FunctionId>> = HashMap::new();
        let mut ids: Vec<_> = functions.keys().copied().collect();
        ids.sort_by_key(|id| id.0);
        for id in ids {
            if let Some(function) = functions.get(&id) {
                if function.source_role != runmat_types::FunctionSourceRole::Nested {
                    names.entry(function.display_name.clone()).or_insert(id);
                }
                if let Some(source_id) = function.source_id {
                    source_functions.entry(source_id).or_default().push(id);
                }
            }
        }
        Self {
            data: std::rc::Rc::new(FunctionRegistryData {
                functions,
                names,
                source_functions,
                retired: HashMap::new(),
                selected_site_views: Vec::new(),
                owner_index: std::cell::RefCell::new(None),
                execution_stacks: OnceLock::new(),
                #[cfg(feature = "native-accel")]
                admitted_fusion_scaffolds: OnceLock::new(),
                #[cfg(feature = "native-accel")]
                selected_image_views: std::cell::RefCell::new(HashMap::new()),
                #[cfg(feature = "native-accel")]
                selected_image_generation: 0,
            }),
            static_overlay: None,
        }
    }

    pub fn get(&self, function: FunctionId) -> Option<&FunctionBytecode> {
        if let Some(overlay) = &self.static_overlay {
            let lease = overlay.authority.local_lease(function.0)?;
            return self.get_owned(function, lease);
        }
        self.functions.get(&function)
    }

    /// The currently published body, regardless of a retained static call
    /// graph. Name-based and workspace calls must use this lookup.
    pub fn get_current(&self, function: FunctionId) -> Option<&FunctionBytecode> {
        self.functions.get(&function)
    }

    pub fn current_view(&self) -> Self {
        Self {
            data: std::rc::Rc::clone(&self.data),
            static_overlay: None,
        }
    }

    /// A compiler-bound static edge in a retained body selects its exact
    /// leased callee. Ordinary current static calls remain numeric because
    /// their current published bytecode already selects the current body.
    pub fn static_binding(
        &self,
        function: FunctionId,
    ) -> Option<runmat_value::BoundFunctionBinding> {
        let overlay = self.static_overlay.as_ref()?;
        if overlay.authority.local_lease(function.0).is_none() {
            // A stale or malformed static edge in an old body must not fall
            // through to the new numeric body at that same ID.
            return Some(runmat_value::BoundFunctionBinding::unavailable(function.0));
        }
        Some(runmat_value::BoundFunctionBinding {
            function: function.0,
            authority: overlay.authority.clone(),
        })
    }

    /// A bound value carries the exact definition lease. Numeric IDs only
    /// narrow the lookup; they never authorize a replacement body.
    pub fn get_owned(
        &self,
        function: FunctionId,
        lease: &runmat_value::FunctionDefinitionLease,
    ) -> Option<&FunctionBytecode> {
        if lease.function() != function.0 {
            return None;
        }
        if let Some(current) = self.functions.get(&function) {
            if lease.matches(&current.definition_owner) {
                return Some(current);
            }
        }
        // A reused session number is not authority for the new body. The
        // bound lease selects its retained immutable body; owned_view also
        // overlays every leased static callee before that body runs.
        if let Some(body) = self.retired.get(&function).and_then(|bodies| {
            bodies
                .iter()
                .find(|body| lease.matches(&body.definition_owner))
        }) {
            return Some(body);
        }
        // Source views form an older-to-newer admission DAG. Searching their
        // private body maps by the exact owner does not expose names/current.
        let mut pending = self.selected_site_views.iter().collect::<Vec<_>>();
        let mut visited = std::collections::HashSet::new();
        while let Some(selected) = pending.pop() {
            let view = &selected.registry;
            if !visited.insert(std::rc::Rc::as_ptr(&view.data)) {
                continue;
            }
            if let Some(body) = view
                .functions
                .get(&function)
                .filter(|body| lease.matches(&body.definition_owner))
                .or_else(|| {
                    view.retired.get(&function).and_then(|bodies| {
                        bodies
                            .iter()
                            .find(|body| lease.matches(&body.definition_owner))
                    })
                })
            {
                return Some(body);
            }
            pending.extend(view.selected_site_views.iter());
        }
        None
    }

    /// Retain an exact precompile target body without publishing its name or
    /// copying its bytecode payload. The caller must still carry a counted
    /// site lease; this view only supplies immutable code for that lease.
    pub fn retain_selected_site_view(
        &mut self,
        candidate: &Self,
        binding: &runmat_value::BoundFunctionBinding,
    ) -> Result<(), &'static str> {
        let lease = binding
            .authority
            .local_definition()
            .ok_or("selected site target has no exact definition lease")?;
        let body = candidate
            .get_owned(FunctionId(binding.function), lease)
            .ok_or("selected site target is absent from its exact source view")?;
        let source = body
            .selected_source_owner
            .as_ref()
            .ok_or("selected site target has no issued source owner")?;
        if !source.is_live()
            || body
                .selected_executable_product
                .as_ref()
                .is_none_or(|product| !product.source_owner().same_issuance(source))
        {
            return Err("selected site target differs from its admitted source");
        }
        if self
            .get_owned(FunctionId(binding.function), lease)
            .is_some()
        {
            return Ok(());
        }
        // Strip the candidate's invocation overlay. Keeping it here would
        // retain its counted leases even after every escaped value is gone.
        self.selected_site_views.push(SelectedSiteView {
            registry: candidate.current_view(),
            root_owner: std::sync::Arc::clone(lease.owner_arc()),
        });
        Ok(())
    }

    /// Build the executable view for an authorized local callable. Retired
    /// dependencies become executable by their original IDs in this view but
    /// remain absent from name and source lookup.
    pub fn owned_view(
        &self,
        function: FunctionId,
        authority: &runmat_value::BoundFunctionAuthority,
    ) -> Option<Self> {
        if self
            .static_overlay
            .as_ref()
            .is_some_and(|overlay| overlay.authority.shares_local_graph(authority))
        {
            self.get_owned(function, authority.local_lease(function.0)?)?;
            return Some(self.clone());
        }
        let runmat_value::BoundFunctionAuthority::Local {
            definition,
            dependencies,
            ..
        } = authority
        else {
            return None;
        };
        for lease in std::iter::once(definition).chain(dependencies.iter()) {
            let id = FunctionId(lease.function());
            self.get_owned(id, lease)?;
        }
        authority.local_lease(function.0)?;
        Some(Self {
            data: std::rc::Rc::clone(&self.data),
            static_overlay: Some(std::rc::Rc::new(StaticDefinitionOverlay {
                authority: authority.clone(),
                execution_stacks: OnceLock::new(),
            })),
        })
    }

    /// Retain the complete statically bound call/closure graph behind a
    /// function handle. An unresolved static edge makes issuance unavailable.
    pub fn definition_dependency_owners(
        &self,
        function: FunctionId,
    ) -> Option<
        Vec<(
            FunctionId,
            std::sync::Arc<runmat_value::FunctionDefinitionOwner>,
        )>,
    > {
        self.definition_dependency_owners_checked(function).ok()
    }

    /// Preserve typed superclass graph failures for consumers that need to
    /// distinguish an ambiguous admitted target from a missing definition.
    pub(crate) fn definition_dependency_owners_checked(
        &self,
        function: FunctionId,
    ) -> Result<
        Vec<(
            FunctionId,
            std::sync::Arc<runmat_value::FunctionDefinitionOwner>,
        )>,
        SuperConstructorDependencyError,
    > {
        if let Some(overlay) = &self.static_overlay {
            let runmat_value::BoundFunctionAuthority::Local {
                definition,
                dependencies,
                ..
            } = &overlay.authority
            else {
                return Err(SuperConstructorDependencyError::MissingOrigin);
            };
            let selected = overlay
                .authority
                .local_lease(function.0)
                .ok_or(SuperConstructorDependencyError::MissingOrigin)?;
            let body = self
                .get_owned(function, selected)
                .ok_or(SuperConstructorDependencyError::MissingOrigin)?;
            let mut owners = vec![(function, std::sync::Arc::clone(&body.definition_owner))];
            for lease in std::iter::once(definition).chain(dependencies.iter()) {
                if lease.function() == function.0 {
                    continue;
                }
                let id = FunctionId(lease.function());
                let body = self
                    .get_owned(id, lease)
                    .ok_or(SuperConstructorDependencyError::MissingOrigin)?;
                owners.push((id, std::sync::Arc::clone(&body.definition_owner)));
            }
            return Ok(owners);
        }
        definition_dependencies::owners(self, function)
    }

    /// Enumerate only definitions selected by the current execution owner.
    /// A retained overlay never falls back to later published source members.
    fn execution_source_group(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        source: runmat_runtime::source_context::RuntimeSourceRef,
    ) -> Vec<&FunctionBytecode> {
        let Some(source_id) = source.resolve_in(runtime) else {
            return Vec::new();
        };
        let mut bodies = Vec::new();
        if let Some(overlay) = &self.static_overlay {
            if let runmat_value::BoundFunctionAuthority::Local {
                definition,
                dependencies,
                ..
            } = &overlay.authority
            {
                for lease in std::iter::once(definition).chain(dependencies.iter()) {
                    if let Some(body) = self.get_owned(FunctionId(lease.function()), lease) {
                        if body.source_id.and_then(|id| id.resolve_in(runtime)) == Some(source_id) {
                            bodies.push(body);
                        }
                    }
                }
            }
        } else {
            let portable = runtime
                .portable_ordinal_for_source(source)
                .map(BytecodeSourceRef::Program);
            for key in [Some(BytecodeSourceRef::Runtime(source)), portable]
                .into_iter()
                .flatten()
            {
                for id in self
                    .source_functions
                    .get(&key)
                    .map(Vec::as_slice)
                    .unwrap_or(&[])
                {
                    if let Some(body) = self.get_current(*id) {
                        if body.source_id.and_then(|id| id.resolve_in(runtime)) == Some(source_id) {
                            bodies.push(body);
                        }
                    }
                }
            }
        }
        bodies.sort_by_key(|body| body.function);
        bodies.dedup_by_key(|body| body.function);
        bodies
    }

    fn prune_retired(&mut self) {
        self.retired.retain(|_, bodies| {
            bodies.retain(|body| body.definition_owner.has_handles());
            !bodies.is_empty()
        });
    }

    /// Retain exact bodies whose definition owners escaped from a candidate
    /// publication. Names and current-source membership remain unchanged;
    /// only an exact bound authority can select these bodies later.
    pub fn retain_live_owned_definitions_from(
        &mut self,
        candidate: &Self,
    ) -> std::collections::HashSet<runmat_execution::FunctionBindingOrigin> {
        self.execution_stacks.take();
        self.prune_retired();
        self.selected_site_views
            .retain(|view| view.root_owner.has_handles());
        for view in &candidate.selected_site_views {
            if !view.root_owner.has_handles() {
                continue;
            }
            if !self.selected_site_views.iter().any(|existing| {
                std::rc::Rc::ptr_eq(&existing.registry.data, &view.registry.data)
                    && std::sync::Arc::ptr_eq(&existing.root_owner, &view.root_owner)
            }) {
                self.selected_site_views.push(view.clone());
            }
        }
        let mut origins = std::collections::HashSet::new();
        for body in candidate
            .functions
            .values()
            .chain(candidate.retired.values().flatten())
            .filter(|body| body.definition_owner.has_handles())
        {
            if let Some(origin) = body.binding_origin {
                origins.insert(origin);
            }
            let retained = self.retired.entry(body.function).or_default();
            if retained.iter().any(|existing| {
                std::sync::Arc::ptr_eq(&existing.definition_owner, &body.definition_owner)
            }) {
                continue;
            }
            retained.push(body.clone());
        }
        origins
    }

    pub fn resolve_name(&self, name: &str) -> Option<FunctionId> {
        self.names.get(name).copied().filter(|function| {
            self.functions
                .get(function)
                .is_some_and(|body| body.source_role != runmat_types::FunctionSourceRole::Nested)
        })
    }

    pub fn resolve_name_in_private_scope(
        &self,
        private_owner_scope: &str,
        name: &str,
    ) -> Option<FunctionId> {
        if private_owner_scope.is_empty() || name.contains('.') {
            return None;
        }
        let scoped_name = format!("{private_owner_scope}.__private__.{name}");
        self.resolve_name(&scoped_name)
    }

    pub fn insert_replacing_name(&mut self, mut function: FunctionBytecode) {
        self.static_overlay = None;
        let data = std::rc::Rc::make_mut(&mut self.data);
        #[cfg(feature = "native-accel")]
        data.invalidate_selected_image_views();
        data.execution_stacks.take();
        #[cfg(feature = "native-accel")]
        data.admitted_fusion_scaffolds.take();
        let mut pruned = false;
        data.retired.retain(|_, bodies| {
            let old_len = bodies.len();
            bodies.retain(|body| body.definition_owner.has_handles());
            pruned |= bodies.len() != old_len;
            !bodies.is_empty()
        });
        if pruned {
            data.owner_index.get_mut().take();
        }
        // An ordinary fresh owner cannot alias this registry. A cloned owner
        // uses the derived pointer index before any name removal can retire a
        // body; only the current/retired bodies authorize the index entries.
        if std::sync::Arc::strong_count(&function.definition_owner) > 1
            && self.owner_index_conflicts(&function)
        {
            function.definition_owner = fresh_definition_owner();
        }
        if function.source_role != runmat_types::FunctionSourceRole::Nested {
            if let Some(previous) = self.resolve_name(&function.display_name) {
                if previous != function.function {
                    self.remove(previous);
                }
            }
        }
        let function_id = function.function;
        if let Some(previous) = self.functions.get(&function_id) {
            if previous.binding_origin.is_some()
                && previous.binding_origin == function.binding_origin
                && (std::sync::Arc::strong_count(&previous.definition_owner) == 1
                    || !self.owner_index_conflicts(previous))
            {
                // A binding origin includes the complete program revision and
                // function ordinal. Re-publishing that immutable definition
                // keeps existing value leases valid across its decoration.
                function.definition_owner = std::sync::Arc::clone(&previous.definition_owner);
            }
        }
        // Refreshing an existing identity does not retire its live aliases,
        // but its previous source membership must still be detached.
        {
            let data = std::rc::Rc::make_mut(&mut self.data);
            if let Some(old) = data.detach_definition(function_id) {
                if old.definition_owner.has_handles() {
                    data.retain_definition(old);
                }
            }
        }
        let data = std::rc::Rc::make_mut(&mut self.data);
        // A nested definition cannot inherit textual aliases previously
        // published under the same ID. A named replacement keeps its other
        // published aliases while refreshing its display name.
        if function.source_role == runmat_types::FunctionSourceRole::Nested {
            data.names.retain(|_, target| *target != function_id);
        } else {
            data.names
                .insert(function.display_name.clone(), function_id);
        }
        if let Some(source_id) = function.source_id {
            let functions = data.source_functions.entry(source_id).or_default();
            if !functions.contains(&function_id) {
                functions.push(function_id);
            }
        }
        if let Some(index) = data.owner_index.get_mut().as_mut() {
            std::sync::Arc::make_mut(index).insert(&function);
        }
        data.functions.insert(function_id, function);
    }

    /// Stage a freshly admitted anonymous body for one closure issuance.
    /// It is current by its compiler-assigned numeric ID only within this
    /// provisional view; no display-name alias is published or displaced.
    /// Escaped bindings retain the exact owner when that view rolls back.
    pub fn insert_selected_unnamed(
        &mut self,
        function: FunctionBytecode,
    ) -> Result<(), &'static str> {
        if !matches!(
            function.source_role,
            runmat_types::FunctionSourceRole::Anonymous | runmat_types::FunctionSourceRole::Nested
        ) {
            return Err("unnamed selected body is not a closure definition");
        }
        if self.functions.contains_key(&function.function)
            || self.owner_index_conflicts(&function)
            || function
                .selected_source_owner
                .as_ref()
                .is_none_or(|owner| !owner.is_live())
            || function
                .selected_executable_product
                .as_ref()
                .is_none_or(|product| {
                    function
                        .selected_source_owner
                        .as_ref()
                        .is_none_or(|owner| !product.source_owner().same_issuance(owner))
                })
        {
            return Err("unnamed selected body conflicts with its exact admission");
        }
        self.static_overlay = None;
        let data = std::rc::Rc::make_mut(&mut self.data);
        #[cfg(feature = "native-accel")]
        data.invalidate_selected_image_views();
        data.execution_stacks.take();
        #[cfg(feature = "native-accel")]
        data.admitted_fusion_scaffolds.take();
        if let Some(source) = function.source_id {
            data.source_functions
                .entry(source)
                .or_default()
                .push(function.function);
        }
        if let Some(index) = data.owner_index.get_mut().as_mut() {
            std::sync::Arc::make_mut(index).insert(&function);
        }
        data.functions.insert(function.function, function);
        Ok(())
    }

    fn owner_index_conflicts(&self, function: &FunctionBytecode) -> bool {
        let mut cached = self.data.owner_index.borrow_mut();
        let index = cached.get_or_insert_with(|| {
            std::sync::Arc::new(owner_index::OwnerIndex::from_bodies(
                self.functions
                    .values()
                    .chain(self.retired.values().flatten()),
            ))
        });
        index.conflicts(function)
    }

    pub fn remove(&mut self, function: FunctionId) -> Option<FunctionBytecode> {
        self.execution_stacks.take();
        self.prune_retired();
        let removed = std::rc::Rc::make_mut(&mut self.data).detach_definition(function)?;
        self.names.retain(|_, target| *target != function);
        if removed.definition_owner.has_handles() {
            std::rc::Rc::make_mut(&mut self.data).retain_definition(removed.clone());
        }
        Some(removed)
    }

    pub fn remove_source(
        &mut self,
        source: runmat_runtime::source_context::RuntimeSourceRef,
    ) -> Vec<FunctionBytecode> {
        let source = BytecodeSourceRef::Runtime(source);
        self.execution_stacks.take();
        let ids = self.source_functions.remove(&source).unwrap_or_default();
        let mut removed = Vec::new();
        for id in ids {
            if let Some(function) = self.remove(id) {
                removed.push(function);
            }
        }
        removed
    }

    pub fn functions_for_source(
        &self,
        source: runmat_runtime::source_context::RuntimeSourceRef,
    ) -> &[FunctionId] {
        self.functions_for_source_ref(BytecodeSourceRef::Runtime(source))
    }

    pub fn functions_for_source_ref(&self, source: BytecodeSourceRef) -> &[FunctionId] {
        self.source_functions
            .get(&source)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Returns the stack required by this function and its reachable callees.
    pub fn execution_stack(&self, function: FunctionId) -> runmat_types::ExecutionStackRequirement {
        if let Some(overlay) = &self.static_overlay {
            return *overlay
                .execution_stacks
                .get_or_init(|| {
                    let runmat_value::BoundFunctionAuthority::Local {
                        definition,
                        dependencies,
                        ..
                    } = &overlay.authority
                    else {
                        return HashMap::new();
                    };
                    std::iter::once(definition)
                        .chain(dependencies.iter())
                        .map(|lease| FunctionId(lease.function()))
                        .map(|function| {
                            (
                                function,
                                self.execution_stack_inner(function, &mut HashSet::new()),
                            )
                        })
                        .collect()
                })
                .get(&function)
                .unwrap_or(&runmat_types::ExecutionStackRequirement::Process);
        }
        *self
            .execution_stacks
            .get_or_init(|| {
                self.functions
                    .keys()
                    .copied()
                    .map(|function| {
                        (
                            function,
                            self.execution_stack_inner(function, &mut HashSet::new()),
                        )
                    })
                    .collect()
            })
            .get(&function)
            .unwrap_or(&runmat_types::ExecutionStackRequirement::Process)
    }

    fn execution_stack_inner(
        &self,
        function: FunctionId,
        visiting: &mut HashSet<FunctionId>,
    ) -> runmat_types::ExecutionStackRequirement {
        if !visiting.insert(function) {
            return runmat_types::ExecutionStackRequirement::Any;
        }
        let required = self
            .get(function)
            .map(|bytecode| {
                bytecode
                    .instructions
                    .iter()
                    .map(|instruction| self.instruction_execution_stack(instruction, visiting))
                    .max()
                    .unwrap_or(runmat_types::ExecutionStackRequirement::Any)
            })
            .unwrap_or(runmat_types::ExecutionStackRequirement::Process);
        visiting.remove(&function);
        required
    }

    fn instruction_execution_stack(
        &self,
        instruction: &Instr,
        visiting: &mut HashSet<FunctionId>,
    ) -> runmat_types::ExecutionStackRequirement {
        use runmat_types::CallableIdentity;

        let identity_stack =
            |identity: &CallableIdentity, registry: &Self, visiting: &mut HashSet<FunctionId>| {
                match identity {
                    CallableIdentity::BoundFunction(function)
                    | CallableIdentity::AnonymousFunction(function)
                    | CallableIdentity::ExternalFunction { function, .. } => {
                        registry.execution_stack_inner(*function, visiting)
                    }
                    CallableIdentity::Builtin(builtin) => builtin_execution_stack(&builtin.0),
                    CallableIdentity::DynamicName(name) => {
                        if runmat_builtins::builtin_name_is_known(&name.0) {
                            builtin_execution_stack(&name.0)
                        } else {
                            runmat_types::ExecutionStackRequirement::Process
                        }
                    }
                    CallableIdentity::Method(method) => registry
                        .registered_named_method_execution_stack(
                            &runmat_types::MethodName::from(method.0.as_str()),
                            visiting,
                        )
                        .unwrap_or(runmat_types::ExecutionStackRequirement::Process),
                    CallableIdentity::ExternalName(_) | CallableIdentity::Imported(_) => {
                        runmat_types::ExecutionStackRequirement::Process
                    }
                }
            };

        match instruction {
            Instr::CallBuiltinMulti(name, ..)
            | Instr::CallBuiltinMultiUsingOutputSlot(name, ..)
            | Instr::CallBuiltinExpandMultiOutput(name, ..)
            | Instr::CallBuiltinExpandMultiOutputUsingOutputSlot(name, ..) => {
                builtin_execution_stack(name)
            }
            Instr::CallFunctionMulti { identity, .. }
            | Instr::CallFunctionMultiUsingOutputSlot { identity, .. }
            | Instr::CallWorkspaceFirstMulti { identity, .. }
            | Instr::CallWorkspaceFirstMultiUsingOutputSlot { identity, .. }
            | Instr::CallFunctionExpandMultiOutput { identity, .. }
            | Instr::CallWorkspaceFirstExpandMultiOutput { identity, .. }
            | Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot { identity, .. } => {
                identity_stack(identity, self, visiting)
            }
            Instr::CallSemanticFunctionMulti(function, ..)
            | Instr::CallSemanticFunctionMultiUsingOutputSlot(function, ..)
            | Instr::CallSemanticFunctionExpandMultiOutput(function, ..)
            | Instr::CreateSemanticFuture(function, ..)
            | Instr::CreateSemanticFutureExpandMultiOutput(function, ..) => {
                self.execution_stack_inner(*function, visiting)
            }
            Instr::CallSemanticNestedFunctionMulti { function, .. }
            | Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { function, .. }
            | Instr::CallSemanticNestedFunctionExpandMultiOutput { function, .. } => {
                self.execution_stack_inner(*function, visiting)
            }
            Instr::CallMethodOrMemberIndexMulti { identity, .. }
            | Instr::CallMethodOrMemberIndexExpandMultiOutput { identity, .. } => identity
                .display_name()
                .and_then(|name| self.registered_member_or_method_execution_stack(&name, visiting))
                .unwrap_or_else(|| identity_stack(identity, self, visiting)),
            Instr::CallFevalMulti(..)
            | Instr::CallFevalMultiUsingOutputSlot(..)
            | Instr::CallFevalExpandMultiOutput(..)
            | Instr::CallFevalExpandMultiOutputUsingOutputSlot(..)
            | Instr::CallSuperConstructorMulti { .. }
            | Instr::CallSuperMethodMulti { .. } => {
                runmat_types::ExecutionStackRequirement::Process
            }
            instruction => self.registered_method_execution_stack(instruction, visiting),
        }
    }

    fn registered_method_execution_stack(
        &self,
        instruction: &Instr,
        visiting: &mut HashSet<FunctionId>,
    ) -> runmat_types::ExecutionStackRequirement {
        if !instruction_has_dynamic_dispatch(instruction) {
            return runmat_types::ExecutionStackRequirement::Any;
        }
        if self.static_overlay.is_some() {
            // Class registration and ordinary name lookup describe the
            // published graph. They cannot authorize an old retained target.
            return runmat_types::ExecutionStackRequirement::Process;
        }
        let mut targets = HashSet::new();
        for function in self.functions.values() {
            for candidate in &function.instructions {
                let Instr::RegisterClass { methods, .. } = candidate else {
                    continue;
                };
                for method in methods {
                    if instruction_may_dispatch_method(instruction, &method.name) {
                        targets.insert(method.function_name.as_str());
                    }
                }
            }
        }
        self.registered_targets_execution_stack(targets, visiting)
            .unwrap_or(runmat_types::ExecutionStackRequirement::Any)
    }

    fn registered_named_method_execution_stack(
        &self,
        requested_name: &runmat_types::MethodName,
        visiting: &mut HashSet<FunctionId>,
    ) -> Option<runmat_types::ExecutionStackRequirement> {
        if self.static_overlay.is_some() {
            return Some(runmat_types::ExecutionStackRequirement::Process);
        }
        let mut targets = HashSet::new();
        for function in self.functions.values() {
            for candidate in &function.instructions {
                let Instr::RegisterClass { methods, .. } = candidate else {
                    continue;
                };
                for method in methods {
                    if &method.name == requested_name {
                        targets.insert(method.function_name.as_str());
                    }
                }
            }
        }
        self.registered_targets_execution_stack(targets, visiting)
    }

    fn registered_member_or_method_execution_stack(
        &self,
        requested_name: &str,
        visiting: &mut HashSet<FunctionId>,
    ) -> Option<runmat_types::ExecutionStackRequirement> {
        if self.static_overlay.is_some() {
            return Some(runmat_types::ExecutionStackRequirement::Process);
        }
        if let Some(requirement) = self.registered_named_method_execution_stack(
            &runmat_types::MethodName::from(requested_name),
            visiting,
        ) {
            return Some(requirement);
        }
        let mut property_is_registered = false;
        let mut targets = HashSet::new();
        for function in self.functions.values() {
            for candidate in &function.instructions {
                let Instr::RegisterClass {
                    properties,
                    methods,
                    ..
                } = candidate
                else {
                    continue;
                };
                if !properties
                    .iter()
                    .any(|property| property.name.display_name() == requested_name)
                {
                    continue;
                }
                property_is_registered = true;
                let getter = runmat_types::MethodName::property_getter(&runmat_types::MemberName(
                    requested_name.to_owned(),
                ));
                for method in methods {
                    if runmat_runtime::OBJECT_SUBSREF_METHOD.is(&method.name)
                        || method.name == getter
                    {
                        targets.insert(method.function_name.as_str());
                    }
                }
            }
        }
        property_is_registered.then(|| {
            self.registered_targets_execution_stack(targets, visiting)
                .unwrap_or(runmat_types::ExecutionStackRequirement::Any)
        })
    }

    fn registered_targets_execution_stack(
        &self,
        targets: HashSet<&str>,
        visiting: &mut HashSet<FunctionId>,
    ) -> Option<runmat_types::ExecutionStackRequirement> {
        targets
            .into_iter()
            .map(|target| {
                self.resolve_name(target)
                    .map(|function| self.execution_stack_inner(function, visiting))
                    .unwrap_or_else(|| {
                        if runmat_builtins::builtin_name_is_known(target) {
                            builtin_execution_stack(target)
                        } else {
                            runmat_types::ExecutionStackRequirement::Process
                        }
                    })
            })
            .max()
    }
}

#[cfg(test)]
#[path = "program/registry_retirement_tests.rs"]
mod registry_retirement_tests;

fn instruction_has_dynamic_dispatch(instruction: &Instr) -> bool {
    !dynamic_dispatch_method_names(instruction).is_empty()
        || matches!(
            instruction,
            Instr::Index(_)
                | Instr::IndexSlice(..)
                | Instr::IndexCell { .. }
                | Instr::IndexCellExpand { .. }
                | Instr::IndexCellList { .. }
                | Instr::StoreIndex(_)
                | Instr::StoreIndexCell { .. }
                | Instr::StoreIndexDelete(_)
                | Instr::StoreIndexCellDelete { .. }
                | Instr::StoreSlice(..)
                | Instr::StoreSliceDelete(..)
                | Instr::LoadMember(_)
                | Instr::LoadMemberOrInit(_)
                | Instr::LoadMemberDynamic
                | Instr::LoadMemberDynamicOrInit
                | Instr::LoadMemberSequence { .. }
                | Instr::LoadMemberDynamicSequence { .. }
                | Instr::StoreMember(_)
                | Instr::StoreMemberOrInit(_)
                | Instr::StoreMemberDynamic
                | Instr::StoreMemberDynamicOrInit
        )
}

fn instruction_may_dispatch_method(
    instruction: &Instr,
    method_name: &runmat_types::MethodName,
) -> bool {
    if dynamic_dispatch_method_names(instruction)
        .iter()
        .any(|candidate| candidate.is(method_name))
    {
        return true;
    }
    match instruction {
        Instr::Index(_)
        | Instr::IndexSlice(..)
        | Instr::IndexCell { .. }
        | Instr::IndexCellExpand { .. }
        | Instr::IndexCellList { .. } => runmat_runtime::OBJECT_SUBSREF_METHOD.is(method_name),
        Instr::StoreIndex(_)
        | Instr::StoreIndexCell { .. }
        | Instr::StoreIndexDelete(_)
        | Instr::StoreIndexCellDelete { .. }
        | Instr::StoreSlice(..)
        | Instr::StoreSliceDelete(..) => runmat_runtime::OBJECT_SUBSASGN_METHOD.is(method_name),
        Instr::LoadMember(name)
        | Instr::LoadMemberOrInit(name)
        | Instr::LoadMemberSequence { member: name, .. } => {
            runmat_runtime::OBJECT_SUBSREF_METHOD.is(method_name)
                || method_name == &runmat_types::MethodName::property_getter(name)
        }
        Instr::LoadMemberDynamic
        | Instr::LoadMemberDynamicOrInit
        | Instr::LoadMemberDynamicSequence { .. } => {
            runmat_runtime::OBJECT_SUBSREF_METHOD.is(method_name)
                || method_name.is_property_getter()
        }
        Instr::StoreMember(name) | Instr::StoreMemberOrInit(name) => {
            runmat_runtime::OBJECT_SUBSASGN_METHOD.is(method_name)
                || method_name == &runmat_types::MethodName::property_setter(name)
        }
        Instr::StoreMemberDynamic | Instr::StoreMemberDynamicOrInit => {
            runmat_runtime::OBJECT_SUBSASGN_METHOD.is(method_name)
                || method_name.is_property_setter()
        }
        _ => false,
    }
}

fn dynamic_dispatch_method_names(instruction: &Instr) -> &'static [runmat_types::StaticMethodName] {
    use runmat_types::StaticMethodName;

    const PLUS: &[StaticMethodName] = &[StaticMethodName::new("plus")];
    const MINUS: &[StaticMethodName] = &[StaticMethodName::new("minus")];
    const MTIMES: &[StaticMethodName] = &[StaticMethodName::new("mtimes")];
    const RIGHT_DIVIDE: &[StaticMethodName] = &[
        StaticMethodName::new("mrdivide"),
        StaticMethodName::new("rdivide"),
    ];
    const LEFT_DIVIDE: &[StaticMethodName] = &[
        StaticMethodName::new("mldivide"),
        StaticMethodName::new("ldivide"),
    ];
    const POWER: &[StaticMethodName] = &[
        StaticMethodName::new("mpower"),
        StaticMethodName::new("power"),
    ];
    const NEGATE: &[StaticMethodName] = &[
        StaticMethodName::new("uminus"),
        StaticMethodName::new("times"),
    ];
    const UPLUS: &[StaticMethodName] = &[StaticMethodName::new("uplus")];
    const TRANSPOSE: &[StaticMethodName] = &[StaticMethodName::new("transpose")];
    const CONJUGATE_TRANSPOSE: &[StaticMethodName] = &[StaticMethodName::new("ctranspose")];
    const TIMES: &[StaticMethodName] = &[StaticMethodName::new("times")];
    const RDIVIDE: &[StaticMethodName] = &[StaticMethodName::new("rdivide")];
    const ELEMENTWISE_POWER: &[StaticMethodName] = &[StaticMethodName::new("power")];
    const LDIVIDE: &[StaticMethodName] = &[StaticMethodName::new("ldivide")];
    const LESS_EQUAL: &[StaticMethodName] = &[
        StaticMethodName::new("le"),
        StaticMethodName::new("gt"),
        StaticMethodName::new("ge"),
        StaticMethodName::new("lt"),
    ];
    const LESS: &[StaticMethodName] = &[StaticMethodName::new("lt"), StaticMethodName::new("gt")];
    const GREATER: &[StaticMethodName] =
        &[StaticMethodName::new("gt"), StaticMethodName::new("lt")];
    const GREATER_EQUAL: &[StaticMethodName] = &[
        StaticMethodName::new("ge"),
        StaticMethodName::new("lt"),
        StaticMethodName::new("le"),
        StaticMethodName::new("gt"),
    ];
    const EQUAL: &[StaticMethodName] = &[StaticMethodName::new("eq")];
    const NOT_EQUAL: &[StaticMethodName] = &[StaticMethodName::new("ne")];
    const NOT: &[StaticMethodName] = &[StaticMethodName::new("not")];
    const AND: &[StaticMethodName] = &[StaticMethodName::new("and")];
    const OR: &[StaticMethodName] = &[StaticMethodName::new("or")];

    match instruction {
        Instr::Add => PLUS,
        Instr::Sub => MINUS,
        Instr::Mul => MTIMES,
        Instr::RightDiv => RIGHT_DIVIDE,
        Instr::LeftDiv => LEFT_DIVIDE,
        Instr::Pow => POWER,
        Instr::Neg => NEGATE,
        Instr::UPlus => UPLUS,
        Instr::Transpose => TRANSPOSE,
        Instr::ConjugateTranspose => CONJUGATE_TRANSPOSE,
        Instr::ElemMul => TIMES,
        Instr::ElemDiv => RDIVIDE,
        Instr::ElemPow => ELEMENTWISE_POWER,
        Instr::ElemLeftDiv => LDIVIDE,
        Instr::LessEqual => LESS_EQUAL,
        Instr::Less => LESS,
        Instr::Greater => GREATER,
        Instr::GreaterEqual => GREATER_EQUAL,
        Instr::Equal => EQUAL,
        Instr::NotEqual => NOT_EQUAL,
        Instr::LogicalNot => NOT,
        Instr::LogicalAnd => AND,
        Instr::LogicalOr => OR,
        _ => &[],
    }
}
fn builtin_execution_stack(name: &str) -> runmat_types::ExecutionStackRequirement {
    runmat_builtins::builtin_execution_stack_requirement(name)
}

impl runmat_runtime::call::descriptor::FunctionNameResolver for FunctionRegistry {
    fn name_resolution_completeness(
        &self,
    ) -> runmat_runtime::call::descriptor::NameResolutionCompleteness {
        runmat_runtime::call::descriptor::NameResolutionCompleteness::Complete
    }

    fn resolve_function(&self, name: &str) -> Option<FunctionId> {
        self.resolve_name(name)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bytecode {
    pub instructions: Vec<Instr>,
    /// Compiler-derived operand-stack capacity. Artifact decoders validate
    /// this claim against the same CFG used by bytecode verification.
    pub stack_geometry: super::BytecodeStackGeometry,
    /// Innermost lexical catch at each instruction, derived from MIR.
    pub catch_scopes: Vec<Option<runmat_mir::CatchScopeId>>,
    #[serde(skip)]
    pub active_function: Option<FunctionId>,
    #[serde(skip)]
    pub active_class_method_owner: Option<runmat_types::ClassMethodOwner>,
    #[serde(default)]
    pub instr_spans: Vec<runmat_hir::Span>,
    #[serde(default)]
    pub call_arg_spans: Vec<Option<Vec<runmat_hir::Span>>>,
    #[serde(default)]
    pub coverage_sites: Vec<Vec<u64>>,
    /// The compiler's function-local MIR statement boundaries. This is the
    /// same authority serialized by `FunctionBytecode`, not a second mapping
    /// inferred from source spans or runtime graph nodes.
    #[serde(default, with = "crate::layout::resume_point_map_serde")]
    pub resume_points: std::collections::BTreeMap<runmat_types::ProgramPointId, usize>,
    #[serde(default)]
    pub source_id: Option<BytecodeSourceRef>,
    pub var_count: usize,
    #[serde(default)]
    pub bound_functions: HashMap<FunctionId, FunctionBytecode>,
    #[serde(default)]
    pub function_registry: FunctionRegistry,
    #[serde(default)]
    pub var_types: Vec<Type>,
    #[serde(default)]
    pub var_names: HashMap<usize, String>,
    #[serde(default, with = "slot_set_serde")]
    pub initially_unassigned_slots: HashSet<usize>,
    #[serde(default)]
    pub layout: Option<VmAssemblyLayout>,
    /// Only the exact admitted function body can carry this compiler layout
    /// into an ordinary function call. It is never serialized as bytecode.
    #[cfg(feature = "native-accel")]
    #[serde(skip)]
    pub(crate) function_private_layout: Option<FunctionPrivateSlotLayout>,
    /// Only the exact canonical registry body can install this runtime-only
    /// view. Public mutable Bytecode/entrypoint paths retain checked lookup.
    #[cfg(feature = "native-accel")]
    #[serde(skip)]
    pub(crate) sealed_function_image: SealedFunctionImage,
    #[serde(default)]
    pub async_metadata: AsyncMetadata,
    /// Canonical region identities mapped onto function-local bytecode PCs.
    #[serde(default)]
    pub regions: Vec<BytecodeRegion>,
    /// Compiler-bound executable records for analyzed `parfor` regions.
    #[serde(default)]
    pub parfor_regions: Vec<BytecodeParforRegion>,
    /// Compiler-bound executable records for analyzed SPMD regions.
    #[serde(default)]
    pub spmd_regions: Vec<BytecodeSpmdRegion>,
    #[serde(default)]
    pub distributed_values: Vec<runmat_types::DistributedValueContract>,
    #[serde(default)]
    pub collective_contracts: Vec<runmat_types::CollectiveContract>,
    #[cfg(feature = "native-accel")]
    #[serde(default)]
    pub accel_graph: Option<AccelGraph>,
    #[cfg(feature = "native-accel")]
    #[serde(default)]
    pub fusion_groups: Vec<FusionGroup>,
    #[cfg(feature = "native-accel")]
    #[serde(default)]
    pub fusion_metadata: FusionMetadata,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AsyncMetadata {
    pub mir_spawn_site_count: usize,
    pub mir_spawn_sites: Vec<SpawnSite>,
    pub mir_await_site_count: usize,
    pub mir_await_sites: Vec<AwaitSite>,
    #[serde(default)]
    pub runtime_model: AsyncRuntimeModel,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AsyncRuntimeModel {
    LazyFutureDescriptorLane,
}

impl Default for AsyncRuntimeModel {
    fn default() -> Self {
        Self::LazyFutureDescriptorLane
    }
}

impl AsyncRuntimeModel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LazyFutureDescriptorLane => "lazy_future_descriptor_lane",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnSite {
    pub function: runmat_hir::FunctionId,
    pub block: runmat_mir::BasicBlockId,
    pub stmt_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwaitSite {
    pub function: runmat_hir::FunctionId,
    pub block: runmat_mir::BasicBlockId,
    pub resume: runmat_mir::BasicBlockId,
}

#[cfg(feature = "native-accel")]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FusionMetadata {
    pub mir_fusion_signal_count: usize,
    pub mir_fusion_candidate_group_count: usize,
    pub mir_fusion_candidate_groups: Vec<FusionCandidateGroup>,
    pub instruction_window_count: usize,
    #[serde(default)]
    pub instruction_windows: Vec<FusionInstructionWindow>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_projection: Option<std::sync::Arc<crate::bytecode::FusionStageProjection>>,
    /// Process-local selected-source descriptor. Serialized artifacts carry
    /// the underlying compiler facts, never this derived execution view.
    #[serde(skip)]
    pub(crate) selected_image: std::rc::Rc<vm_image_plan::SelectedImageCache>,
}

#[cfg(feature = "native-accel")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionCandidateGroup {
    pub id: usize,
    pub signal_count: usize,
    pub function: runmat_hir::FunctionId,
    pub block: runmat_mir::BasicBlockId,
    pub stmt_start: usize,
    pub stmt_end: usize,
    #[serde(default)]
    pub source_span: runmat_hir::Span,
}

#[cfg(feature = "native-accel")]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FusionInstructionKind {
    Elementwise,
    Reduction,
    Matmul,
}

#[cfg(feature = "native-accel")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FusionInstructionWindow {
    pub span: runmat_accelerate::graph::InstrSpan,
    pub kind: FusionInstructionKind,
}

#[cfg(feature = "native-accel")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeAccelGraphSource {
    NotMaterialized,
    RuntimeMaterializedFromInstructions,
}

#[cfg(feature = "native-accel")]
impl RuntimeAccelGraphSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotMaterialized => "not_materialized",
            Self::RuntimeMaterializedFromInstructions => "runtime_materialized_from_instructions",
        }
    }
}

impl Bytecode {
    /// Exact, half-open statement instruction ranges derived from the
    /// compiler's canonical resume points. Empty-emission statements remain
    /// visible; no fusion execution decision is made here.
    pub fn statement_pc_ranges(&self) -> Result<Vec<StatementPcRange>, StatementPcProjectionError> {
        project_statement_pcs(&self.resume_points, self.instructions.len())
    }

    pub fn validate_selected_source_owners(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<(), &'static str> {
        if matches!(self.source_id, Some(BytecodeSourceRef::Runtime(reference)) if reference.resolve_in(runtime).is_none())
            || self.bound_functions.values().any(|body| {
                matches!(body.source_id, Some(BytecodeSourceRef::Runtime(reference)) if reference.resolve_in(runtime).is_none())
            })
        {
            return Err("bytecode carries a source issued by another runtime owner");
        }
        self.function_registry
            .validate_selected_source_owners(runtime)
    }

    pub fn validate_portable_source_origins(&self) -> Result<(), &'static str> {
        if self
            .source_id
            .is_some_and(|source| !matches!(source, BytecodeSourceRef::Program(_)))
            || self.bound_functions.values().any(|body| {
                body.source_id
                    .is_some_and(|source| !matches!(source, BytecodeSourceRef::Program(_)))
            })
        {
            return Err("portable bytecode contains a session source identity");
        }
        self.function_registry.validate_portable_source_origins()
    }

    pub(crate) fn validate_catch_scope_metadata(&self) -> Result<(), String> {
        validate_catch_scope_metadata(&self.instructions, &self.catch_scopes)
    }

    pub fn empty() -> Self {
        Self {
            instructions: Vec::new(),
            stack_geometry: super::BytecodeStackGeometry::default(),
            catch_scopes: Vec::new(),
            active_function: None,
            active_class_method_owner: None,
            instr_spans: Vec::new(),
            call_arg_spans: Vec::new(),
            coverage_sites: Vec::new(),
            resume_points: std::collections::BTreeMap::new(),
            source_id: None,
            var_count: 0,
            bound_functions: HashMap::new(),
            function_registry: FunctionRegistry::default(),
            var_types: Vec::new(),
            var_names: HashMap::new(),
            initially_unassigned_slots: HashSet::new(),
            layout: None,
            #[cfg(feature = "native-accel")]
            function_private_layout: None,
            #[cfg(feature = "native-accel")]
            sealed_function_image: SealedFunctionImage::Unsealed,
            async_metadata: AsyncMetadata::default(),
            regions: Vec::new(),
            parfor_regions: Vec::new(),
            spmd_regions: Vec::new(),
            distributed_values: Vec::new(),
            collective_contracts: Vec::new(),
            #[cfg(feature = "native-accel")]
            accel_graph: None,
            #[cfg(feature = "native-accel")]
            fusion_groups: Vec::new(),
            #[cfg(feature = "native-accel")]
            fusion_metadata: FusionMetadata::default(),
        }
    }

    pub fn with_instructions(instructions: Vec<Instr>, var_count: usize) -> Self {
        let stack_geometry = super::derive_stack_geometry(&instructions)
            .unwrap_or(super::BytecodeStackGeometry::Dynamic);
        let instr_spans = vec![runmat_hir::Span::default(); instructions.len()];
        let call_arg_spans = vec![None; instructions.len()];
        let coverage_sites = vec![Vec::new(); instructions.len()];
        let catch_scopes = vec![None; instructions.len()];
        Self {
            instructions,
            stack_geometry,
            catch_scopes,
            instr_spans,
            call_arg_spans,
            coverage_sites,
            var_count,
            ..Self::empty()
        }
    }

    pub fn function_registry(&self) -> FunctionRegistry {
        if self.function_registry.functions.is_empty() && !self.bound_functions.is_empty() {
            return FunctionRegistry::new(self.bound_functions.clone());
        }
        self.function_registry.clone()
    }

    pub fn for_function(
        function: &FunctionBytecode,
        registry: FunctionRegistry,
        mut layout: VmAssemblyLayout,
    ) -> Self {
        // `derive_layout` has semantic slots but no emitted instruction PCs.
        // Only the exact selected compiled body may bind those PCs; an existing
        // conflicting map is never replaced by a coincident function ID.
        let exact_selected_body = registry
            .get(function.function)
            .is_some_and(|selected| std::ptr::eq(selected, function));
        if exact_selected_body {
            if let Some(selected_layout) = layout.functions.get_mut(&function.function) {
                if selected_layout.function == function.function
                    && selected_layout.local_count == function.var_count
                    && selected_layout.resume_points.is_empty()
                {
                    selected_layout.resume_points = function.resume_points.clone();
                }
            }
        }
        let mut bytecode = function.execution_bytecode(&registry);
        #[cfg(feature = "native-accel")]
        if !exact_selected_body || function.validate_fusion_stage_projection().is_err() {
            // A detached same-ID body may execute normally, but cannot borrow
            // the registry's selected MIR/portable-origin proposal authority.
            bytecode.fusion_metadata.stage_projection = None;
        }
        bytecode.var_types = vec![Type::Unknown; function.var_count];
        bytecode.layout = Some(layout);
        #[cfg(feature = "native-accel")]
        {
            // This public constructor accepts a caller-supplied layout. It
            // must use checked source lookup, never the canonical registry's
            // pre-finalized function view.
            bytecode.sealed_function_image = SealedFunctionImage::Unsealed;
        }
        bytecode
    }

    #[cfg(feature = "native-accel")]
    pub fn runtime_fusion_groups(&self) -> Vec<FusionGroup> {
        let metadata_present = self.fusion_metadata.mir_fusion_signal_count > 0
            || self.fusion_metadata.mir_fusion_candidate_group_count > 0
            || !self.fusion_metadata.mir_fusion_candidate_groups.is_empty()
            || self.fusion_metadata.instruction_window_count > 0
            || !self.fusion_metadata.instruction_windows.is_empty();

        if !metadata_present {
            return self.fusion_groups.clone();
        }

        if self.fusion_metadata.mir_fusion_candidate_group_count == 0
            || self.fusion_metadata.instruction_windows.is_empty()
        {
            return Vec::new();
        }
        let mut groups: Vec<FusionGroup> = self
            .fusion_metadata
            .instruction_windows
            .iter()
            .enumerate()
            .map(|(id, window)| FusionGroup {
                id,
                kind: match window.kind {
                    FusionInstructionKind::Elementwise => {
                        runmat_accelerate::fusion::FusionKind::ElementwiseChain
                    }
                    FusionInstructionKind::Reduction => {
                        runmat_accelerate::fusion::FusionKind::Reduction
                    }
                    FusionInstructionKind::Matmul => {
                        runmat_accelerate::fusion::FusionKind::MatmulEpilogue
                    }
                },
                nodes: Vec::new(),
                shape: runmat_accelerate::graph::ShapeInfo::Unknown,
                span: window.span.clone(),
                pattern: None,
                stack_layout: None,
            })
            .collect();
        super::fusion_private_temp::coalesce_reduction_groups(
            &mut groups,
            &super::fusion_private_temp::bridges(self),
        );
        groups
    }

    #[cfg(feature = "native-accel")]
    pub fn runtime_fusion_groups_for_graph(&self, graph: &AccelGraph) -> Vec<FusionGroup> {
        if graph.nodes.is_empty() {
            return Vec::new();
        }
        let mut groups = self.runtime_fusion_groups();
        if groups.is_empty() {
            return groups;
        }
        // Structural `*` windows are candidates, not execution decisions.
        // Refine from the same graph that owns scalar-vs-matrix normalization.
        // Never fabricate a scalar group from an empty or unsupported window.
        groups.retain_mut(|group| {
            use runmat_accelerate::{graph::AccelOpCategory, FusionKind};
            if group.kind != FusionKind::MatmulEpilogue {
                return true;
            }
            let mut has_matrix = false;
            let mut has_reduction = false;
            let mut has_elementwise = false;
            for node in graph.nodes.iter().filter(|node| {
                node.span.start >= group.span.start && node.span.end <= group.span.end
            }) {
                match node.category {
                    AccelOpCategory::MatMul => has_matrix = true,
                    AccelOpCategory::Reduction => has_reduction = true,
                    AccelOpCategory::Elementwise => has_elementwise = true,
                    AccelOpCategory::Transpose => {}
                    AccelOpCategory::Other => return false,
                }
            }
            if has_matrix {
                true
            } else if has_reduction {
                group.kind = FusionKind::Reduction;
                true
            } else if has_elementwise {
                group.kind = FusionKind::ElementwiseChain;
                true
            } else {
                false
            }
        });
        super::fusion_private_temp::coalesce_chain_groups(
            &mut groups,
            &super::fusion_private_temp::bridges(self),
        );
        if groups.iter().any(|group| group.stack_layout.is_none()) {
            // Span proposals carry no node ownership yet. Stack occurrence
            // discovery still needs the exact graph inventory in each span;
            // final preparation independently sanitizes and re-annotates the
            // selected nodes before installing any execution right.
            let mut stack_groups = groups.clone();
            for group in &mut stack_groups {
                if group.nodes.is_empty() {
                    group.nodes = graph
                        .nodes
                        .iter()
                        .filter(|node| {
                            group.span.start <= node.span.start && node.span.end <= group.span.end
                        })
                        .map(|node| node.id)
                        .collect();
                }
            }
            annotate_fusion_groups_with_stack_layout(&self.instructions, graph, &mut stack_groups);
            for (group, annotated) in groups.iter_mut().zip(stack_groups) {
                group.stack_layout = annotated.stack_layout;
            }
        }
        groups
    }

    #[cfg(feature = "native-accel")]
    pub fn runtime_accel_graph_for_fusion(
        &self,
        runtime_groups: &[FusionGroup],
    ) -> Option<AccelGraph> {
        self.runtime_accel_graph_for_fusion_with_source(runtime_groups)
            .0
    }

    #[cfg(feature = "native-accel")]
    pub fn runtime_accel_graph_for_fusion_with_source(
        &self,
        runtime_groups: &[FusionGroup],
    ) -> (Option<AccelGraph>, RuntimeAccelGraphSource) {
        if runtime_groups.is_empty() || self.fusion_metadata.mir_fusion_candidate_group_count == 0 {
            return (None, RuntimeAccelGraphSource::NotMaterialized);
        }
        (
            Some(build_accel_graph(&self.instructions, &self.var_types)),
            RuntimeAccelGraphSource::RuntimeMaterializedFromInstructions,
        )
    }
}

fn validate_catch_scope_metadata(
    instructions: &[Instr],
    catch_scopes: &[Option<runmat_mir::CatchScopeId>],
) -> Result<(), String> {
    let has_handler = instructions
        .iter()
        .any(|instruction| matches!(instruction, Instr::EnterTry { .. }));
    if catch_scopes.len() != instructions.len() && (has_handler || !catch_scopes.is_empty()) {
        return Err("bytecode lexical catch metadata length does not match instructions".into());
    }
    let mut targets = std::collections::HashSet::new();
    let mut declared_scopes = std::collections::HashSet::new();
    for instruction in instructions {
        if let Instr::EnterTry {
            scope, catch_pc, ..
        } = instruction
        {
            if !declared_scopes.insert(runmat_mir::CatchScopeId(*scope)) {
                return Err("bytecode lexical catch scope is declared more than once".into());
            }
            if !targets.insert(*catch_pc) {
                return Err("bytecode catch entry PC is ambiguous".into());
            }
            if catch_scopes.get(*catch_pc) != Some(&Some(runmat_mir::CatchScopeId(*scope))) {
                return Err("bytecode catch entry has no matching lexical scope".into());
            }
        }
    }
    if catch_scopes
        .iter()
        .flatten()
        .any(|scope| !declared_scopes.contains(scope))
    {
        return Err("bytecode catch membership names an undeclared lexical scope".into());
    }
    if catch_scopes.len() == instructions.len() {
        for (pc, instruction) in instructions.iter().enumerate() {
            if let Instr::StochasticEvolution { fallback_pc, .. } = instruction {
                if catch_scopes.get(pc) != catch_scopes.get(*fallback_pc) {
                    return Err(
                        "stochastic source fallback crosses its original lexical catch scope"
                            .into(),
                    );
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod function_registry_tests {
    use super::{fresh_definition_owner, Bytecode, FunctionBytecode, FunctionRegistry};
    use crate::Instr;
    use runmat_hir::FunctionId;
    use runmat_types::{
        ProgramFunctionId, ProgramPointId, ProgramSourceId, ProgramSpan, RegionContract, RegionId,
        RegionProvenance, REGION_CONTRACT_SCHEMA_VERSION,
    };
    use std::collections::{HashMap, HashSet};

    #[test]
    fn omitted_capture_copies_require_every_compiled_slot_owner() {
        use super::{LexicalBindingKind, LexicalBindingSlot};
        let mut body = FunctionBytecode {
            capture_slots: vec![2, 3],
            lexical_binding_slots: vec![
                LexicalBindingSlot {
                    binding: runmat_types::BindingId(7),
                    slot: 2,
                    kind: LexicalBindingKind::Inherited,
                },
                LexicalBindingSlot {
                    binding: runmat_types::BindingId(8),
                    slot: 3,
                    kind: LexicalBindingKind::Inherited,
                },
            ],
            ..Default::default()
        };
        assert!(body.captures_owned_without_copies(true));
        assert!(!body.captures_owned_without_copies(false));
        body.lexical_binding_slots[1].kind = LexicalBindingKind::Transit;
        assert!(!body.captures_owned_without_copies(true));
        body.lexical_binding_slots.pop();
        assert!(!body.captures_owned_without_copies(true));
    }

    #[test]
    fn catch_scope_metadata_rejects_missing_or_mismatched_artifact_facts() {
        let mut bytecode = Bytecode::with_instructions(
            vec![
                Instr::EnterTry {
                    scope: 7,
                    catch_pc: 1,
                    catch_var: None,
                },
                Instr::Return,
            ],
            0,
        );
        assert!(bytecode.validate_catch_scope_metadata().is_err());
        bytecode.catch_scopes[1] = Some(runmat_mir::CatchScopeId(7));
        assert!(bytecode.validate_catch_scope_metadata().is_ok());
        bytecode.catch_scopes.pop();
        assert!(bytecode.validate_catch_scope_metadata().is_err());
        bytecode
            .catch_scopes
            .push(Some(runmat_mir::CatchScopeId(7)));
        bytecode.instructions.push(Instr::Return);
        bytecode
            .catch_scopes
            .push(Some(runmat_mir::CatchScopeId(8)));
        assert!(bytecode.validate_catch_scope_metadata().is_err());
        bytecode.catch_scopes[2] = None;
        bytecode.instructions[1] = Instr::EnterTry {
            scope: 7,
            catch_pc: 2,
            catch_var: None,
        };
        bytecode.catch_scopes[2] = Some(runmat_mir::CatchScopeId(7));
        assert!(bytecode.validate_catch_scope_metadata().is_err());
    }

    fn test_function(id: usize, display_name: &str, private_owner_scope: &str) -> FunctionBytecode {
        FunctionBytecode {
            spawn_contract: None,
            function: FunctionId(id),
            definition_owner: fresh_definition_owner(),
            display_name: display_name.into(),
            class_method_owner: None,
            private_owner_scope: private_owner_scope.to_string(),
            source_id: None,
            selected_source_owner: None,
            selected_interop_receipt: None,
            selected_class_product: None,
            selected_executable_product: None,
            selected_static_capture: None,
            source_file: None,
            private_source_owner: None,
            source_role: runmat_types::FunctionSourceRole::Unknown,
            lexical_parent: None,
            portable_function: runmat_types::ProgramFunctionId(0),
            portable_lexical_parent: None,
            binding_origin: None,
            capabilities: Default::default(),
            #[cfg(feature = "native-accel")]
            fusion_stage_projection: None,
            #[cfg(feature = "native-accel")]
            fusion_scaffold: None,
            instructions: vec![Instr::Return],
            catch_scopes: vec![None],
            instr_spans: Vec::new(),
            call_arg_spans: Vec::new(),
            coverage_sites: Vec::new(),
            var_count: 0,
            input_slots: Vec::new(),
            varargin_slot: None,
            implicit_nargin_slot: None,
            output_slots: Vec::new(),
            varargout_slot: None,
            implicit_nargout_slot: None,
            capture_slots: Vec::new(),
            lexical_binding_slots: Vec::new(),
            inherited_storage_slots: Vec::new(),
            creates_lexical_environment: false,
            var_names: HashMap::new(),
            initially_unassigned_slots: HashSet::new(),
            argument_validations: Vec::new(),
            resume_points: std::collections::BTreeMap::new(),
            regions: Vec::new(),
            parfor_regions: Vec::new(),
            spmd_regions: Vec::new(),
            distributed_values: Vec::new(),
            collective_contracts: Vec::new(),
        }
    }

    #[test]
    fn nested_replacement_never_inherits_a_textual_name_alias() {
        let id = FunctionId(17);
        let mut original = test_function(id.0, "inner", "");
        original.source_role = runmat_types::FunctionSourceRole::Local;
        let mut registry = FunctionRegistry::new(HashMap::from([(id, original)]));
        assert_eq!(registry.resolve_name("inner"), Some(id));

        let mut nested = test_function(id.0, "inner", "");
        nested.source_role = runmat_types::FunctionSourceRole::Nested;
        nested.lexical_parent = Some(FunctionId(1));
        registry.insert_replacing_name(nested);
        assert_eq!(registry.resolve_name("inner"), None);
        assert!(!registry.names.contains_key("inner"));

        let mut encoded = serde_json::to_value(&registry).expect("serialize registry");
        encoded["names"]["inner"] = serde_json::json!(id.0);
        let decoded: FunctionRegistry =
            serde_json::from_value(encoded).expect("deserialize registry");
        assert_eq!(decoded.resolve_name("inner"), None);
        assert!(!decoded.names.contains_key("inner"));
    }

    #[test]
    fn named_replacement_keeps_published_aliases() {
        for role in [
            runmat_types::FunctionSourceRole::Primary,
            runmat_types::FunctionSourceRole::Local,
            runmat_types::FunctionSourceRole::Unknown,
        ] {
            let id = FunctionId(17);
            let mut original = test_function(id.0, "original", "");
            original.source_role = role;
            let mut registry = FunctionRegistry::new(HashMap::from([(id, original)]));
            registry.names.insert("published_alias".into(), id);

            let mut replacement = test_function(id.0, "replacement", "");
            replacement.source_role = role;
            registry.insert_replacing_name(replacement);

            assert_eq!(registry.resolve_name("published_alias"), Some(id));
            assert_eq!(registry.resolve_name("replacement"), Some(id));
        }
    }

    #[test]
    fn session_rebinding_preserves_immutable_function_origin() {
        let origin = runmat_execution::FunctionBindingOrigin {
            program: runmat_execution::Digest::sha256("original program"),
            function: runmat_types::ProgramFunctionId(3),
        };
        let mut function = test_function(3, "f", "");
        function.portable_function = origin.function;
        function.binding_origin = Some(origin);
        function.rebind_identity(FunctionId(77));
        assert_eq!(function.function, FunctionId(77));
        assert_eq!(
            function.portable_function,
            runmat_types::ProgramFunctionId(3)
        );
        assert_eq!(function.binding_origin, Some(origin));
    }

    #[test]
    fn function_registry_resolves_private_name_in_owner_scope() {
        let mut functions = HashMap::new();
        functions.insert(FunctionId(1), test_function(1, "helper", ""));
        functions.insert(FunctionId(2), test_function(2, "C.__private__.helper", "C"));
        let registry = FunctionRegistry::new(functions);

        assert_eq!(
            registry.resolve_name("helper"),
            Some(FunctionId(1)),
            "unscoped lookup should keep ordinary name resolution"
        );
        assert_eq!(
            registry.resolve_name_in_private_scope("C", "helper"),
            Some(FunctionId(2)),
            "class owner scope should prefer its synthetic private helper"
        );
        assert_eq!(
            registry.resolve_name_in_private_scope("", "helper"),
            None,
            "empty owner scope should not expose synthetic private helpers"
        );
        assert_eq!(
            registry.resolve_name_in_private_scope("C", "pkg.helper"),
            None,
            "qualified names should not be rewritten as private-folder aliases"
        );
    }

    #[test]
    fn execution_stack_requirement_propagates_through_static_calls_and_cycles() {
        let mut entry = test_function(1, "entry", "");
        entry.instructions = vec![Instr::CallSemanticFunctionMulti(FunctionId(2), 0, 1)];
        let mut helper = test_function(2, "helper", "");
        helper.instructions = vec![
            Instr::CallSemanticFunctionMulti(FunctionId(1), 0, 1),
            Instr::CallBuiltinMulti("javaObject".into(), 1, 1),
        ];
        let registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), entry),
            (FunctionId(2), helper),
        ]));

        assert_eq!(
            registry.execution_stack(FunctionId(1)),
            runmat_types::ExecutionStackRequirement::Process
        );
        assert_eq!(
            registry.execution_stack(FunctionId(2)),
            runmat_types::ExecutionStackRequirement::Process
        );
    }

    #[test]
    fn execution_stack_requirement_keeps_pure_functions_segmentable() {
        let registry = FunctionRegistry::new(HashMap::from([(
            FunctionId(1),
            test_function(1, "pure_helper", ""),
        )]));

        assert_eq!(
            registry.execution_stack(FunctionId(1)),
            runmat_types::ExecutionStackRequirement::Any
        );
    }

    #[test]
    fn execution_stack_requirement_is_conservative_at_dynamic_boundaries() {
        let mut callback = test_function(1, "callback", "");
        callback.instructions = vec![Instr::CallFevalMulti(1, 1)];
        let registry = FunctionRegistry::new(HashMap::from([(FunctionId(1), callback)]));

        assert_eq!(
            registry.execution_stack(FunctionId(1)),
            runmat_types::ExecutionStackRequirement::Process
        );
    }

    #[test]
    fn execution_stack_requirement_follows_registered_operator_overloads() {
        let mut caller = test_function(1, "caller", "");
        caller.instructions = vec![
            Instr::RegisterClass {
                name: "Example".into(),
                super_class: None,
                is_sealed: false,
                is_abstract: false,
                properties: Vec::new(),
                methods: vec![crate::bytecode::instr::BytecodeClassMethod {
                    name: "plus".into(),
                    function_name: "Example.plus".into(),
                    role: runmat_types::ClassMethodRole::Ordinary,
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: runmat_types::MemberAccess::Public,
                }],
                enumerations: Vec::new(),
                events: Vec::new(),
            },
            Instr::Add,
        ];
        let mut operator = test_function(2, "Example.plus", "");
        operator.instructions = vec![Instr::CallBuiltinMulti("javaObject".into(), 1, 1)];
        let registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), caller),
            (FunctionId(2), operator),
        ]));

        assert_eq!(
            registry.execution_stack(FunctionId(1)),
            runmat_types::ExecutionStackRequirement::Process
        );
    }

    fn property_dispatch_registry(property_body: Vec<Instr>) -> (FunctionRegistry, FunctionId) {
        let mut caller = test_function(1, "caller", "");
        caller.instructions = vec![
            Instr::RegisterClass {
                name: "Example".into(),
                super_class: None,
                is_sealed: false,
                is_abstract: false,
                properties: vec![crate::bytecode::instr::BytecodeClassProperty {
                    name: "data".into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    default_literal: None,
                    get_access: runmat_types::MemberAccess::Public,
                    set_access: runmat_types::MemberAccess::Public,
                }],
                methods: vec![crate::bytecode::instr::BytecodeClassMethod {
                    name: "get.data".into(),
                    function_name: "Example.get.data".into(),
                    role: runmat_types::ClassMethodRole::Ordinary,
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: runmat_types::MemberAccess::Public,
                }],
                enumerations: Vec::new(),
                events: Vec::new(),
            },
            Instr::CallMethodOrMemberIndexMulti {
                identity: runmat_types::CallableIdentity::DynamicName(runmat_types::SymbolName(
                    "data".into(),
                )),
                fallback_policy: runmat_types::CallableFallbackPolicy::ObjectDispatch,
                arg_count: 1,
                out_count: 1,
            },
        ];
        let mut getter = test_function(2, "Example.get.data", "");
        getter.instructions = property_body;
        (
            FunctionRegistry::new(HashMap::from([
                (FunctionId(1), caller),
                (FunctionId(2), getter),
            ])),
            FunctionId(1),
        )
    }

    #[test]
    fn execution_stack_requirement_resolves_registered_property_dispatch() {
        let (registry, caller) = property_dispatch_registry(vec![Instr::Return]);

        assert_eq!(
            registry.execution_stack(caller),
            runmat_types::ExecutionStackRequirement::Any
        );
    }

    #[test]
    fn execution_stack_requirement_follows_registered_property_accessors() {
        let (registry, caller) =
            property_dispatch_registry(vec![Instr::CallBuiltinMulti("javaObject".into(), 1, 1)]);

        assert_eq!(
            registry.execution_stack(caller),
            runmat_types::ExecutionStackRequirement::Process
        );
    }

    #[test]
    fn execution_stack_requirement_keeps_unresolved_member_dispatch_conservative() {
        let mut caller = test_function(1, "caller", "");
        caller.instructions = vec![Instr::CallMethodOrMemberIndexMulti {
            identity: runmat_types::CallableIdentity::DynamicName(runmat_types::SymbolName(
                "unknown_member".into(),
            )),
            fallback_policy: runmat_types::CallableFallbackPolicy::ObjectDispatch,
            arg_count: 1,
            out_count: 1,
        }];
        let registry = FunctionRegistry::new(HashMap::from([(FunctionId(1), caller)]));

        assert_eq!(
            registry.execution_stack(FunctionId(1)),
            runmat_types::ExecutionStackRequirement::Process
        );
    }

    #[test]
    fn execution_stack_cache_is_invalidated_when_functions_change() {
        let mut registry = FunctionRegistry::new(HashMap::from([(
            FunctionId(1),
            test_function(1, "replaceable", ""),
        )]));
        assert_eq!(
            registry.execution_stack(FunctionId(1)),
            runmat_types::ExecutionStackRequirement::Any
        );

        let mut replacement = test_function(1, "replaceable", "");
        replacement.instructions = vec![Instr::CallBuiltinMulti("javaObject".into(), 1, 1)];
        registry.insert_replacing_name(replacement);
        assert_eq!(
            registry.execution_stack(FunctionId(1)),
            runmat_types::ExecutionStackRequirement::Process
        );
    }

    fn region_contract() -> RegionContract {
        let function = ProgramFunctionId(7);
        RegionContract {
            schema_version: REGION_CONTRACT_SCHEMA_VERSION,
            id: RegionId {
                function,
                ordinal: 2,
            },
            source: ProgramSourceId(1),
            span: ProgramSpan { start: 10, end: 20 },
            entry: ProgramPointId {
                function,
                block: 3,
                position: 1,
            },
            exits: vec![ProgramPointId {
                function,
                block: 3,
                position: 4,
            }],
            live_in: Vec::new(),
            live_out: Vec::new(),
            value_facts: Vec::new(),
            effects: Default::default(),
            capabilities: Default::default(),
            guards: Vec::new(),
            provenance: RegionProvenance::Inferred,
        }
    }

    #[test]
    fn bytecode_installs_exact_region_boundaries_for_every_function_authority() {
        let contract = region_contract();
        let mut function = test_function(7, "region_owner", "");
        function.resume_points.insert(contract.entry, 5);
        function.resume_points.insert(contract.exits[0], 11);
        let mut bytecode = Bytecode::empty();
        bytecode
            .function_registry
            .functions
            .insert(FunctionId(7), function.clone());
        bytecode
            .bound_functions
            .insert(FunctionId(7), function.clone());

        bytecode
            .install_regions(
                std::slice::from_ref(&contract),
                &std::collections::BTreeMap::from([(
                    FunctionId(7),
                    runmat_types::ProgramFunctionId(7),
                )]),
            )
            .unwrap();

        assert_eq!(bytecode.regions.len(), 1);
        assert_eq!(bytecode.regions[0].entry.pc, 5);
        assert_eq!(bytecode.regions[0].exits[0].pc, 11);
        assert_eq!(
            bytecode
                .function_registry
                .functions
                .get(&FunctionId(7))
                .unwrap()
                .regions,
            bytecode.regions
        );
        assert_eq!(
            bytecode
                .bound_functions
                .get(&FunctionId(7))
                .unwrap()
                .regions,
            bytecode.regions
        );
    }

    #[test]
    fn installing_new_unit_regions_preserves_foreign_original_metadata() {
        let contract = region_contract();
        let mut function = test_function(7, "original_owner", "");
        function.resume_points.insert(contract.entry, 5);
        function.resume_points.insert(contract.exits[0], 11);
        let mut bytecode = Bytecode::empty();
        bytecode
            .function_registry
            .functions
            .insert(FunctionId(7), function.clone());
        bytecode.bound_functions.insert(FunctionId(7), function);
        bytecode
            .install_regions(
                std::slice::from_ref(&contract),
                &std::collections::BTreeMap::from([(
                    FunctionId(7),
                    runmat_types::ProgramFunctionId(7),
                )]),
            )
            .unwrap();
        let original = bytecode.function_registry.functions[&FunctionId(7)]
            .regions
            .clone();
        assert_eq!(original.len(), 1);
        bytecode.install_regions(&[], &Default::default()).unwrap();
        assert!(bytecode.regions.is_empty());
        assert_eq!(
            bytecode.function_registry.functions[&FunctionId(7)].regions,
            original
        );
        assert_eq!(bytecode.bound_functions[&FunctionId(7)].regions, original);
    }

    #[test]
    fn bytecode_omits_regions_without_an_exact_resume_boundary() {
        let contract = region_contract();
        let mut function = test_function(7, "region_owner", "");
        function.resume_points.insert(contract.entry, 5);
        function.resume_points.insert(contract.exits[0], 11);
        let mut bytecode = Bytecode::empty();
        bytecode
            .function_registry
            .functions
            .insert(FunctionId(7), function);
        bytecode
            .install_regions(
                std::slice::from_ref(&contract),
                &std::collections::BTreeMap::from([(
                    FunctionId(7),
                    runmat_types::ProgramFunctionId(7),
                )]),
            )
            .unwrap();
        bytecode
            .function_registry
            .functions
            .get_mut(&FunctionId(7))
            .unwrap()
            .resume_points
            .remove(&contract.exits[0]);

        bytecode
            .install_regions(
                std::slice::from_ref(&contract),
                &std::collections::BTreeMap::from([(
                    FunctionId(7),
                    runmat_types::ProgramFunctionId(7),
                )]),
            )
            .unwrap();

        assert!(bytecode.regions.is_empty());
        assert!(bytecode
            .function_registry
            .functions
            .get(&FunctionId(7))
            .unwrap()
            .regions
            .is_empty());
    }

    #[test]
    fn region_installation_preserves_functions_outside_the_compiled_assembly() {
        let contract = region_contract();
        let mut retained = test_function(7, "retained_region_owner", "");
        retained.resume_points.insert(contract.entry, 5);
        retained.resume_points.insert(contract.exits[0], 11);
        retained.regions.push(crate::BytecodeRegion {
            id: contract.id,
            entry: crate::BytecodeRegionBoundary {
                point: contract.entry,
                pc: 5,
            },
            exits: vec![crate::BytecodeRegionBoundary {
                point: contract.exits[0],
                pc: 11,
            }],
        });
        let expected = retained.regions.clone();
        let mut bytecode = Bytecode::empty();
        bytecode.function_registry =
            FunctionRegistry::new(HashMap::from([(FunctionId(7), retained.clone())]));
        bytecode.bound_functions.insert(FunctionId(7), retained);

        bytecode
            .install_regions(
                &[],
                &std::collections::BTreeMap::from([(
                    FunctionId(8),
                    runmat_types::ProgramFunctionId(8),
                )]),
            )
            .unwrap();

        assert_eq!(
            bytecode.function_registry.functions[&FunctionId(7)].regions,
            expected
        );
        assert_eq!(bytecode.bound_functions[&FunctionId(7)].regions, expected);

        let error = bytecode
            .install_regions(
                std::slice::from_ref(&contract),
                &std::collections::BTreeMap::from([(
                    FunctionId(8),
                    runmat_types::ProgramFunctionId(8),
                )]),
            )
            .expect_err("foreign region ownership must fail before mutation");
        assert!(error.contains("outside the compiled assembly"));
        assert_eq!(
            bytecode.function_registry.functions[&FunctionId(7)].regions,
            expected
        );
    }
}

#[cfg(all(test, feature = "native-accel"))]
mod tests {
    use super::{Bytecode, FusionInstructionKind, FusionInstructionWindow};
    use runmat_accelerate::graph::InstrSpan;
    use runmat_accelerate::graph::{AccelNodeLabel, PrimitiveOp};

    #[test]
    fn runtime_fusion_groups_fallback_to_semantic_windows_when_bytecode_groups_are_empty() {
        let mut bytecode = Bytecode::empty();
        bytecode.fusion_metadata.mir_fusion_candidate_group_count = 1;
        bytecode.fusion_metadata.instruction_windows = vec![FusionInstructionWindow {
            span: InstrSpan { start: 2, end: 4 },
            kind: FusionInstructionKind::Elementwise,
        }];

        let groups = bytecode.runtime_fusion_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].span.start, 2);
        assert_eq!(groups[0].span.end, 4);
        assert!(groups[0].nodes.is_empty());
        assert_eq!(
            groups[0].kind,
            runmat_accelerate::fusion::FusionKind::ElementwiseChain
        );
    }

    #[test]
    fn runtime_fusion_groups_use_semantic_windows_when_metadata_is_present() {
        let mut bytecode = Bytecode::empty();
        bytecode.fusion_groups = vec![runmat_accelerate::fusion::FusionGroup {
            id: 7,
            kind: runmat_accelerate::fusion::FusionKind::ElementwiseChain,
            nodes: vec![1],
            shape: runmat_accelerate::graph::ShapeInfo::Unknown,
            span: InstrSpan { start: 5, end: 5 },
            pattern: None,
            stack_layout: None,
        }];
        bytecode.fusion_metadata.mir_fusion_candidate_group_count = 1;
        bytecode.fusion_metadata.instruction_windows = vec![FusionInstructionWindow {
            span: InstrSpan { start: 10, end: 20 },
            kind: FusionInstructionKind::Elementwise,
        }];

        let groups = bytecode.runtime_fusion_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].id, 0);
        assert!(groups[0].nodes.is_empty());
        assert_eq!(groups[0].span.start, 10);
        assert_eq!(groups[0].span.end, 20);
    }

    #[test]
    fn runtime_fusion_groups_ignore_stale_compile_groups_when_semantic_candidates_are_empty() {
        let mut bytecode = Bytecode::empty();
        bytecode.fusion_groups = vec![runmat_accelerate::fusion::FusionGroup {
            id: 7,
            kind: runmat_accelerate::fusion::FusionKind::ElementwiseChain,
            nodes: vec![1],
            shape: runmat_accelerate::graph::ShapeInfo::Unknown,
            span: InstrSpan { start: 5, end: 5 },
            pattern: None,
            stack_layout: None,
        }];
        bytecode.fusion_metadata.mir_fusion_signal_count = 2;
        bytecode.fusion_metadata.mir_fusion_candidate_group_count = 0;

        let groups = bytecode.runtime_fusion_groups();
        assert!(
            groups.is_empty(),
            "semantic metadata should gate runtime fusion groups when no candidates exist"
        );
    }

    #[test]
    fn runtime_fusion_groups_fallback_to_existing_bytecode_groups_without_semantic_metadata() {
        let mut bytecode = Bytecode::empty();
        bytecode.fusion_groups = vec![runmat_accelerate::fusion::FusionGroup {
            id: 7,
            kind: runmat_accelerate::fusion::FusionKind::ElementwiseChain,
            nodes: vec![1],
            shape: runmat_accelerate::graph::ShapeInfo::Unknown,
            span: InstrSpan { start: 5, end: 5 },
            pattern: None,
            stack_layout: None,
        }];

        let groups = bytecode.runtime_fusion_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].id, 7);
        assert_eq!(groups[0].nodes, vec![1]);
        assert_eq!(groups[0].span.start, 5);
        assert_eq!(groups[0].span.end, 5);
    }

    #[test]
    fn runtime_accel_graph_materializes_when_semantic_groups_exist_and_compile_graph_is_missing() {
        let mut bytecode = Bytecode::empty();
        bytecode.instructions = vec![crate::Instr::Add];
        bytecode.var_types = vec![
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
        ];
        bytecode.fusion_metadata.mir_fusion_candidate_group_count = 1;
        bytecode.fusion_metadata.instruction_windows = vec![FusionInstructionWindow {
            span: InstrSpan { start: 0, end: 0 },
            kind: FusionInstructionKind::Elementwise,
        }];

        let runtime_groups = bytecode.runtime_fusion_groups();
        let (graph, source) = bytecode.runtime_accel_graph_for_fusion_with_source(&runtime_groups);
        assert!(
            graph.is_some(),
            "runtime graph should be materialized when semantic runtime groups exist and compile graph is missing"
        );
        assert_eq!(
            source,
            super::RuntimeAccelGraphSource::RuntimeMaterializedFromInstructions
        );
    }

    #[test]
    fn runtime_accel_graph_materializes_when_semantic_groups_exist_and_compile_graph_is_present() {
        let mut bytecode = Bytecode::empty();
        bytecode.instructions = vec![
            crate::Instr::LoadVar(0),
            crate::Instr::LoadVar(1),
            crate::Instr::Add,
        ];
        bytecode.var_types = vec![
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
        ];
        bytecode.accel_graph = Some(crate::accel::graph::build_accel_graph(
            &bytecode.instructions,
            &bytecode.var_types,
        ));
        bytecode.fusion_metadata.mir_fusion_candidate_group_count = 1;
        bytecode.fusion_metadata.instruction_windows = vec![FusionInstructionWindow {
            span: InstrSpan { start: 2, end: 2 },
            kind: FusionInstructionKind::Elementwise,
        }];

        let runtime_groups = bytecode.runtime_fusion_groups();
        let (graph, source) = bytecode.runtime_accel_graph_for_fusion_with_source(&runtime_groups);
        assert!(
            graph.is_some(),
            "runtime graph should still be materialized when compile graph metadata is present"
        );
        assert_eq!(
            source,
            super::RuntimeAccelGraphSource::RuntimeMaterializedFromInstructions
        );
    }

    #[test]
    fn runtime_accel_graph_ignores_stale_compile_graph_metadata() {
        let mut bytecode = Bytecode::empty();
        bytecode.instructions = vec![
            crate::Instr::LoadVar(0),
            crate::Instr::LoadVar(1),
            crate::Instr::Add,
        ];
        bytecode.var_types = vec![
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
        ];

        let stale_graph = crate::accel::graph::build_accel_graph(
            &[
                crate::Instr::LoadVar(0),
                crate::Instr::LoadVar(1),
                crate::Instr::Mul,
            ],
            &bytecode.var_types,
        );
        bytecode.accel_graph = Some(stale_graph);
        bytecode.fusion_metadata.mir_fusion_candidate_group_count = 1;
        bytecode.fusion_metadata.instruction_windows = vec![FusionInstructionWindow {
            span: InstrSpan { start: 2, end: 2 },
            kind: FusionInstructionKind::Elementwise,
        }];

        let runtime_groups = bytecode.runtime_fusion_groups();
        let (graph, source) = bytecode.runtime_accel_graph_for_fusion_with_source(&runtime_groups);
        let graph =
            graph.expect("runtime graph should be materialized from active bytecode instructions");
        assert!(
            graph
                .nodes
                .iter()
                .any(|node| matches!(node.label, AccelNodeLabel::Primitive(PrimitiveOp::Add))),
            "runtime graph should reflect active bytecode instructions"
        );
        assert!(
            !graph
                .nodes
                .iter()
                .any(|node| matches!(node.label, AccelNodeLabel::Primitive(PrimitiveOp::Mul))),
            "stale compile graph metadata should not be reused at runtime"
        );
        assert_eq!(
            source,
            super::RuntimeAccelGraphSource::RuntimeMaterializedFromInstructions
        );
    }

    #[test]
    fn runtime_accel_graph_is_not_materialized_when_runtime_groups_are_empty() {
        let bytecode = Bytecode::empty();
        let (graph, source) = bytecode.runtime_accel_graph_for_fusion_with_source(&[]);
        assert!(
            graph.is_none(),
            "runtime graph materialization should remain gated when semantic runtime groups are absent"
        );
        assert_eq!(source, super::RuntimeAccelGraphSource::NotMaterialized);
    }
}
