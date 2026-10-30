//! Canonical executable-definition fingerprint shared by publication,
//! invalidation, and retained artifact admission.

use std::collections::{BTreeMap, BTreeSet};

#[cfg(feature = "native-accel")]
#[derive(serde::Serialize)]
struct SelectedFunctionLayoutIdentity<'a> {
    function: &'a runmat_hir::FunctionId,
    parent: &'a Option<runmat_hir::FunctionId>,
    display_name: &'a str,
    source_role: &'a runmat_types::FunctionSourceRole,
    private_owner_scope: &'a str,
    frame_abi: &'a crate::layout::VmFrameAbi,
    binding_slots: Vec<(&'a runmat_hir::BindingId, &'a crate::layout::VmSlotId)>,
    mir_local_slots: Vec<(&'a runmat_mir::MirLocalId, &'a crate::layout::VmSlotId)>,
    captures: &'a [crate::layout::VmCaptureSlot],
    local_count: usize,
    resume_points: Vec<(&'a runmat_types::ProgramPointId, &'a usize)>,
}

#[cfg(feature = "native-accel")]
impl<'a> SelectedFunctionLayoutIdentity<'a> {
    fn from_layout(layout: &'a crate::layout::VmFunctionLayout) -> Self {
        let mut binding_slots = layout.binding_slots.iter().collect::<Vec<_>>();
        binding_slots.sort_unstable_by_key(|(id, _)| **id);
        let mut mir_local_slots = layout.mir_local_slots.iter().collect::<Vec<_>>();
        mir_local_slots.sort_unstable_by_key(|(id, _)| **id);
        Self {
            function: &layout.function,
            parent: &layout.parent,
            display_name: &layout.display_name,
            source_role: &layout.source_role,
            private_owner_scope: &layout.private_owner_scope,
            frame_abi: &layout.frame_abi,
            binding_slots,
            mir_local_slots,
            captures: &layout.captures,
            local_count: layout.local_count,
            resume_points: layout.resume_points.iter().collect(),
        }
    }
}

#[cfg(feature = "native-accel")]
pub(crate) fn fusion_stage_source_fingerprint(
    projection: &crate::bytecode::FusionStageProjection,
    stream: &crate::bytecode::fusion_stage::FusionStageStream<'_>,
) -> Result<String, serde_json::Error> {
    let crate::bytecode::FusionStageProjection {
        revision,
        portable_origin,
        selection_origin,
        owner,
        mir_local_count,
        statements,
        private_temporary_slots,
        selected_function_layout,
        source_fingerprint: _,
    } = projection;
    let mut payload = Vec::new();
    macro_rules! append {
        ($value:expr) => {{
            let bytes = serde_json::to_vec($value)?;
            payload.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            payload.extend_from_slice(&bytes);
        }};
    }
    append!("selected-mir-fusion-stage-stream-v1");
    append!(revision);
    append!(&portable_origin);
    append!(&owner);
    append!(&mir_local_count);
    append!(statements);
    append!(stream.instructions);
    append!(stream.catch_scopes);
    append!(&stream.resume_points.iter().collect::<Vec<_>>());
    append!(&stream.var_count);
    append!(&semantic_source_revision(stream.source_id));
    // Entry streams retain their historical payload. Function layout maps
    // cannot use HashMap iteration order as executable source identity.
    if let Some(layout) = selected_function_layout {
        append!("selected-mir-function-layout-v1");
        append!(&SelectedFunctionLayoutIdentity::from_layout(layout));
    }
    // Absence retains the exact pre-origin fingerprint for legacy artifacts;
    // only a compiler-minted origin can authorize new image proposals.
    if let Some(origin) = selection_origin {
        append!("selected-mir-fusion-stage-origin-v1");
        append!(&origin);
    }
    if !private_temporary_slots.is_empty() {
        append!("selected-mir-private-temporary-slots-v1");
        append!(private_temporary_slots);
    }
    Ok(runmat_execution::Digest::sha256(&payload).to_string())
}

#[derive(serde::Serialize)]
pub(super) enum SemanticSourceRevision {
    Program(runmat_types::ProgramSourceId),
    Runtime(runmat_execution::Digest),
}

pub(super) fn semantic_source_revision(
    source: Option<super::BytecodeSourceRef>,
) -> Option<SemanticSourceRevision> {
    source.map(|source| match source {
        super::BytecodeSourceRef::Program(id) => SemanticSourceRevision::Program(id),
        super::BytecodeSourceRef::Runtime(reference) => {
            SemanticSourceRevision::Runtime(reference.semantic_content_revision())
        }
    })
}

pub fn semantic_definition_fingerprint(
    function: &super::FunctionBytecode,
) -> Result<String, serde_json::Error> {
    semantic_definition_fingerprint_with_source(function, function.source_id)
}

pub(super) fn semantic_definition_fingerprint_with_source(
    function: &super::FunctionBytecode,
    source: Option<super::BytecodeSourceRef>,
) -> Result<String, serde_json::Error> {
    // Spell out every semantic field so a new FunctionBytecode field cannot
    // enter publication or import without an explicit identity decision.
    let super::FunctionBytecode {
        function,
        // Admission renews this nonwire evidence from the same canonical MIR.
        // Its local presence is not part of a portable semantic identity.
        spawn_contract: _,
        definition_owner: _,
        display_name,
        class_method_owner,
        private_owner_scope,
        source_id: _,
        selected_source_owner: _,
        // Checked foreign archive ownership is not bytecode semantics.
        selected_interop_receipt: _,
        // The port-free selected product owns issuance, not body semantics.
        selected_executable_product: _,
        selected_static_capture: _,
        // Local selected dependency ownership is not source definition identity.
        selected_class_product: _,
        // Display provenance follows the authenticated source-map owner.
        source_file: _,
        private_source_owner,
        source_role,
        lexical_parent: _,
        portable_function: _,
        portable_lexical_parent,
        binding_origin: _,
        capabilities,
        #[cfg(feature = "native-accel")]
        fusion_stage_projection,
        #[cfg(feature = "native-accel")]
        fusion_scaffold,
        instructions,
        catch_scopes,
        instr_spans,
        call_arg_spans,
        coverage_sites,
        var_count,
        input_slots,
        varargin_slot,
        implicit_nargin_slot,
        output_slots,
        varargout_slot,
        implicit_nargout_slot,
        capture_slots,
        lexical_binding_slots,
        inherited_storage_slots,
        creates_lexical_environment,
        var_names,
        initially_unassigned_slots,
        argument_validations,
        resume_points,
        regions,
        parfor_regions,
        spmd_regions,
        distributed_values,
        collective_contracts,
    } = function;
    let var_names = var_names.iter().collect::<BTreeMap<_, _>>();
    let initially_unassigned_slots = initially_unassigned_slots.iter().collect::<BTreeSet<_>>();
    let resume_points = resume_points.iter().collect::<Vec<_>>();
    let mut payload = Vec::new();
    macro_rules! append {
        ($value:expr) => {{
            let bytes = serde_json::to_vec($value)?;
            payload.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            payload.extend_from_slice(&bytes);
        }};
    }
    append!(function);
    append!(display_name);
    append!(class_method_owner);
    append!(private_owner_scope);
    append!(&semantic_source_revision(source));
    if let Some(owner) = private_source_owner {
        // Only the path-free equivalence partition affects semantic reuse.
        // The native host path is intentionally excluded from this identity.
        append!("private-source-aliases-v1");
        append!(owner.aliases());
    }
    append!(source_role);
    append!(portable_lexical_parent);
    append!(capabilities);
    #[cfg(feature = "native-accel")]
    append!(&fusion_stage_projection
        .as_ref()
        .map(|projection| projection.portable_definition_identity()));
    // Legacy definitions without a scaffold retain their prior fingerprint.
    // New function-local authority is sealed together with its owner IDs.
    #[cfg(feature = "native-accel")]
    if let Some(scaffold) = fusion_scaffold {
        append!(scaffold);
    }
    append!(instructions);
    append!(catch_scopes);
    append!(instr_spans);
    append!(call_arg_spans);
    append!(coverage_sites);
    append!(var_count);
    append!(input_slots);
    append!(varargin_slot);
    append!(implicit_nargin_slot);
    append!(output_slots);
    append!(varargout_slot);
    append!(implicit_nargout_slot);
    append!(capture_slots);
    append!(lexical_binding_slots);
    // Session owner IDs are remapped during publication. The paired portable
    // owner and exact storage/name facts define this semantic dependency.
    append!(&inherited_storage_slots
        .iter()
        .map(|entry| (
            entry.binding,
            entry.slot,
            entry.portable_owner,
            &entry.name,
            &entry.owner_function,
            entry.kind,
        ))
        .collect::<Vec<_>>());
    append!(creates_lexical_environment);
    append!(&var_names);
    append!(&initially_unassigned_slots);
    append!(argument_validations);
    append!(&resume_points);
    append!(regions);
    append!(parfor_regions);
    append!(spmd_regions);
    append!(distributed_values);
    append!(collective_contracts);
    Ok(runmat_execution::Digest::sha256(&payload).to_string())
}

/// Compare the semantic contents of a selected local execution product without
/// attempting to serialize its session-issued source references. Issuance and
/// process-local definition owners must be checked separately by the caller.
pub(crate) fn local_bytecode_semantic_fingerprint(
    bytecode: &super::Bytecode,
) -> Result<String, serde_json::Error> {
    let super::Bytecode {
        instructions,
        stack_geometry,
        catch_scopes,
        active_function: _,
        active_class_method_owner: _,
        instr_spans,
        call_arg_spans,
        coverage_sites,
        resume_points,
        source_id,
        var_count,
        bound_functions,
        function_registry,
        var_types,
        var_names,
        initially_unassigned_slots,
        // Compared with typed PartialEq at the admitted projection boundary;
        // its nested HashMaps have no canonical JSON iteration order.
        layout: _,
        #[cfg(feature = "native-accel")]
            function_private_layout: _,
        // Derived process-local acceleration is not source identity.
        #[cfg(feature = "native-accel")]
            sealed_function_image: _,
        async_metadata,
        regions,
        parfor_regions,
        spmd_regions,
        distributed_values,
        collective_contracts,
        #[cfg(feature = "native-accel")]
        accel_graph,
        #[cfg(feature = "native-accel")]
        fusion_groups,
        #[cfg(feature = "native-accel")]
        fusion_metadata,
    } = bytecode;
    let function_facts =
        |functions: &std::collections::HashMap<runmat_hir::FunctionId, super::FunctionBytecode>| {
            let mut entries = functions.iter().collect::<Vec<_>>();
            entries.sort_by_key(|(id, _)| id.0);
            entries
                .into_iter()
                .map(|(id, body)| {
                    Ok((
                        *id,
                        semantic_definition_fingerprint(body)?,
                        body.lexical_parent,
                        body.source_file.clone(),
                        body.binding_origin,
                        body.portable_function,
                    ))
                })
                .collect::<Result<Vec<_>, serde_json::Error>>()
        };
    let mut names = function_registry.names.iter().collect::<Vec<_>>();
    names.sort_by(|left, right| left.0.cmp(right.0));
    let var_names = var_names.iter().collect::<BTreeMap<_, _>>();
    let unassigned = initially_unassigned_slots.iter().collect::<BTreeSet<_>>();
    let mut payload = Vec::new();
    macro_rules! append {
        ($value:expr) => {{
            let bytes = serde_json::to_vec($value)?;
            payload.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            payload.extend_from_slice(&bytes);
        }};
    }
    append!("selected-local-bytecode-semantics-v1");
    append!(instructions);
    append!(stack_geometry);
    append!(catch_scopes);
    append!(instr_spans);
    append!(call_arg_spans);
    append!(coverage_sites);
    append!(&resume_points.iter().collect::<Vec<_>>());
    append!(&semantic_source_revision(*source_id));
    append!(var_count);
    append!(&function_facts(bound_functions)?);
    // Published functions and names are execution semantics. The registry's
    // source index, retired bodies, owner index, and static overlay are local
    // ownership/cache state: both projections clone the same inherited
    // registry and merge only the selected current eval definitions.
    append!(&function_facts(&function_registry.functions)?);
    append!(&names);
    append!(var_types);
    append!(&var_names);
    append!(&unassigned);
    append!(async_metadata);
    append!(regions);
    append!(parfor_regions);
    append!(spmd_regions);
    append!(distributed_values);
    append!(collective_contracts);
    #[cfg(feature = "native-accel")]
    {
        // Graph node/value order is semantic; binding-map iteration order is
        // not. Sort only the two map projections before framing their bytes.
        let graph_facts = accel_graph.as_ref().map(|graph| {
            (
                &graph.nodes,
                &graph.values,
                graph.var_bindings.iter().collect::<BTreeMap<_, _>>(),
                graph.node_bindings.iter().collect::<BTreeMap<_, _>>(),
            )
        });
        append!(&graph_facts);
        append!(fusion_groups);
        append!(fusion_metadata);
    }
    Ok(runmat_execution::Digest::sha256(&payload).to_string())
}

#[cfg(all(test, feature = "native-accel"))]
#[path = "semantic_fingerprint/tests.rs"]
mod tests;
