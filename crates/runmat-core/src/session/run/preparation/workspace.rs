use super::product::PreparedInteractiveProduct;
#[cfg(not(target_arch = "wasm32"))]
use crate::session::run::mir_local_fact_count_for_entrypoint;
use crate::session::run::{
    entry_statement_count, execution_workspace_mapping, first_entry_statement,
};
use crate::session::*;

pub(in crate::session::run) struct PreparedInteractiveWorkspace {
    pub unit: Rc<crate::ExecutableUnit>,
    pub runtime: runmat_runtime::context::RuntimeContext,
    pub bytecode: runmat_vm::Bytecode,
    pub publication: crate::session::source_evaluation::FunctionRegistryPublication,
    pub durable_registry: runmat_vm::FunctionRegistry,
    pub next_semantic_function_id: usize,
    pub display_context: crate::workspace::DisplayContext,
    pub display_var_ids: Vec<usize>,
    pub stmt_count: usize,
    pub execution_vars: HashMap<String, usize>,
    pub frame_extent: usize,
    pub workspace_projection: Rc<runmat_runtime::workspace::BaseWorkspaceProjectionService>,
    pub workspace_admission: Option<runmat_vm::PendingWorkspaceAdmission>,
    pub fusion_snapshot: Option<FusionPlanSnapshot>,
    pub variable_array: Option<runmat_vm::InterpreterVars>,
}

impl RunMatSession {
    #[tracing::instrument(name = "runtime.prepare_workspace", skip_all)]
    pub(in crate::session::run) fn prepare_interactive_workspace(
        &mut self,
        product: PreparedInteractiveProduct,
        preserve_layout_var_names: bool,
        debug_trace: bool,
    ) -> std::result::Result<PreparedInteractiveWorkspace, RunError> {
        let PreparedInteractiveProduct {
            unit: interactive_unit,
            runtime: execution_runtime,
            mut bytecode,
            publication: source_registry_publication,
            durable_registry: function_registry_after_success,
            next_semantic_function_id: next_semantic_function_id_after_success,
            lowering,
        } = product;
        let display = execution_display_context(&lowering.assembly, bytecode.layout.as_ref());
        let display_context = display.context;
        let display_var_ids = display.display_var_ids;
        let stmt_count = entry_statement_count(&lowering.assembly);
        if self.verbose {
            debug!("Semantic entry body len: {stmt_count}");
            if let Some(stmt) = first_entry_statement(&lowering.assembly) {
                debug!("Semantic HIR statement: {stmt:?}");
            }
        }
        let execution_vars = execution_workspace_mapping(&bytecode);
        if debug_trace {
            debug!(?execution_vars, "[repl] execution vars");
        }
        let workspace_projection = self
            .base_workspace
            .begin_projection()
            .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime)
            .map_err(RunError::Runtime)?;
        let workspace_projection = std::rc::Rc::new(
            runmat_runtime::workspace::BaseWorkspaceProjectionService::new_for_context(
                workspace_projection,
                &execution_runtime,
            )
            .with_publication_observer(
                self.workspace_publication_observer_with_context(&execution_runtime)
                    .map_err(RunError::Runtime)?,
            ),
        );
        if debug_trace {
            let workspace_values_before =
                runmat_runtime::context::RuntimeWorkspaceService::try_snapshot_with_context(
                    workspace_projection.as_ref(),
                    &execution_runtime,
                )
                .map_err(RunError::Runtime)?
                .into_iter()
                .collect::<HashMap<_, _>>();
            debug!(
                ?workspace_values_before,
                "[repl] workspace snapshot before execution"
            );
        }
        let id_to_name: HashMap<usize, String> = execution_vars
            .iter()
            .map(|(name, var_id)| (*var_id, name.clone()))
            .collect();
        let global_aliases = runmat_runtime::workspace::bindings::FrameGlobalBindings::default();
        let mut assigned_snapshot = HashSet::new();
        for (name, slot) in &execution_vars {
            if let Some(target) = self
                .base_workspace
                .global_target(name)
                .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime)
                .map_err(RunError::Runtime)?
            {
                let binding = global_aliases
                    .prepare_target(&execution_runtime, *slot, &target)
                    .map_err(|cause| {
                        RunError::Runtime(crate::session::workspace::global_key::admission_error(
                            &execution_runtime,
                            cause,
                        ))
                    })?;
                binding.checkpoint().map_err(|cause| {
                    RunError::Runtime(crate::session::workspace::global_key::admission_error(
                        &execution_runtime,
                        cause,
                    ))
                })?;
                binding.commit();
                if self
                    .runtime_context
                    .session_variable_authority()
                    .contains_global_target(&target)
                    .map_err(RunError::Runtime)?
                {
                    assigned_snapshot.insert(name.clone());
                }
            } else if runmat_runtime::context::RuntimeWorkspaceService::try_lookup_with_context(
                workspace_projection.as_ref(),
                &execution_runtime,
                name,
            )
            .map_err(RunError::Runtime)?
            .is_some()
            {
                assigned_snapshot.insert(name.clone());
            }
        }
        if debug_trace {
            debug!(?assigned_snapshot, "[repl] assigned snapshot");
        }
        let workspace_admission = Some(runmat_vm::workspace_admission_with_global_bindings(
            execution_vars.clone(),
            assigned_snapshot.clone(),
            global_aliases,
        ));
        if self.verbose {
            debug!("HIR generated successfully");
        }

        if preserve_layout_var_names && bytecode.layout.is_some() {
            for (slot, name) in &id_to_name {
                bytecode.var_names.insert(*slot, name.clone());
            }
        } else {
            bytecode.var_names = id_to_name.clone();
        }
        if self.verbose {
            debug!(
                "Bytecode compiled: {} instructions",
                bytecode.instructions.len()
            );
        }

        #[cfg(not(target_arch = "wasm32"))]
        let fusion_snapshot = if self.emit_fusion_plan {
            let runtime_groups = bytecode.runtime_fusion_groups();
            let (runtime_graph, runtime_graph_source) =
                bytecode.runtime_accel_graph_for_fusion_with_source(&runtime_groups);
            build_fusion_snapshot(
                &runtime_groups,
                &bytecode.fusion_metadata.mir_fusion_candidate_groups,
                &bytecode.fusion_metadata.instruction_windows,
                Some(crate::fusion::FusionPlannerMetadata {
                    source: "semantic-mir-analysis-runtime".to_string(),
                    accel_graph_state: if runtime_graph.is_some() {
                        "present".to_string()
                    } else {
                        "missing".to_string()
                    },
                    accel_graph_source: runtime_graph_source.as_str().to_string(),
                    mir_local_fact_count: mir_local_fact_count_for_entrypoint(
                        interactive_unit.analysis(),
                        &lowering.assembly,
                    ),
                    mir_diagnostic_count: interactive_unit.analysis().diagnostics.len(),
                    mir_fusion_signal_count: bytecode.fusion_metadata.mir_fusion_signal_count,
                    mir_fusion_candidate_group_count: bytecode
                        .fusion_metadata
                        .mir_fusion_candidate_group_count,
                    mir_semantic_instruction_window_count: bytecode
                        .fusion_metadata
                        .instruction_window_count,
                }),
            )
        } else {
            None
        };
        #[cfg(target_arch = "wasm32")]
        let fusion_snapshot: Option<FusionPlanSnapshot> = None;

        // Metadata only: the cold VM route acquires its actual frame once
        // selected, before executing. Native results keep their own root.
        let frame_extent = execution_vars
            .values()
            .copied()
            .max()
            .map(|slot| slot.checked_add(1))
            .unwrap_or(Some(0))
            .ok_or_else(|| {
                RunError::Runtime(
                    build_runtime_error("workspace slot extent overflow")
                        .with_identifier("RunMat:WorkspaceBindingSlots")
                        .build(),
                )
            })?
            .max(bytecode.var_count);

        Ok(PreparedInteractiveWorkspace {
            unit: interactive_unit,
            runtime: execution_runtime,
            bytecode,
            publication: source_registry_publication,
            durable_registry: function_registry_after_success,
            next_semantic_function_id: next_semantic_function_id_after_success,
            display_context,
            display_var_ids,
            stmt_count,
            execution_vars,
            frame_extent,
            workspace_projection,
            workspace_admission,
            fusion_snapshot,
            variable_array: None,
        })
    }
}
