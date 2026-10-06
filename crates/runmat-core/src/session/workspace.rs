use super::*;

#[cfg(not(target_arch = "wasm32"))]
fn workspace_runtime_error(error: RuntimeError) -> anyhow::Error {
    anyhow::Error::new(error)
}

#[cfg(target_arch = "wasm32")]
fn workspace_runtime_error(error: RuntimeError) -> anyhow::Error {
    // Browser-local error sources need not be Send + Sync, while anyhow's
    // source chain does. Keep the diagnostic at this public workspace boundary.
    anyhow::anyhow!(error.to_string())
}

impl RunMatSession {
    pub(crate) fn base_workspace_snapshot(
        &self,
    ) -> std::result::Result<runmat_runtime::workspace::BaseWorkspaceSnapshot, RuntimeError> {
        self.base_workspace
            .snapshot()
            .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime)
    }

    pub fn workspace_handle(&self) -> crate::abi::WorkspaceHandle {
        self.abi_workspace_handle
    }

    pub(crate) fn workspace_binding_key(&self, name: &str) -> crate::abi::WorkspaceBindingKey {
        let binding = runmat_hir::BindingName(name.to_string());
        if let Some(source) = self.active_source_identity.clone() {
            crate::abi::WorkspaceBindingKey::SourceBinding {
                source,
                def_path: source_binding_def_path(self.current_source_name()),
                binding,
            }
        } else {
            crate::abi::WorkspaceBindingKey::Interactive {
                session: self.abi_workspace_handle.0,
                name: binding,
            }
        }
    }

    pub(crate) fn bind_workspace_slot(&mut self, name: String, slot: usize) {
        let key = self.workspace_binding_key(&name);
        self.workspace_bindings
            .insert(name, SessionWorkspaceBinding { key, slot });
    }

    /// Refresh compiler/VM slot metadata from a completed invocation without
    /// treating the register file as a second base-workspace authority.
    pub(crate) fn refresh_workspace_binding_slots(
        &mut self,
        completed: &HashMap<String, (usize, Value)>,
    ) {
        self.workspace_bindings.clear();
        for (name, (slot, _)) in completed {
            self.bind_workspace_slot(name.clone(), *slot);
        }
    }

    pub(crate) fn lowering_workspace_bindings(&self) -> HashMap<String, usize> {
        self.workspace_bindings
            .iter()
            .map(|(name, binding)| (name.clone(), binding.slot))
            .collect()
    }

    pub fn clear_variables(&mut self) -> std::result::Result<(), RuntimeError> {
        let mut delta = runmat_runtime::workspace::WorkspaceDelta::new();
        delta.clear();
        self.base_workspace
            .apply(delta)
            .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime)?;
        self.workspace_bindings.clear();
        self.workspace_preview_tokens.clear();
        Ok(())
    }

    pub async fn export_workspace_state(
        &mut self,
        mode: WorkspaceExportMode,
    ) -> Result<Option<Vec<u8>>> {
        if matches!(mode, WorkspaceExportMode::Off) {
            return Ok(None);
        }

        let snapshot = self
            .base_workspace_snapshot()
            .map_err(workspace_runtime_error)?;
        let mut entries = Vec::with_capacity(snapshot.len());
        for (name, value) in snapshot.iter() {
            let gathered = gather_if_needed_async(value)
                .await
                .map_err(workspace_runtime_error)?;
            entries.push((name.to_string(), gathered));
        }

        if entries.is_empty() && matches!(mode, WorkspaceExportMode::Auto) {
            return Ok(None);
        }

        let replay_mode = match mode {
            WorkspaceExportMode::Auto => WorkspaceReplayMode::Auto,
            WorkspaceExportMode::Force => WorkspaceReplayMode::Force,
            WorkspaceExportMode::Off => WorkspaceReplayMode::Off,
        };

        runtime_export_workspace_state(&entries, replay_mode)
            .await
            .map_err(workspace_runtime_error)
    }

    pub fn import_workspace_state(&mut self, bytes: &[u8]) -> Result<()> {
        let entries = runtime_import_workspace_state(bytes).map_err(workspace_runtime_error)?;
        for (_, value) in &entries {
            runmat_runtime::execution::validate_storable_value(value)
                .map_err(workspace_runtime_error)?;
        }
        let mut delta = runmat_runtime::workspace::WorkspaceDelta::new();
        delta.clear();
        for (name, value) in &entries {
            delta
                .assign(name.clone(), value.clone())
                .map_err(base_workspace_error)?;
        }
        self.base_workspace
            .apply(delta)
            .map_err(base_workspace_error)?;

        self.workspace_bindings.clear();
        for (index, (name, _)) in entries.into_iter().enumerate() {
            self.bind_workspace_slot(name.clone(), index);
        }

        self.workspace_preview_tokens.clear();
        Ok(())
    }

    pub fn workspace_snapshot(&mut self) -> std::result::Result<WorkspaceSnapshot, RuntimeError> {
        let snapshot = self.base_workspace_snapshot()?;
        let mut entries: Vec<WorkspaceEntry> = snapshot
            .iter()
            .map(|(name, value)| workspace_entry(name, value))
            .collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(WorkspaceSnapshot {
            full: true,
            version: snapshot.revision().get(),
            values: self.attach_workspace_preview_tokens(entries),
        })
    }

    /// Materialize a workspace variable for inspection (optionally identified by preview token).
    pub async fn materialize_variable(
        &mut self,
        target: WorkspaceMaterializeTarget,
        options: WorkspaceMaterializeOptions,
    ) -> Result<MaterializedVariable> {
        let name = match target {
            WorkspaceMaterializeTarget::Name(name) => name,
            WorkspaceMaterializeTarget::Token(id) => self
                .workspace_preview_tokens
                .get(&id)
                .map(|ticket| ticket.name.clone())
                .ok_or_else(|| anyhow::anyhow!("Unknown workspace preview token"))?,
        };
        let value = self
            .base_workspace
            .lookup(&name)
            .map_err(base_workspace_error)?
            .ok_or_else(|| anyhow::anyhow!("Variable '{name}' not found in workspace"))?;

        let is_gpu = matches!(value, Value::GpuTensor(_));
        let residency = if is_gpu {
            WorkspaceResidency::Gpu
        } else {
            WorkspaceResidency::Cpu
        };
        // For CPU values we can materialize directly. For GPU tensors, avoid downloading the
        // entire buffer into wasm memory; gather only the requested preview/slice.
        let host_value = value.clone();
        let value_shape_vec = value_shape(&host_value).unwrap_or_default();
        let mut preview = None;
        if is_gpu {
            if let Value::GpuTensor(handle) = &value {
                if let Some((values, truncated)) =
                    gather_gpu_preview_values(handle, &value_shape_vec, &options).await?
                {
                    preview = Some(WorkspacePreview {
                        values: values
                            .into_iter()
                            .map(crate::NumericPreviewValue::Float)
                            .collect(),
                        truncated,
                    });
                }
            }
        } else {
            if let Some(slice_opts) = options
                .slice
                .as_ref()
                .and_then(|slice| slice.sanitized(&value_shape_vec))
            {
                let slice_elements = slice_opts.shape.iter().product::<usize>();
                let slice_limit = slice_elements.clamp(1, MATERIALIZE_DEFAULT_LIMIT);
                if let Some(slice_value) = slice_value_for_preview(&host_value, &slice_opts) {
                    preview = preview_numeric_values(&slice_value, slice_limit)
                        .map(|(values, truncated)| WorkspacePreview { values, truncated });
                }
            }
            if preview.is_none() {
                let max_elements = options.max_elements.clamp(1, MATERIALIZE_DEFAULT_LIMIT);
                preview = preview_numeric_values(&host_value, max_elements)
                    .map(|(values, truncated)| WorkspacePreview { values, truncated });
            }
        }
        Ok(MaterializedVariable {
            name,
            class_name: matlab_class_name(&host_value),
            dtype: if let Value::GpuTensor(handle) = &host_value {
                gpu_dtype_label(handle).map(|label| label.to_string())
            } else {
                numeric_dtype_label(&host_value).map(|label| label.to_string())
            },
            shape: value_shape_vec,
            is_gpu,
            residency,
            size_bytes: if let Value::GpuTensor(handle) = &host_value {
                gpu_size_bytes(handle)
            } else {
                approximate_size_bytes(&host_value)
            },
            preview,
            value: host_value,
        })
    }

    /// Get the current persistent workspace values keyed by source name.
    pub fn get_variables(&self) -> std::result::Result<HashMap<String, Value>, RuntimeError> {
        let snapshot = self.base_workspace_snapshot()?;
        Ok(snapshot
            .iter()
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect())
    }

    pub(crate) fn build_workspace_snapshot(
        &mut self,
        entries: Vec<WorkspaceEntry>,
        full: bool,
    ) -> std::result::Result<WorkspaceSnapshot, RuntimeError> {
        let version = self
            .base_workspace
            .revision()
            .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime)?
            .get();
        Ok(WorkspaceSnapshot {
            full,
            version,
            values: self.attach_workspace_preview_tokens(entries),
        })
    }

    fn attach_workspace_preview_tokens(
        &mut self,
        entries: Vec<WorkspaceEntry>,
    ) -> Vec<WorkspaceEntry> {
        self.workspace_preview_tokens.clear();
        let mut values = Vec::with_capacity(entries.len());
        for mut entry in entries {
            let token = Uuid::new_v4();
            self.workspace_preview_tokens.insert(
                token,
                WorkspaceMaterializeTicket {
                    name: entry.name.clone(),
                },
            );
            entry.preview_token = Some(token);
            values.push(entry);
        }
        values
    }
}

fn base_workspace_error(error: runmat_runtime::workspace::BaseWorkspaceError) -> anyhow::Error {
    workspace_runtime_error(runmat_runtime::workspace::base_workspace_error_to_runtime(
        error,
    ))
}

fn source_binding_def_path(source_name: &str) -> runmat_hir::DefPath {
    use std::path::Path;

    let stem = Path::new(source_name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.trim().is_empty())
        .unwrap_or("entrypoint")
        .to_string();

    runmat_hir::DefPath {
        package: runmat_hir::PackageName("workspace".to_string()),
        module: runmat_hir::QualifiedName(vec![runmat_hir::SymbolName(stem.clone())]),
        item: vec![runmat_hir::DefPathSegment::Function(
            runmat_hir::SymbolName(stem),
        )],
    }
}
