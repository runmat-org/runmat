use super::*;

impl RunMatSession {
    pub fn clear_variables(&mut self) {
        self.variables.clear();
        self.variable_array.clear();
        self.variable_names.clear();
        self.workspace_values.clear();
        self.workspace_preview_tokens.clear();
    }

    pub async fn export_workspace_state(
        &mut self,
        mode: WorkspaceExportMode,
    ) -> Result<Option<Vec<u8>>> {
        if matches!(mode, WorkspaceExportMode::Off) {
            return Ok(None);
        }

        let source_map = if self.workspace_values.is_empty() {
            &self.variables
        } else {
            &self.workspace_values
        };

        let mut entries: Vec<(String, Value)> = Vec::with_capacity(source_map.len());
        for (name, value) in source_map {
            let gathered = gather_if_needed_async(value).await?;
            entries.push((name.clone(), gathered));
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
            .map_err(Into::into)
    }

    pub fn import_workspace_state(&mut self, bytes: &[u8]) -> Result<()> {
        let entries = runtime_import_workspace_state(bytes)?;
        self.clear_variables();

        for (index, (name, value)) in entries.into_iter().enumerate() {
            self.variable_names.insert(name.clone(), index);
            self.variable_array.push(value.clone());
            self.variables.insert(name.clone(), value.clone());
            self.workspace_values.insert(name, value);
        }

        self.workspace_preview_tokens.clear();
        self.workspace_version = self.workspace_version.wrapping_add(1);
        Ok(())
    }

    pub fn workspace_snapshot(&mut self) -> WorkspaceSnapshot {
        let source_map = if self.workspace_values.is_empty() {
            &self.variables
        } else {
            &self.workspace_values
        };

        let mut entries: Vec<WorkspaceEntry> = source_map
            .iter()
            .map(|(name, value)| workspace_entry(name, value))
            .collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));

        WorkspaceSnapshot {
            full: true,
            version: self.workspace_version,
            values: self.attach_workspace_preview_tokens(entries),
        }
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
            .workspace_values
            .get(&name)
            .or_else(|| self.variables.get(&name))
            .cloned()
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
                    preview = Some(WorkspacePreview { values, truncated });
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

    /// Get a copy of current variables
    pub fn get_variables(&self) -> &HashMap<String, Value> {
        &self.variables
    }

    pub(crate) fn build_workspace_snapshot(
        &mut self,
        entries: Vec<WorkspaceEntry>,
        full: bool,
    ) -> WorkspaceSnapshot {
        self.workspace_version = self.workspace_version.wrapping_add(1);
        let version = self.workspace_version;
        WorkspaceSnapshot {
            full,
            version,
            values: self.attach_workspace_preview_tokens(entries),
        }
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
