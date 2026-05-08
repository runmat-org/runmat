use super::*;

impl RunMatSession {
    pub(crate) fn compile_input(
        &mut self,
        input: &str,
    ) -> std::result::Result<PreparedExecution, RunError> {
        let source_name = self.current_source_name().to_string();
        let source_id = self.source_pool.intern(&source_name, input);
        let ast = {
            let _span = info_span!("runtime.parse").entered();
            parse_with_options(input, ParserOptions::new(self.compat_mode))?
        };
        let lowering = {
            let _span = info_span!("runtime.lower").entered();
            runmat_hir::lower(
                &ast,
                &LoweringContext::new(
                    &self.legacy_variable_names,
                    &self.legacy_function_definitions,
                ),
            )?
        };
        let mut bytecode = {
            let _span = info_span!("runtime.compile.bytecode").entered();
            self.compile_semantic_bytecode(&lowering)?
        };
        bytecode.source_id = Some(source_id);
        let new_function_names: HashSet<String> = lowering.functions.keys().cloned().collect();
        for (name, func) in bytecode.functions.iter_mut() {
            if new_function_names.contains(name) {
                func.source_id = Some(source_id);
            }
        }
        Ok(PreparedExecution {
            ast,
            lowering,
            bytecode,
        })
    }

    pub(crate) fn populate_callstack(&self, error: &mut RuntimeError) {
        if !error.context.call_stack.is_empty() || error.context.call_frames.is_empty() {
            return;
        }
        let mut rendered = Vec::new();
        if error.context.call_frames_elided > 0 {
            rendered.push(format!(
                "... {} frames elided ...",
                error.context.call_frames_elided
            ));
        }
        for frame in error.context.call_frames.iter().rev() {
            let mut line = frame.function.clone();
            if let (Some(source_id), Some((start, _end))) = (frame.source_id, frame.span) {
                if let Some(source) = self.source_pool.get(SourceId(source_id)) {
                    let (line_num, col) = line_col_from_offset(&source.text, start);
                    line = format!("{} @ {}:{}:{}", frame.function, source.name, line_num, col);
                }
            }
            rendered.push(line);
        }
        error.context.call_stack = rendered;
    }

    fn compile_semantic_bytecode(
        &self,
        lowering: &LoweringResult,
    ) -> std::result::Result<runmat_vm::Bytecode, RunError> {
        let entrypoint = lowering.assembly.entrypoints.first().ok_or_else(|| {
            RunError::Compile(runmat_vm::CompileError::new(
                "semantic bytecode compile requires an entrypoint",
            ))
        })?;
        let mir = runmat_mir::lowering::lower_assembly(&lowering.assembly)?;
        Ok(runmat_vm::compile(&lowering.assembly, &mir, entrypoint.id)?)
    }

    pub(crate) fn normalize_error_namespace(&self, error: &mut RuntimeError) {
        let Some(identifier) = error.identifier.clone() else {
            return;
        };
        let suffix = identifier
            .split_once(':')
            .map(|(_, suffix)| suffix)
            .unwrap_or(identifier.as_str());
        error.identifier = Some(format!("{}:{suffix}", self.error_namespace));
    }

    /// Compile the input and produce a fusion plan snapshot without executing.
    pub fn compile_fusion_plan(
        &mut self,
        input: &str,
    ) -> std::result::Result<Option<FusionPlanSnapshot>, RunError> {
        let prepared = self.compile_input(input)?;
        Ok(build_fusion_snapshot(
            prepared.bytecode.accel_graph.as_ref(),
            &prepared.bytecode.fusion_groups,
        ))
    }

    pub(crate) fn prepare_variable_array_for_execution(
        &mut self,
        bytecode: &runmat_vm::Bytecode,
        updated_var_mapping: &HashMap<String, usize>,
        debug_trace: bool,
    ) {
        // Create a new variable array of the correct size
        let max_var_id = updated_var_mapping.values().copied().max().unwrap_or(0);
        let required_len = std::cmp::max(bytecode.var_count, max_var_id + 1);
        let mut new_variable_array = vec![Value::Num(0.0); required_len];
        if debug_trace {
            debug!(
                bytecode_var_count = bytecode.var_count,
                required_len, max_var_id, "[repl] prepare variable array"
            );
        }

        // Populate with existing values based on the variable mapping
        for (var_name, &new_var_id) in updated_var_mapping {
            if new_var_id < new_variable_array.len() {
                if let Some(value) = self.workspace_values.get(var_name) {
                    if debug_trace {
                        debug!(
                            var_name,
                            var_id = new_var_id,
                            ?value,
                            "[repl] prepare set var"
                        );
                    }
                    new_variable_array[new_var_id] = value.clone();
                }
            } else if debug_trace {
                debug!(
                    var_name,
                    var_id = new_var_id,
                    len = new_variable_array.len(),
                    "[repl] prepare skipping var"
                );
            }
        }

        // Update our variable array and mapping
        self.variable_array = new_variable_array;
    }
}
