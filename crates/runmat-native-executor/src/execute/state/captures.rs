use super::*;

impl HostState {
    pub fn lexical_captures(
        &mut self,
        function: runmat_types::ProgramFunctionId,
    ) -> NativeExecutorResult<Option<Vec<runmat_runtime::call::lexical::LexicalCapture>>> {
        let target = self
            .functions
            .iter()
            .find(|candidate| candidate.id == function);
        let Some(target) = target else {
            return Ok(None);
        };
        let nested = target.source_role == runmat_types::FunctionSourceRole::Nested;
        if nested && !self.nested_function_is_visible(target.lexical_parent) {
            return Err(NativeExecutorError::Host(
                "native nested target is outside the caller's lexical scope".into(),
            ));
        }
        let capture_bindings = target
            .locals
            .iter()
            .filter(|local| local.kind == NativeLocalKind::Capture)
            .map(|local| local.binding)
            .collect::<Vec<_>>();
        let mut captures = Vec::new();
        for target_binding in capture_bindings {
            let binding = target_binding.ok_or_else(|| {
                NativeExecutorError::Host("native capture local has no semantic binding".into())
            })?;
            let source = self
                .function
                .locals
                .iter()
                .find(|local| local.binding == Some(binding))
                .ok_or_else(|| {
                    NativeExecutorError::Host(
                        "nested native capture is not visible in its caller".into(),
                    )
                })?;
            let slot = source.id.0 as usize;
            let value = self.local_reference(slot)?;
            // The binding-keyed activation, not this transport snapshot, owns
            // nested semantics. An unassigned capture may be initialized by
            // the callee before its first read.
            let value = if value.is_null() && self.lexical_environment.is_some() {
                runmat_value::Value::Num(0.0)
            } else {
                self.arena.get(value)?.clone()
            };
            captures.push(runmat_runtime::call::lexical::LexicalCapture { binding, value });
        }
        Ok((nested || !captures.is_empty()).then_some(captures))
    }

    fn nested_function_is_visible(
        &self,
        target_parent: Option<runmat_types::ProgramFunctionId>,
    ) -> bool {
        let Some(target_parent) = target_parent else {
            return false;
        };
        let mut frame = Some(self.function.id);
        for _ in 0..self.functions.len() {
            let Some(current) = frame else {
                return false;
            };
            if current == target_parent {
                return true;
            }
            frame = self
                .program_function(current)
                .and_then(|body| body.lexical_parent);
        }
        false
    }

    pub fn program_function(
        &self,
        function: runmat_types::ProgramFunctionId,
    ) -> Option<&NativeFunction> {
        self.functions
            .iter()
            .find(|candidate| candidate.id == function)
    }

    pub fn apply_lexical_captures(
        &mut self,
        captures: Vec<runmat_runtime::call::lexical::LexicalCapture>,
    ) -> NativeExecutorResult<()> {
        for capture in captures {
            if self.lexical_environment.is_some()
                && self
                    .function
                    .lexical_bindings
                    .iter()
                    .any(|binding| binding.binding == capture.binding)
            {
                // A shared cell owns nested state; an anonymous transit
                // binding is a snapshot here but its ancestor cell is still
                // authoritative for the nested descendant.
                continue;
            }
            let slot = self
                .function
                .locals
                .iter()
                .find(|local| local.binding == Some(capture.binding))
                .ok_or_else(|| {
                    NativeExecutorError::Host("returned lexical binding is not visible".into())
                })?
                .id
                .0 as usize;
            if self.global_bindings.binding_for_slot(slot).is_some()
                || self.persistent_bindings.contains_key(&slot)
            {
                // The callee already published its mutation through the
                // canonical session binding. Replaying a returned capture
                // snapshot could overwrite a newer global/persistent value.
                continue;
            }
            let value = self.arena.insert(capture.value);
            self.set_local(slot, value)?;
        }
        Ok(())
    }

    pub fn capture_results(
        &self,
    ) -> NativeExecutorResult<Vec<runmat_runtime::call::lexical::LexicalCapture>> {
        self.function
            .locals
            .iter()
            .filter(|local| local.kind == NativeLocalKind::Capture)
            .map(|local| {
                let binding = local.binding.ok_or_else(|| {
                    NativeExecutorError::Host("native capture local has no semantic binding".into())
                })?;
                if let Some(environment) = self.lexical_environment.as_ref().filter(|_| {
                    self.function.lexical_bindings.iter().any(|entry| {
                        entry.binding == binding
                            && entry.kind
                                != runmat_native_codegen::NativeLexicalBindingKind::Transit
                    })
                }) {
                    if let Some(value) =
                        runmat_runtime::call::lexical::read_binding(environment, binding)?
                    {
                        return Ok(runmat_runtime::call::lexical::LexicalCapture {
                            binding,
                            value,
                        });
                    }
                }
                let value = self.locals[local.id.0 as usize];
                Ok(runmat_runtime::call::lexical::LexicalCapture {
                    binding,
                    value: self.arena.get(value)?.clone(),
                })
            })
            .collect()
    }

    pub fn enter_block(
        &mut self,
        block: runmat_native_codegen::NativeBlockId,
    ) -> NativeExecutorResult<()> {
        let parameters = self
            .function
            .blocks
            .iter()
            .find(|candidate| candidate.id == block)
            .ok_or_else(|| {
                NativeExecutorError::Host(format!("native block {} is unavailable", block.0))
            })?
            .parameters
            .clone();
        for parameter in parameters {
            let value = self
                .locals
                .get(parameter.local.0 as usize)
                .copied()
                .ok_or_else(|| {
                    NativeExecutorError::Host("block parameter local is out of bounds".into())
                })?;
            self.values.insert(parameter.value, value);
        }
        Ok(())
    }
}
