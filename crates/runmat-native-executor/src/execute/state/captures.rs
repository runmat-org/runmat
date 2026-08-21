use super::*;

impl HostState {
    pub fn lexical_captures(
        &self,
        function: runmat_types::ProgramFunctionId,
    ) -> NativeExecutorResult<Option<Vec<runmat_runtime::call::lexical::LexicalCapture>>> {
        let target = self
            .functions
            .iter()
            .find(|candidate| candidate.id == function);
        let Some(target) = target else {
            return Ok(None);
        };
        let mut captures = Vec::new();
        for target_local in target
            .locals
            .iter()
            .filter(|local| local.kind == NativeLocalKind::Capture)
        {
            let binding = target_local.binding.ok_or_else(|| {
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
            let value = self
                .locals
                .get(source.id.0 as usize)
                .copied()
                .ok_or_else(|| {
                    NativeExecutorError::Host("native capture local is out of bounds".into())
                })?;
            captures.push(runmat_runtime::call::lexical::LexicalCapture {
                binding,
                value: self.arena.get(value)?.clone(),
            });
        }
        Ok((!captures.is_empty()).then_some(captures))
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
            let local = self
                .function
                .locals
                .iter()
                .find(|local| local.binding == Some(capture.binding))
                .ok_or_else(|| {
                    NativeExecutorError::Host("returned lexical binding is not visible".into())
                })?;
            self.locals[local.id.0 as usize] = self.arena.insert(capture.value);
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
