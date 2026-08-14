use std::collections::BTreeMap;

use runmat_native_codegen::{NativeFunction, NativeValueId};
use runmat_runtime::native::{NativeRoot, NativeRootKind, NativeRootSet, NativeValueRef};
use runmat_runtime::{context::RuntimeContext, RuntimeError};

use crate::memory::ValueArena;
use crate::{JitError, JitResult};

pub(super) struct HostState {
    pub function: NativeFunction,
    pub runtime: RuntimeContext,
    pub arena: ValueArena,
    pub locals: Vec<NativeValueRef>,
    pub values: BTreeMap<NativeValueId, NativeValueRef>,
    pub roots: Vec<NativeRoot>,
    pub last_error: Option<RuntimeError>,
    pub host_failure: Option<JitError>,
    pub current_source: runmat_runtime::native::NativeSourceLocation,
}

impl HostState {
    pub fn new(
        function: NativeFunction,
        arguments: Vec<runmat_value::Value>,
        runtime: RuntimeContext,
    ) -> JitResult<(Self, Vec<NativeValueRef>)> {
        if function.abi.varargin.is_some() {
            return Err(JitError::UnsupportedSite(
                "varargin entry requires the generic call-shape cohort".into(),
            ));
        }
        if arguments.len() != function.abi.fixed_inputs.len() {
            return Err(JitError::Host(format!(
                "function {} expects {} inputs but received {}",
                function.name,
                function.abi.fixed_inputs.len(),
                arguments.len()
            )));
        }
        let mut arena = ValueArena::default();
        let argument_refs = arguments
            .into_iter()
            .map(|value| arena.insert(value))
            .collect::<Vec<_>>();
        let mut locals = vec![NativeValueRef::NULL; function.local_count as usize];
        for (local, value) in function.abi.fixed_inputs.iter().zip(&argument_refs) {
            let slot = locals.get_mut(local.0 as usize).ok_or_else(|| {
                JitError::Host("function ABI input local is out of bounds".into())
            })?;
            *slot = *value;
        }
        let roots = locals
            .iter()
            .enumerate()
            .map(|(slot, value)| NativeRoot {
                value: *value,
                kind: NativeRootKind::LOCAL,
                slot: slot as u32,
            })
            .collect::<Vec<_>>();
        let source = function.source.0;
        let mut state = Self {
            function,
            runtime,
            arena,
            locals,
            values: BTreeMap::new(),
            roots,
            last_error: None,
            host_failure: None,
            current_source: runmat_runtime::native::NativeSourceLocation {
                source,
                ..runmat_runtime::native::NativeSourceLocation::default()
            },
        };
        state.enter_block(state.function.entry)?;
        Ok((state, argument_refs))
    }

    pub fn enter_block(&mut self, block: runmat_native_codegen::NativeBlockId) -> JitResult<()> {
        let parameters = self
            .function
            .blocks
            .iter()
            .find(|candidate| candidate.id == block)
            .ok_or_else(|| JitError::Host(format!("native block {} is unavailable", block.0)))?
            .parameters
            .clone();
        for parameter in parameters {
            let value = self
                .locals
                .get(parameter.local.0 as usize)
                .copied()
                .ok_or_else(|| JitError::Host("block parameter local is out of bounds".into()))?;
            self.values.insert(parameter.value, value);
        }
        Ok(())
    }

    pub fn refresh_roots(&mut self) -> NativeRootSet {
        for (root, value) in self.roots.iter_mut().zip(&self.locals) {
            root.value = *value;
        }
        NativeRootSet {
            roots: self.roots.as_ptr(),
            count: self.roots.len(),
        }
    }
}
