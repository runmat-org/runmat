use std::collections::{BTreeMap, BTreeSet};

use runmat_native_codegen::{NativeBlockId, NativeFunction, NativeTerminatorKind, NativeValueId};
use runmat_runtime::native::{NativeRoot, NativeRootKind, NativeRootSet, NativeValueRef};
use runmat_runtime::{context::RuntimeContext, RuntimeError};

use crate::memory::ValueArena;
use crate::{JitError, JitResult};

pub(super) struct ActiveForLoop {
    pub iterator: runmat_runtime::iteration::ForColumnIterator,
    body_blocks: BTreeSet<NativeBlockId>,
}

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
    pub pending_place_mutation: Option<runmat_mir::MirPlaceMutation>,
    global_bindings: BTreeMap<usize, String>,
    persistent_bindings: BTreeMap<usize, String>,
    active_for_loops: BTreeMap<NativeBlockId, ActiveForLoop>,
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
        let mut locals = vec![NativeValueRef::NULL; function.local_count()];
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
            active_for_loops: BTreeMap::new(),
            pending_place_mutation: None,
            global_bindings: BTreeMap::new(),
            persistent_bindings: BTreeMap::new(),
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

    pub fn set_local(&mut self, slot: usize, value: NativeValueRef) -> JitResult<()> {
        let local = self
            .locals
            .get_mut(slot)
            .ok_or_else(|| JitError::Host("native local is out of bounds".into()))?;
        *local = value;
        self.synchronize_session_binding(slot, value)
    }

    pub fn declare_global(&mut self, slot: usize, name: String) -> JitResult<()> {
        if self.persistent_bindings.contains_key(&slot) {
            return Err(JitError::Host(
                "native local cannot be both global and persistent".into(),
            ));
        }
        self.global_bindings.insert(slot, name.clone());
        let value = runmat_runtime::workspace::session::global_value(&name)
            .unwrap_or_else(empty_workspace_value);
        let reference = self.arena.insert(value);
        self.set_local(slot, reference)
    }

    pub fn declare_persistent(&mut self, slot: usize, name: String) -> JitResult<()> {
        if self.global_bindings.contains_key(&slot) {
            return Err(JitError::Host(
                "native local cannot be both global and persistent".into(),
            ));
        }
        self.persistent_bindings.insert(slot, name.clone());
        let value =
            runmat_runtime::workspace::session::persistent_named_value(&self.function.name, &name)
                .unwrap_or_else(empty_workspace_value);
        let reference = self.arena.insert(value);
        self.set_local(slot, reference)
    }

    fn synchronize_session_binding(
        &mut self,
        slot: usize,
        reference: NativeValueRef,
    ) -> JitResult<()> {
        if reference.is_null() {
            return Ok(());
        }
        let value = self.arena.get(reference)?.clone();
        if let Some(name) = self.global_bindings.get(&slot) {
            runmat_runtime::workspace::session::store_global_named(name, value.clone());
        }
        if let Some(name) = self.persistent_bindings.get(&slot) {
            runmat_runtime::workspace::session::store_persistent_named(
                &self.function.name,
                name,
                value,
            );
        }
        Ok(())
    }

    pub fn has_for_loop(&self, header: NativeBlockId) -> bool {
        self.active_for_loops.contains_key(&header)
    }

    pub fn start_for_loop(
        &mut self,
        header: NativeBlockId,
        body: NativeBlockId,
        iterator: runmat_runtime::iteration::ForColumnIterator,
    ) {
        let body_blocks = self.blocks_reaching_header(header, body);
        self.active_for_loops.insert(
            header,
            ActiveForLoop {
                iterator,
                body_blocks,
            },
        );
    }

    pub fn for_loop_mut(&mut self, header: NativeBlockId) -> JitResult<&mut ActiveForLoop> {
        self.active_for_loops
            .get_mut(&header)
            .ok_or_else(|| JitError::Host("native for-loop state is unavailable".into()))
    }

    /// Retire loop snapshots when control leaves their natural loop body.
    /// This covers `break` and exceptional structured exits as well as normal
    /// exhaustion, while retaining state across body backedges and `continue`.
    pub fn take_control_edge(&mut self, target: NativeBlockId) {
        self.active_for_loops
            .retain(|header, active| target == *header || active.body_blocks.contains(&target));
    }

    fn blocks_reaching_header(
        &self,
        header: NativeBlockId,
        body: NativeBlockId,
    ) -> BTreeSet<NativeBlockId> {
        let mut reverse = BTreeMap::<NativeBlockId, Vec<NativeBlockId>>::new();
        for block in &self.function.blocks {
            for target in successor_blocks(&block.terminator.kind) {
                reverse.entry(target).or_default().push(block.id);
            }
        }
        let mut reaches_header = BTreeSet::from([header]);
        let mut pending = vec![header];
        while let Some(target) = pending.pop() {
            for predecessor in reverse.get(&target).into_iter().flatten() {
                if reaches_header.insert(*predecessor) {
                    pending.push(*predecessor);
                }
            }
        }
        let mut reachable_from_body = BTreeSet::new();
        let mut pending = vec![body];
        while let Some(block) = pending.pop() {
            if block == header
                || !reaches_header.contains(&block)
                || !reachable_from_body.insert(block)
            {
                continue;
            }
            if let Some(block) = self
                .function
                .blocks
                .iter()
                .find(|candidate| candidate.id == block)
            {
                pending.extend(successor_blocks(&block.terminator.kind));
            }
        }
        reachable_from_body
    }
}

fn empty_workspace_value() -> runmat_value::Value {
    runmat_value::Value::Tensor(
        runmat_value::Tensor::new(Vec::new(), vec![0, 0])
            .expect("the canonical empty workspace shape is valid"),
    )
}

fn successor_blocks(kind: &NativeTerminatorKind) -> Vec<NativeBlockId> {
    match kind {
        NativeTerminatorKind::Goto { edge } => vec![edge.target],
        NativeTerminatorKind::Branch {
            then_edge,
            else_edge,
            ..
        } => vec![then_edge.target, else_edge.target],
        NativeTerminatorKind::Switch {
            cases, otherwise, ..
        } => cases
            .iter()
            .map(|(_, edge)| edge.target)
            .chain(std::iter::once(otherwise.target))
            .collect(),
        NativeTerminatorKind::For { body, exit, .. }
        | NativeTerminatorKind::ParFor { body, exit, .. }
        | NativeTerminatorKind::Spmd { body, exit, .. } => vec![body.target, exit.target],
        NativeTerminatorKind::TryCatch {
            try_edge,
            catch_edge,
            ..
        } => vec![try_edge.target, catch_edge.target],
        NativeTerminatorKind::Await { resume, .. } => vec![resume.target],
        NativeTerminatorKind::Return { .. } | NativeTerminatorKind::Unreachable => Vec::new(),
    }
}
