use crate::{MirLocal, MirLocalId, MirLocalKind, MirSequenceLocalId};
use runmat_hir::{BindingId, FunctionId, HirError, HirFunction, Span};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub(crate) struct MirLoweringContext {
    binding_locals: HashMap<BindingId, MirLocalId>,
    async_functions: HashSet<FunctionId>,
    next_local: usize,
    temp_locals: RefCell<Vec<MirLocal>>,
    function: runmat_types::ProgramFunctionId,
    active_spmd_regions: RefCell<Vec<runmat_types::ParallelRegionId>>,
    next_distributed_value: Cell<u32>,
    next_collective: Cell<u32>,
    next_sequence_local: Cell<usize>,
}

impl Default for MirLoweringContext {
    fn default() -> Self {
        Self {
            binding_locals: HashMap::new(),
            async_functions: HashSet::new(),
            next_local: 0,
            temp_locals: RefCell::new(Vec::new()),
            function: runmat_types::ProgramFunctionId(0),
            active_spmd_regions: RefCell::new(Vec::new()),
            next_distributed_value: Cell::new(0),
            next_collective: Cell::new(0),
            next_sequence_local: Cell::new(0),
        }
    }
}

impl MirLoweringContext {
    pub(crate) fn with_async_functions(
        async_functions: HashSet<FunctionId>,
        function: FunctionId,
    ) -> Self {
        Self {
            async_functions,
            function: runmat_types::ProgramFunctionId(
                u32::try_from(function.0).expect("HIR function IDs fit portable identities"),
            ),
            ..Self::default()
        }
    }

    pub(crate) fn with_spmd_region<T>(
        &self,
        region: runmat_types::ParallelRegionId,
        operation: impl FnOnce() -> Result<T, HirError>,
    ) -> Result<T, HirError> {
        self.active_spmd_regions.borrow_mut().push(region);
        let result = operation();
        let popped = self.active_spmd_regions.borrow_mut().pop();
        debug_assert_eq!(popped, Some(region));
        result
    }

    pub(crate) fn distributed_identity(
        &self,
    ) -> (
        runmat_types::DistributedValueId,
        runmat_types::DistributedOwner,
    ) {
        let ordinal = self.next_distributed_value.get();
        self.next_distributed_value.set(ordinal + 1);
        let owner = self
            .active_spmd_regions
            .borrow()
            .last()
            .copied()
            .map(runmat_types::DistributedOwner::Region)
            .unwrap_or(runmat_types::DistributedOwner::Client(self.function));
        (
            runmat_types::DistributedValueId {
                function: self.function,
                ordinal,
            },
            owner,
        )
    }

    pub(crate) fn collective_identity(&self) -> Result<runmat_types::CollectiveId, HirError> {
        let region = self
            .active_spmd_regions
            .borrow()
            .last()
            .copied()
            .ok_or_else(|| {
                HirError::new("collective operation requires an enclosing SPMD region")
            })?;
        let ordinal = self.next_collective.get();
        self.next_collective.set(ordinal + 1);
        Ok(runmat_types::CollectiveId { region, ordinal })
    }

    pub(crate) fn in_spmd_region(&self) -> bool {
        !self.active_spmd_regions.borrow().is_empty()
    }

    pub(crate) fn is_async_function(&self, function: FunctionId) -> bool {
        self.async_functions.contains(&function)
    }

    pub(crate) fn local_for_binding(&self, binding: BindingId) -> Result<MirLocalId, HirError> {
        self.binding_locals
            .get(&binding)
            .copied()
            .ok_or_else(|| HirError::new("binding has no MIR local"))
    }

    pub(crate) fn locals_for_function(&mut self, function: &HirFunction) -> Vec<MirLocal> {
        let mut locals: Vec<_> = function
            .locals
            .iter()
            .enumerate()
            .map(|(idx, binding)| {
                let local = MirLocalId(idx);
                self.binding_locals.insert(*binding, local);
                let kind = if function.params.contains(binding) {
                    MirLocalKind::Parameter
                } else if function.outputs.contains(binding) {
                    MirLocalKind::Output
                } else {
                    MirLocalKind::Binding
                };
                MirLocal {
                    id: local,
                    binding: Some(*binding),
                    kind,
                    span: function.span,
                }
            })
            .collect();

        for capture in &function.captures {
            if self.binding_locals.contains_key(&capture.binding) {
                continue;
            }
            let local = MirLocalId(locals.len());
            self.binding_locals.insert(capture.binding, local);
            locals.push(MirLocal {
                id: local,
                binding: Some(capture.binding),
                kind: MirLocalKind::Capture,
                span: function.span,
            });
        }

        self.next_local = locals.len();
        locals
    }

    pub(crate) fn fresh_temp(&self, span: Span) -> MirLocalId {
        let local = MirLocalId(self.next_local + self.temp_locals.borrow().len());
        self.temp_locals.borrow_mut().push(MirLocal {
            id: local,
            binding: None,
            kind: MirLocalKind::Temporary,
            span,
        });
        local
    }

    pub(crate) fn fresh_sequence_local(&self) -> MirSequenceLocalId {
        let next = self.next_sequence_local.get();
        self.next_sequence_local.set(next + 1);
        MirSequenceLocalId(next)
    }

    pub(crate) fn take_temp_locals(&self) -> Vec<MirLocal> {
        std::mem::take(&mut *self.temp_locals.borrow_mut())
    }
}
