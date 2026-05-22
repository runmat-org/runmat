use crate::{MirLocal, MirLocalId, MirLocalKind};
use runmat_hir::{BindingId, FunctionId, HirError, HirFunction, Span};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default)]
pub(crate) struct MirLoweringContext {
    binding_locals: HashMap<BindingId, MirLocalId>,
    async_functions: HashSet<FunctionId>,
    next_local: usize,
    temp_locals: RefCell<Vec<MirLocal>>,
}

impl MirLoweringContext {
    pub(crate) fn with_async_functions(async_functions: HashSet<FunctionId>) -> Self {
        Self {
            async_functions,
            ..Self::default()
        }
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

    pub(crate) fn take_temp_locals(&self) -> Vec<MirLocal> {
        std::mem::take(&mut *self.temp_locals.borrow_mut())
    }
}
