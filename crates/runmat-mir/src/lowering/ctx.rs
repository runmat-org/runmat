use crate::{MirLocal, MirLocalId, MirLocalKind, MirLocalSource};
use runmat_hir::{BindingId, ExprId, HirFunction, SemanticError, Span};
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct MirLoweringContext {
    pub(crate) binding_locals: HashMap<BindingId, MirLocalId>,
    next_local: usize,
    temp_locals: RefCell<Vec<MirLocal>>,
    temp_sources: RefCell<Vec<MirLocalSource>>,
}

impl MirLoweringContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn local_for_binding(
        &self,
        binding: BindingId,
    ) -> Result<MirLocalId, SemanticError> {
        self.binding_locals
            .get(&binding)
            .copied()
            .ok_or_else(|| SemanticError::new("binding has no MIR local"))
    }

    pub(crate) fn locals_for_function(
        &mut self,
        function: &HirFunction,
    ) -> (Vec<MirLocal>, Vec<MirLocalSource>) {
        self.next_local = function.locals.len();
        function
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
                (
                    MirLocal {
                        id: local,
                        binding: Some(*binding),
                        kind,
                        span: function.span,
                    },
                    MirLocalSource {
                        local,
                        binding: Some(*binding),
                        expr: None,
                        span: function.span,
                    },
                )
            })
            .unzip()
    }

    pub(crate) fn fresh_temp(&self, span: Span, expr: Option<ExprId>) -> MirLocalId {
        let local = MirLocalId(self.next_local + self.temp_locals.borrow().len());
        self.temp_locals.borrow_mut().push(MirLocal {
            id: local,
            binding: None,
            kind: MirLocalKind::Temporary,
            span,
        });
        self.temp_sources.borrow_mut().push(MirLocalSource {
            local,
            binding: None,
            expr,
            span,
        });
        local
    }

    pub(crate) fn take_temp_locals(&self) -> (Vec<MirLocal>, Vec<MirLocalSource>) {
        (
            std::mem::take(&mut *self.temp_locals.borrow_mut()),
            std::mem::take(&mut *self.temp_sources.borrow_mut()),
        )
    }
}
