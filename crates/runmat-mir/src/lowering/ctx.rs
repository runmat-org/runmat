use crate::{MirLocal, MirLocalId, MirLocalKind, MirLocalSource};
use runmat_hir::{BindingId, HirFunction};
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct MirLoweringContext {
    pub(crate) binding_locals: HashMap<BindingId, MirLocalId>,
}

impl MirLoweringContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn locals_for_function(
        &mut self,
        function: &HirFunction,
    ) -> (Vec<MirLocal>, Vec<MirLocalSource>) {
        function
            .locals
            .iter()
            .enumerate()
            .map(|(idx, binding)| {
                let local = MirLocalId(idx);
                self.binding_locals.insert(*binding, local);
                (
                    MirLocal {
                        id: local,
                        binding: Some(*binding),
                        kind: MirLocalKind::Binding,
                        span: function.span,
                    },
                    MirLocalSource {
                        local,
                        binding: Some(*binding),
                        span: function.span,
                    },
                )
            })
            .unzip()
    }
}
