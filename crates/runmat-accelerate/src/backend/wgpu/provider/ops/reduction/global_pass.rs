//! The first pass sees source values; later passes see already reduced partials.
use crate::backend::wgpu::types::GlobalReduceOp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GlobalPassKind {
    SourceMapping,
    PartialCombine,
}

pub(super) struct GlobalReductionPassPolicy {
    source: GlobalReduceOp,
    partial: GlobalReduceOp,
    source_requires_dispatch: bool,
}

impl GlobalReductionPassPolicy {
    pub(super) fn for_operation(operation: GlobalReduceOp) -> Self {
        match operation {
            GlobalReduceOp::CountNonZero => Self {
                source: GlobalReduceOp::CountNonZero,
                partial: GlobalReduceOp::Sum,
                source_requires_dispatch: true,
            },
            GlobalReduceOp::Sum
            | GlobalReduceOp::Prod
            | GlobalReduceOp::Min
            | GlobalReduceOp::Max => Self {
                source: operation,
                partial: operation,
                source_requires_dispatch: false,
            },
        }
    }

    pub(super) fn should_dispatch(&self, kind: GlobalPassKind, len: usize) -> bool {
        len > 1 || (kind == GlobalPassKind::SourceMapping && self.source_requires_dispatch)
    }

    pub(super) fn operation(&self, kind: GlobalPassKind) -> GlobalReduceOp {
        match kind {
            GlobalPassKind::SourceMapping => self.source,
            GlobalPassKind::PartialCombine => self.partial,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn source_mapping_is_once_even_for_singleton_count() {
        for operation in [
            GlobalReduceOp::Sum,
            GlobalReduceOp::Prod,
            GlobalReduceOp::Min,
            GlobalReduceOp::Max,
        ] {
            let policy = GlobalReductionPassPolicy::for_operation(operation);
            assert!(!policy.should_dispatch(GlobalPassKind::SourceMapping, 1));
            assert_eq!(
                policy.operation(GlobalPassKind::SourceMapping) as u32,
                operation as u32
            );
            assert_eq!(
                policy.operation(GlobalPassKind::PartialCombine) as u32,
                operation as u32
            );
        }
        let count = GlobalReductionPassPolicy::for_operation(GlobalReduceOp::CountNonZero);
        assert!(count.should_dispatch(GlobalPassKind::SourceMapping, 1));
        assert!(!count.should_dispatch(GlobalPassKind::PartialCombine, 1));
        assert_eq!(count.operation(GlobalPassKind::SourceMapping) as u32, 4);
        assert_eq!(count.operation(GlobalPassKind::PartialCombine) as u32, 0);
    }
}
