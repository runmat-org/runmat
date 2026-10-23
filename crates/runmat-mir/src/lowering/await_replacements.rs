use std::collections::HashMap;

use runmat_hir::ExprId;

use crate::{MirOperand, MirSequenceLocalId};

/// Values already evaluated by the canonical Await prefix lowering. A CSL
/// receipt is not a scalar operand: only argument/selector expansion may
/// consume it. This is compile-time source substitution, not a runtime table.
#[derive(Clone, Default)]
pub(crate) struct AwaitReplacements {
    operands: HashMap<ExprId, MirOperand>,
    sequences: HashMap<ExprId, MirSequenceLocalId>,
}

impl AwaitReplacements {
    pub fn get(&self, expression: &ExprId) -> Option<&MirOperand> {
        self.operands.get(expression)
    }
    pub fn insert(&mut self, expression: ExprId, operand: MirOperand) -> Option<MirOperand> {
        self.sequences.remove(&expression);
        self.operands.insert(expression, operand)
    }
    pub fn contains_key(&self, expression: &ExprId) -> bool {
        self.operands.contains_key(expression) || self.sequences.contains_key(expression)
    }

    pub fn sequence(&self, expression: ExprId) -> Option<MirSequenceLocalId> {
        self.sequences.get(&expression).copied()
    }

    pub fn insert_sequence(&mut self, expression: ExprId, sequence: MirSequenceLocalId) {
        self.operands.remove(&expression);
        self.sequences.insert(expression, sequence);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacing_receipt_kind_never_leaves_a_second_authority() {
        let expression = ExprId(3);
        let mut replacements = AwaitReplacements::default();
        let operand = MirOperand::Local(crate::MirLocalId(4));
        replacements.insert(expression, operand.clone());
        assert_eq!(replacements.get(&expression), Some(&operand));
        replacements.insert_sequence(expression, MirSequenceLocalId(5));
        assert!(replacements.get(&expression).is_none());
        assert_eq!(
            replacements.sequence(expression),
            Some(MirSequenceLocalId(5))
        );
        assert!(replacements.contains_key(&expression));
        replacements.insert(expression, operand.clone());
        assert!(replacements.sequence(expression).is_none());
        assert_eq!(replacements.get(&expression), Some(&operand));
    }
}
