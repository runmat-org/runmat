use super::{MirExpressionRegion, MirExpressionStep};
use crate::{MirOperand, MirRvalue, MirSequenceLocalId};
use std::collections::BTreeSet;

impl MirExpressionRegion {
    pub fn visit_operands(&self, mut visitor: impl FnMut(&MirOperand)) {
        self.visit_operands_dyn(&mut visitor);
    }

    /// Visits only values supplied by the owning MIR body. Locals defined by
    /// this region are private implementation details and must not become
    /// parent-site inputs or liveness roots.
    pub fn visit_external_operands(&self, mut visitor: impl FnMut(&MirOperand)) {
        let mut defined = self.defined_locals().collect::<BTreeSet<_>>();
        self.visit_nested_regions(&mut |region| defined.extend(region.defined_locals()));
        self.visit_operands(|operand| {
            if !matches!(operand, MirOperand::Local(local) if defined.contains(local)) {
                visitor(operand);
            }
        });
    }

    pub(crate) fn visit_operands_dyn(&self, visitor: &mut dyn FnMut(&MirOperand)) {
        for step in &self.steps {
            match step {
                MirExpressionStep::Let { value, .. } => value.visit_operands_dyn(visitor),
                MirExpressionStep::CaptureSequence { source, .. } => {
                    source.visit_operands(&mut *visitor)
                }
            }
        }
        visitor(&self.result);
    }

    pub fn visit_operands_mut(&mut self, mut visitor: impl FnMut(&mut MirOperand)) {
        self.visit_operands_mut_dyn(&mut visitor);
    }

    pub(crate) fn visit_operands_mut_dyn(&mut self, visitor: &mut dyn FnMut(&mut MirOperand)) {
        for step in &mut self.steps {
            match step {
                MirExpressionStep::Let { value, .. } => value.visit_operands_mut_dyn(visitor),
                MirExpressionStep::CaptureSequence { source, .. } => {
                    source.visit_operands_mut(&mut *visitor)
                }
            }
        }
        visitor(&mut self.result);
    }

    pub fn visit_rvalues_mut(&mut self, mut visitor: impl FnMut(&mut MirRvalue)) {
        for step in &mut self.steps {
            if let MirExpressionStep::Let { value, .. } = step {
                visitor(value);
            }
        }
    }

    pub fn try_visit_rvalues_mut<E>(
        &mut self,
        mut visitor: impl FnMut(&mut MirRvalue) -> Result<(), E>,
    ) -> Result<(), E> {
        for step in &mut self.steps {
            if let MirExpressionStep::Let { value, .. } = step {
                visitor(value)?;
            }
        }
        Ok(())
    }

    pub fn visit_capture_sources_mut(
        &mut self,
        mut visitor: impl FnMut(&mut crate::MirExpansionSource),
    ) {
        for step in &mut self.steps {
            if let MirExpressionStep::CaptureSequence { source, .. } = step {
                visitor(source);
            }
        }
    }

    pub fn try_visit_capture_sources_mut<E>(
        &mut self,
        mut visitor: impl FnMut(&mut crate::MirExpansionSource) -> Result<(), E>,
    ) -> Result<(), E> {
        for step in &mut self.steps {
            if let MirExpressionStep::CaptureSequence { source, .. } = step {
                visitor(source)?;
            }
        }
        Ok(())
    }

    pub fn visit_sequence_locals(&self, mut visitor: impl FnMut(&MirSequenceLocalId)) {
        for step in &self.steps {
            if let MirExpressionStep::Let { value, .. } = step {
                value.visit_sequence_locals(&mut visitor);
            }
        }
    }

    pub fn contains_contextual_end(&self) -> bool {
        self.steps.iter().any(|step| {
            matches!(step, MirExpressionStep::Let { value, .. } if value.contains_contextual_end())
        })
    }

    pub(crate) fn visit_nested_regions(&self, visitor: &mut dyn FnMut(&MirExpressionRegion)) {
        for step in &self.steps {
            match step {
                MirExpressionStep::Let { value, .. } => value.visit_expression_regions_dyn(visitor),
                MirExpressionStep::CaptureSequence { source, .. } => {
                    source.visit_expression_regions_dyn(visitor)
                }
            }
        }
    }
}
