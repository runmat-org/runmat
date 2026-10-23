use crate::{
    MirCallArg, MirExpressionRegion, MirExpressionRegionRef, MirExpressionStep, MirOperand,
    MirSequenceLocalId,
};
use runmat_types::Span;
use serde::{Deserialize, Serialize};

/// One source-level dotted-call argument. The expression preparation remains
/// attached to its invocation step so a later argument cannot run before an
/// earlier subscript-path invocation completes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirDottedInvokeArgument {
    Colon,
    Contextual(MirExpressionRegion),
    Call(MirCallArgumentRegion),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirCallArgumentRegion {
    pub steps: Vec<MirExpressionStep>,
    pub argument: MirCallArg,
    pub span: Span,
}

impl MirDottedInvokeArgument {
    pub(super) fn validate(&self) -> Result<(), String> {
        match self {
            Self::Colon => Ok(()),
            Self::Contextual(region) => region.validate(),
            Self::Call(region) => region.validate().map_err(str::to_owned),
        }
    }

    pub(crate) fn visit_sequence_locals(&self, visitor: &mut dyn FnMut(&MirSequenceLocalId)) {
        match self {
            Self::Colon => {}
            Self::Contextual(region) => {
                let local_sequences = region
                    .defined_sequences()
                    .collect::<std::collections::BTreeSet<_>>();
                region.visit_sequence_locals(|sequence| {
                    if !local_sequences.contains(sequence) {
                        visitor(sequence);
                    }
                });
            }
            Self::Call(region) => region.visit_sequence_locals(visitor),
        }
    }
    pub(super) fn contains_contextual_end(&self) -> bool {
        match self {
            Self::Colon => false,
            Self::Contextual(region) => region.contains_contextual_end(),
            Self::Call(region) => region.contains_contextual_end(),
        }
    }

    pub(super) fn visit_operands(&self, visitor: &mut dyn FnMut(&MirOperand)) {
        match self {
            Self::Colon => {}
            Self::Contextual(region) => region.visit_operands(visitor),
            Self::Call(region) => region.visit_operands(visitor),
        }
    }

    pub(super) fn visit_outer_operands(&self, visitor: &mut dyn FnMut(&MirOperand)) {
        match self {
            Self::Colon => {}
            Self::Contextual(region) => region.visit_external_operands(visitor),
            Self::Call(region) => region.visit_outer_operands(visitor),
        }
    }

    pub(super) fn visit_operands_mut(&mut self, visitor: &mut dyn FnMut(&mut MirOperand)) {
        match self {
            Self::Colon => {}
            Self::Contextual(region) => region.visit_operands_mut(visitor),
            Self::Call(region) => region.visit_operands_mut(visitor),
        }
    }

    pub(super) fn visit_expression_regions(
        &self,
        visitor: &mut dyn FnMut(MirExpressionRegionRef<'_>),
    ) {
        match self {
            Self::Colon => {}
            Self::Contextual(region) => visitor(MirExpressionRegionRef::Contextual(region)),
            Self::Call(region) => region.visit_expression_regions(visitor),
        }
    }

    pub(super) fn visit_direct_expression_regions_dyn(
        &self,
        visitor: &mut dyn FnMut(MirExpressionRegionRef<'_>),
    ) {
        self.visit_expression_regions(visitor);
    }
}

impl MirCallArgumentRegion {
    pub(crate) fn visit_sequence_locals(&self, visitor: &mut dyn FnMut(&MirSequenceLocalId)) {
        let local_sequences = self
            .steps
            .iter()
            .filter_map(|step| match step {
                MirExpressionStep::CaptureSequence { destination, .. } => Some(*destination),
                MirExpressionStep::Let { .. } => None,
            })
            .collect::<std::collections::BTreeSet<_>>();
        let mut visit_external = |sequence: &MirSequenceLocalId| {
            if !local_sequences.contains(sequence) {
                visitor(sequence);
            }
        };
        for step in &self.steps {
            match step {
                MirExpressionStep::Let { value, .. } => {
                    value.visit_sequence_locals(&mut visit_external)
                }
                MirExpressionStep::CaptureSequence { source, .. } => {
                    source.visit_sequence_locals(&mut visit_external)
                }
            }
        }
        self.argument.visit_sequence_locals(&mut visit_external);
    }
    pub(crate) fn visit_direct_preparation_steps(
        &self,
        visitor: &mut dyn FnMut(&[MirExpressionStep]),
    ) {
        visitor(&self.steps);
        self.argument.visit_direct_expression_steps_dyn(visitor);
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        self.validate_with_external_sequences(&std::collections::BTreeSet::new())
    }

    pub(crate) fn validate_with_external_sequences(
        &self,
        external: &std::collections::BTreeSet<MirSequenceLocalId>,
    ) -> Result<(), &'static str> {
        use std::collections::BTreeSet;
        let all_locals = self
            .steps
            .iter()
            .filter_map(|step| match step {
                MirExpressionStep::Let { local, .. } => Some(*local),
                MirExpressionStep::CaptureSequence { .. } => None,
            })
            .collect::<BTreeSet<_>>();
        let all_sequences = self
            .steps
            .iter()
            .filter_map(|step| match step {
                MirExpressionStep::CaptureSequence { destination, .. } => Some(*destination),
                MirExpressionStep::Let { .. } => None,
            })
            .collect::<BTreeSet<_>>();
        let mut defined_locals = BTreeSet::new();
        let mut defined_sequences = BTreeSet::new();
        let mut sequence_uses = std::collections::BTreeMap::new();
        for step in &self.steps {
            let mut invalid_local = false;
            let mut invalid_sequence = false;
            match step {
                MirExpressionStep::Let { local, value, .. } => {
                    value.visit_operands(|operand| {
                        if matches!(operand, MirOperand::Local(candidate)
                            if all_locals.contains(candidate) && !defined_locals.contains(candidate)) {
                            invalid_local = true;
                        }
                    });
                    value.visit_sequence_locals(|candidate| {
                        *sequence_uses.entry(*candidate).or_insert(0usize) += 1;
                        if all_sequences.contains(candidate)
                            && !defined_sequences.contains(candidate)
                        {
                            invalid_sequence = true;
                        }
                    });
                    if !defined_locals.insert(*local) {
                        return Err("dotted argument local is defined more than once");
                    }
                }
                MirExpressionStep::CaptureSequence {
                    destination,
                    source,
                    ..
                } => {
                    source.visit_operands(|operand| {
                        if matches!(operand, MirOperand::Local(candidate)
                            if all_locals.contains(candidate) && !defined_locals.contains(candidate)) {
                            invalid_local = true;
                        }
                    });
                    source.visit_sequence_locals(&mut |candidate| {
                        *sequence_uses.entry(*candidate).or_insert(0usize) += 1;
                        if all_sequences.contains(candidate)
                            && !defined_sequences.contains(candidate)
                        {
                            invalid_sequence = true;
                        }
                    });
                    if !defined_sequences.insert(*destination) {
                        return Err("dotted argument sequence is captured more than once");
                    }
                }
            }
            if invalid_local {
                return Err("dotted argument reads a local before its definition");
            }
            if invalid_sequence {
                return Err("dotted argument reads a sequence before its capture");
            }
        }
        let mut invalid_local = false;
        self.argument.visit_operands(|operand| {
            if matches!(operand, MirOperand::Local(candidate)
                if all_locals.contains(candidate) && !defined_locals.contains(candidate))
            {
                invalid_local = true;
            }
        });
        if invalid_local {
            return Err("dotted argument result reads a local before its definition");
        }
        let mut invalid_result_sequence = false;
        self.argument.visit_sequence_locals(&mut |sequence| {
            if !external.contains(sequence)
                && (!all_sequences.contains(sequence) || !defined_sequences.contains(sequence))
            {
                invalid_result_sequence = true;
            }
            *sequence_uses.entry(*sequence).or_insert(0usize) += 1;
        });
        if invalid_result_sequence {
            return Err("dotted argument references an uncaptured sequence");
        }
        if all_sequences
            .iter()
            .any(|sequence| sequence_uses.get(sequence) != Some(&1))
        {
            return Err("dotted argument must consume each captured sequence exactly once");
        }
        let mut invalid_nested_region = false;
        self.visit_expression_regions(&mut |region| {
            invalid_nested_region |= region.validate().is_err();
        });
        if invalid_nested_region {
            return Err("dotted argument contains an invalid contextual region");
        }
        Ok(())
    }

    pub fn contains_contextual_end(&self) -> bool {
        self.steps.iter().any(|step| match step {
            MirExpressionStep::Let { value, .. } => value.contains_contextual_end(),
            MirExpressionStep::CaptureSequence { source, .. } => source.contains_contextual_end(),
        }) || self.argument.contains_contextual_end()
    }

    /// End that needs the enclosing selector's receiver, as opposed to an
    /// End owned by a nested indexing/expansion receiver. Final Single is an
    /// operand, not an rvalue; expansion captures have their own receiver.
    pub fn contains_unscoped_end(&self) -> bool {
        self.steps.iter().any(|step| match step {
            MirExpressionStep::Let { value, .. } => value.contains_unscoped_end(),
            MirExpressionStep::CaptureSequence { .. } => false,
        })
    }

    pub fn visit_operands(&self, visitor: &mut dyn FnMut(&MirOperand)) {
        for step in &self.steps {
            match step {
                MirExpressionStep::Let { value, .. } => value.visit_operands_dyn(visitor),
                MirExpressionStep::CaptureSequence { source, .. } => {
                    source.visit_operands(&mut *visitor)
                }
            }
        }
        self.argument.visit_operands(visitor);
    }

    pub fn visit_outer_operands(&self, visitor: &mut dyn FnMut(&MirOperand)) {
        // A region's own locals are internal preparation details, not inputs to
        // the enclosing subscript path.
        let mut local_defs = self
            .steps
            .iter()
            .filter_map(|step| match step {
                MirExpressionStep::Let { local, .. } => Some(*local),
                MirExpressionStep::CaptureSequence { .. } => None,
            })
            .collect::<std::collections::BTreeSet<_>>();
        self.visit_expression_regions(&mut |region| local_defs.extend(region.defined_locals()));
        self.visit_operands(&mut |operand| {
            if !matches!(operand, MirOperand::Local(local) if local_defs.contains(local)) {
                visitor(operand);
            }
        });
    }

    pub fn visit_operands_mut(&mut self, visitor: &mut dyn FnMut(&mut MirOperand)) {
        for step in &mut self.steps {
            match step {
                MirExpressionStep::Let { value, .. } => value.visit_operands_mut_dyn(visitor),
                MirExpressionStep::CaptureSequence { source, .. } => {
                    source.visit_operands_mut(&mut *visitor)
                }
            }
        }
        self.argument.visit_operands_mut(visitor);
    }

    pub fn visit_expression_regions(&self, visitor: &mut dyn FnMut(MirExpressionRegionRef<'_>)) {
        for step in &self.steps {
            match step {
                MirExpressionStep::Let { value, .. } => value.visit_expression_regions_dyn(visitor),
                MirExpressionStep::CaptureSequence { source, .. } => {
                    source.visit_expression_regions_dyn(visitor)
                }
            }
        }
        self.argument.visit_direct_expression_regions_dyn(visitor);
    }
}
