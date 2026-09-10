use crate::{MirLocalId, MirOperand, MirRvalue, MirSequenceLocalId};
use runmat_hir::Span;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirExpressionRegion {
    pub(super) steps: Vec<MirExpressionStep>,
    pub(super) result: MirOperand,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirExpressionStep {
    Let {
        local: MirLocalId,
        value: MirRvalue,
        span: Span,
    },
    CaptureSequence {
        destination: MirSequenceLocalId,
        source: crate::MirExpansionSource,
        span: Span,
    },
}

impl MirExpressionRegion {
    pub fn steps(&self) -> &[MirExpressionStep] {
        &self.steps
    }

    pub fn result(&self) -> &MirOperand {
        &self.result
    }

    pub fn result_mut(&mut self) -> &mut MirOperand {
        &mut self.result
    }

    pub fn defined_locals(&self) -> impl Iterator<Item = MirLocalId> + '_ {
        self.steps.iter().filter_map(|step| match step {
            MirExpressionStep::Let { local, .. } => Some(*local),
            MirExpressionStep::CaptureSequence { .. } => None,
        })
    }

    pub fn defined_sequences(&self) -> impl Iterator<Item = MirSequenceLocalId> + '_ {
        self.steps.iter().filter_map(|step| match step {
            MirExpressionStep::CaptureSequence { destination, .. } => Some(*destination),
            MirExpressionStep::Let { .. } => None,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        super::validation::validate(self)
    }
}
