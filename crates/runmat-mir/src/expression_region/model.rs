use crate::{MirLocalId, MirOperand, MirRvalue, MirSequenceLocalId};
use runmat_types::Span;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirExpressionRegion {
    pub(super) steps: Vec<MirExpressionStep>,
    pub(super) result: MirOperand,
}

/// Straight-line selector preparation. Unlike `MirExpressionRegion`, its
/// result need not depend on a contextual `end` receiver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirPreparedExpressionRegion {
    pub(super) steps: Vec<MirExpressionStep>,
    pub(super) result: MirOperand,
}

/// A read-only traversal view. The variant preserves the validation role;
/// prepared regions can never be borrowed as contextual regions.
#[derive(Clone, Copy)]
pub enum MirExpressionRegionRef<'a> {
    Contextual(&'a MirExpressionRegion),
    Prepared(&'a MirPreparedExpressionRegion),
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
        super::validation::validate(self.steps(), self.result(), true)
    }
}

impl MirPreparedExpressionRegion {
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
        if self.steps.iter().any(|step| {
            matches!(step, MirExpressionStep::Let { value, .. } if value.contains_unscoped_end())
        }) {
            return Err("prepared expression contains an end without its own indexing receiver".to_owned());
        }
        super::validation::validate(self.steps(), self.result(), false)
    }
}

impl<'a> MirExpressionRegionRef<'a> {
    pub fn steps(self) -> &'a [MirExpressionStep] {
        match self {
            Self::Contextual(region) => region.steps(),
            Self::Prepared(region) => region.steps(),
        }
    }
    pub fn result(self) -> &'a MirOperand {
        match self {
            Self::Contextual(region) => region.result(),
            Self::Prepared(region) => region.result(),
        }
    }
    pub fn validate(self) -> Result<(), String> {
        match self {
            Self::Contextual(region) => region.validate(),
            Self::Prepared(region) => region.validate(),
        }
    }
    pub fn defined_locals(self) -> impl Iterator<Item = MirLocalId> + 'a {
        self.steps().iter().filter_map(|step| match step {
            MirExpressionStep::Let { local, .. } => Some(*local),
            MirExpressionStep::CaptureSequence { .. } => None,
        })
    }
    pub fn defined_sequences(self) -> impl Iterator<Item = MirSequenceLocalId> + 'a {
        self.steps().iter().filter_map(|step| match step {
            MirExpressionStep::CaptureSequence { destination, .. } => Some(*destination),
            MirExpressionStep::Let { .. } => None,
        })
    }
    pub fn visit_sequence_locals(self, mut visitor: impl FnMut(&MirSequenceLocalId)) {
        for step in self.steps() {
            match step {
                MirExpressionStep::Let { value, .. } => {
                    value.visit_direct_sequence_locals(&mut visitor)
                }
                MirExpressionStep::CaptureSequence { source, .. } => {
                    source.visit_sequence_locals(&mut visitor)
                }
            }
        }
    }
    pub(crate) fn visit_nested_regions(self, visitor: &mut dyn FnMut(MirExpressionRegionRef<'_>)) {
        for step in self.steps() {
            match step {
                MirExpressionStep::Let { value, .. } => value.visit_expression_regions_dyn(visitor),
                MirExpressionStep::CaptureSequence { source, .. } => {
                    source.visit_expression_regions_dyn(visitor)
                }
            }
        }
    }
}
