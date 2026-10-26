//! Flat references to the original GC-owned variable definition.
use crate::{ExactArrayStorage, SharedShape};
use runmat_gc_api::{GcHandle, Trace, Tracer};

/// Setter authority is independent of the physical index encoding. Indexing an
/// entire array still produces a Reference, not a new mutable definition role.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimizationVariableReferenceRole {
    Definition,
    Reference,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimizationVariableAxisSource {
    Unnamed,
    DefinitionAxis(usize),
}

/// Whole-array references need no element-sized identity-index vector.
#[derive(Clone, Debug, PartialEq)]
pub enum OptimizationVariableIndices {
    Whole,
    Indexed(ExactArrayStorage<usize>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimizationVariableSelectionError {
    ShapeRank,
    AxisSourcesRank,
    ShapeOverflow,
    WholeLength,
    IndexLength,
    IndexOutOfRange { position: usize },
    IncompleteValidation,
}

/// Import/constructor input, not proof that the GC target has the correct
/// class, definition generation, ownership, or variable name. Runtime checks
/// the actual target and roots it before constructing or publishing a view.
pub struct OptimizationVariableSelectionCandidate {
    pub definition: GcHandle,
    pub role: OptimizationVariableReferenceRole,
    pub definition_elements: usize,
    pub indices: OptimizationVariableIndices,
    pub shape: SharedShape,
    pub axis_sources: ExactArrayStorage<OptimizationVariableAxisSource>,
}

#[derive(Debug, PartialEq)]
pub struct OptimizationVariableSelection {
    definition: GcHandle,
    role: OptimizationVariableReferenceRole,
    indices: OptimizationVariableIndices,
    shape: SharedShape,
    axis_sources: ExactArrayStorage<OptimizationVariableAxisSource>,
    element_count: usize,
}

pub struct OptimizationVariableSelectionValidation {
    candidate: OptimizationVariableSelectionCandidate,
    element_count: usize,
    cursor: usize,
}

pub struct OptimizationVariableSelectionPreparation {
    candidate: OptimizationVariableSelectionCandidate,
    shape: crate::array::ShapeProgress,
    complete: bool,
}
impl OptimizationVariableSelectionCandidate {
    pub fn begin_incremental(
        self,
    ) -> Result<OptimizationVariableSelectionPreparation, OptimizationVariableSelectionError> {
        use OptimizationVariableSelectionError as Error;
        if self.shape.len() < 2 {
            return Err(Error::ShapeRank);
        }
        if self.axis_sources.len() != self.shape.len() {
            return Err(Error::AxisSourcesRank);
        }
        Ok(OptimizationVariableSelectionPreparation {
            candidate: self,
            shape: crate::array::ShapeProgress::new(),
            complete: false,
        })
    }
    pub fn begin(
        self,
    ) -> Result<OptimizationVariableSelectionValidation, OptimizationVariableSelectionError> {
        let mut prepared = self.begin_incremental()?;
        while prepared.prepare_next()? {}
        prepared.finish()
    }
}
impl OptimizationVariableSelectionPreparation {
    pub fn prepare_next(&mut self) -> Result<bool, OptimizationVariableSelectionError> {
        use OptimizationVariableSelectionError as Error;
        if !self.shape.is_complete() {
            self.shape
                .advance(
                    &self.candidate.shape,
                    std::num::NonZeroUsize::new(1).expect("one"),
                    |_| "optimization variable reference shape overflows usize".into(),
                )
                .map_err(|_| Error::ShapeOverflow)?;
            return Ok(true);
        }
        match &self.candidate.indices {
            OptimizationVariableIndices::Whole
                if self.shape.elements != self.candidate.definition_elements =>
            {
                return Err(Error::WholeLength)
            }
            OptimizationVariableIndices::Indexed(indices)
                if indices.len() != self.shape.elements =>
            {
                return Err(Error::IndexLength)
            }
            _ => {}
        }
        self.complete = true;
        Ok(false)
    }
    pub fn finish(
        self,
    ) -> Result<OptimizationVariableSelectionValidation, OptimizationVariableSelectionError> {
        if !self.complete {
            return Err(OptimizationVariableSelectionError::IncompleteValidation);
        }
        Ok(OptimizationVariableSelectionValidation {
            candidate: self.candidate,
            element_count: self.shape.elements,
            cursor: 0,
        })
    }
}

impl OptimizationVariableSelectionValidation {
    /// Consume at most one explicit index. Whole-array references finish
    /// immediately, independent of their element count.
    pub fn validate_next(&mut self) -> Result<bool, OptimizationVariableSelectionError> {
        let OptimizationVariableIndices::Indexed(indices) = &self.candidate.indices else {
            return Ok(false);
        };
        let Some(&index) = indices.get(self.cursor) else {
            return Ok(false);
        };
        if index >= self.candidate.definition_elements {
            return Err(OptimizationVariableSelectionError::IndexOutOfRange {
                position: self.cursor,
            });
        }
        self.cursor += 1;
        Ok(true)
    }

    pub fn finish(
        self,
    ) -> Result<OptimizationVariableSelection, OptimizationVariableSelectionError> {
        if matches!(&self.candidate.indices, OptimizationVariableIndices::Indexed(indices) if self.cursor != indices.len())
        {
            return Err(OptimizationVariableSelectionError::IncompleteValidation);
        }
        Ok(OptimizationVariableSelection {
            definition: self.candidate.definition,
            role: self.candidate.role,
            indices: self.candidate.indices,
            shape: self.candidate.shape,
            axis_sources: self.candidate.axis_sources,
            element_count: self.element_count,
        })
    }
}

impl OptimizationVariableSelection {
    pub fn index_storage(&self) -> Option<&ExactArrayStorage<usize>> {
        match &self.indices {
            OptimizationVariableIndices::Whole => None,
            OptimizationVariableIndices::Indexed(indices) => Some(indices),
        }
    }
    pub fn axis_sources(&self) -> &ExactArrayStorage<OptimizationVariableAxisSource> {
        &self.axis_sources
    }
    pub fn role(&self) -> OptimizationVariableReferenceRole {
        self.role
    }
    pub fn is_whole(&self) -> bool {
        matches!(self.indices, OptimizationVariableIndices::Whole)
    }
    pub fn definition(&self) -> GcHandle {
        self.definition
    }
    pub fn shape(&self) -> &SharedShape {
        &self.shape
    }
    pub fn element_count(&self) -> usize {
        self.element_count
    }
    pub fn definition_index(&self, position: usize) -> Option<usize> {
        if position >= self.element_count {
            return None;
        }
        match &self.indices {
            OptimizationVariableIndices::Whole => Some(position),
            OptimizationVariableIndices::Indexed(indices) => indices.get(position).copied(),
        }
    }
}

impl Trace for OptimizationVariableSelection {
    fn trace(&self, tracer: &mut dyn Tracer) {
        tracer.mark(self.definition);
    }
}
