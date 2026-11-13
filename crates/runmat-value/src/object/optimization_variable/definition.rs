use super::{OptimizationBoundRole, OptimizationBoundStorage, OptimizationBoundView};
use crate::optimization_model::ModelAxisNames;
use crate::{NumericDType, SharedShape, Utf8Text};

use super::OptimizationVariableDomain;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimizationVariableLayoutError {
    ShapeRank,
    ShapeOverflow,
    BoundClass(OptimizationBoundRole),
    BoundLength(OptimizationBoundRole),
    BoundNaN(OptimizationBoundRole),
    IndexNamesRank,
    IndexNamesLength { axis: usize },
    IncompleteValidation,
}

/// Unvalidated constructor/edit input. Moving it never grants allocation,
/// class, solver, or callable authority. Runtime admits its backing first.
#[derive(Debug, PartialEq)]
pub struct OptimizationVariableCandidate {
    pub name: Utf8Text,
    pub domain: OptimizationVariableDomain,
    pub shape: SharedShape,
    /// One entry per axis. None denotes ordinary numeric indexing on that axis.
    pub index_names: ModelAxisNames,
    pub lower: OptimizationBoundStorage,
    pub upper: OptimizationBoundStorage,
}

/// One immutable definition generation, stored in the canonical GC handle.
/// Public label uniqueness, identifier syntax, and solver-domain feasibility
/// belong to their Runtime/model boundaries, not this storage representation.
#[derive(Debug, PartialEq)]
pub struct OptimizationVariableData {
    input: OptimizationVariableCandidate,
    element_count: usize,
}

/// Incremental validation lets Runtime charge its original CPU-work budget.
/// No numeric buffer, label collection, or shape is copied while validating.
pub struct OptimizationVariableValidation {
    input: OptimizationVariableCandidate,
    element_count: usize,
    cursor: usize,
    work_len: usize,
}

/// Incremental geometry/metadata preparation; the candidate owns its shape.
pub struct OptimizationVariablePreparation {
    input: OptimizationVariableCandidate,
    shape: crate::array::ShapeProgress,
    axis: usize,
    phase: u8,
}
impl OptimizationVariableCandidate {
    pub fn begin_incremental(
        self,
    ) -> Result<OptimizationVariablePreparation, OptimizationVariableLayoutError> {
        if self.shape.len() < 2 {
            return Err(OptimizationVariableLayoutError::ShapeRank);
        }
        Ok(OptimizationVariablePreparation {
            input: self,
            shape: crate::array::ShapeProgress::new(),
            axis: 0,
            phase: 0,
        })
    }
    pub fn begin(self) -> Result<OptimizationVariableValidation, OptimizationVariableLayoutError> {
        let mut prepared = self.begin_incremental()?;
        while prepared.prepare_next()? {}
        prepared.finish()
    }
}
impl OptimizationVariablePreparation {
    /// At most one canonical shape step or one axis is inspected per call.
    pub fn prepare_next(&mut self) -> Result<bool, OptimizationVariableLayoutError> {
        use OptimizationVariableLayoutError as Error;
        if self.phase == 0 {
            if !self.shape.is_complete() {
                self.shape
                    .advance(
                        &self.input.shape,
                        std::num::NonZeroUsize::new(1).expect("one"),
                        |_| "optimization variable shape overflows usize".into(),
                    )
                    .map_err(|_| Error::ShapeOverflow)?;
                return Ok(true);
            }
            if self.input.index_names.len() != self.input.shape.len() {
                return Err(Error::IndexNamesRank);
            }
            self.phase = 1;
        }
        if self.phase == 1 {
            if self.axis < self.input.shape.len() {
                if self.input.index_names[self.axis]
                    .as_ref()
                    .is_some_and(|names| names.len() != self.input.shape[self.axis])
                {
                    return Err(Error::IndexNamesLength { axis: self.axis });
                }
                self.axis += 1;
                return Ok(true);
            }
            for (role, values) in [
                (OptimizationBoundRole::Lower, &self.input.lower),
                (OptimizationBoundRole::Upper, &self.input.upper),
            ] {
                match values {
                    OptimizationBoundStorage::Uniform(value) if value.is_nan() => {
                        return Err(Error::BoundNaN(role))
                    }
                    OptimizationBoundStorage::Uniform(_) => {}
                    OptimizationBoundStorage::PerElement(values) => {
                        if values.numeric_dtype() != NumericDType::F64 {
                            return Err(Error::BoundClass(role));
                        }
                        if values.len() != self.shape.elements {
                            return Err(Error::BoundLength(role));
                        }
                    }
                }
            }
            self.phase = 2;
        }
        Ok(false)
    }
    pub fn finish(self) -> Result<OptimizationVariableValidation, OptimizationVariableLayoutError> {
        if self.phase != 2 {
            return Err(OptimizationVariableLayoutError::IncompleteValidation);
        }
        let work_len = self
            .input
            .lower
            .numeric_work_len()
            .max(self.input.upper.numeric_work_len());
        Ok(OptimizationVariableValidation {
            input: self.input,
            element_count: self.shape.elements,
            cursor: 0,
            work_len,
        })
    }
}

impl OptimizationVariableValidation {
    /// Validate at most one element of each nonuniform bound. A uniform bound
    /// was checked at begin, so even a large uniform variable has O(rank) setup.
    pub fn validate_next(&mut self) -> Result<bool, OptimizationVariableLayoutError> {
        if self.cursor == self.work_len {
            return Ok(false);
        }
        for (role, values) in [
            (OptimizationBoundRole::Lower, &self.input.lower),
            (OptimizationBoundRole::Upper, &self.input.upper),
        ] {
            if values.value(self.cursor).is_some_and(f64::is_nan) {
                return Err(OptimizationVariableLayoutError::BoundNaN(role));
            }
        }
        self.cursor += 1;
        Ok(true)
    }

    pub fn finish(self) -> Result<OptimizationVariableData, OptimizationVariableLayoutError> {
        if self.cursor != self.work_len {
            return Err(OptimizationVariableLayoutError::IncompleteValidation);
        }
        Ok(OptimizationVariableData {
            input: self.input,
            element_count: self.element_count,
        })
    }
}

impl OptimizationVariableData {
    pub fn name_owner(&self) -> &Utf8Text {
        &self.input.name
    }
    pub fn bound_storage(&self, role: OptimizationBoundRole) -> &OptimizationBoundStorage {
        match role {
            OptimizationBoundRole::Lower => &self.input.lower,
            OptimizationBoundRole::Upper => &self.input.upper,
        }
    }
    pub fn name(&self) -> &str {
        self.input.name.as_str()
    }
    pub fn domain(&self) -> OptimizationVariableDomain {
        self.input.domain
    }
    pub fn shape(&self) -> &SharedShape {
        &self.input.shape
    }
    pub fn element_count(&self) -> usize {
        self.element_count
    }
    pub fn index_names(&self) -> &ModelAxisNames {
        &self.input.index_names
    }
    pub fn bound(&self, role: OptimizationBoundRole) -> OptimizationBoundView<'_> {
        match role {
            OptimizationBoundRole::Lower => self.input.lower.view(),
            OptimizationBoundRole::Upper => self.input.upper.view(),
        }
    }
    pub fn bound_value(&self, role: OptimizationBoundRole, index: usize) -> Option<f64> {
        if index >= self.element_count {
            return None;
        }
        match role {
            OptimizationBoundRole::Lower => self.input.lower.value(index),
            OptimizationBoundRole::Upper => self.input.upper.value(index),
        }
    }
    /// Transfer an unpublished/retired generation for an atomic property edit.
    /// The caller retains the GC handle identity and publishes only after
    /// validating the replacement, so aliases never observe a partial edit.
    pub fn into_candidate(self) -> OptimizationVariableCandidate {
        self.input
    }
}
