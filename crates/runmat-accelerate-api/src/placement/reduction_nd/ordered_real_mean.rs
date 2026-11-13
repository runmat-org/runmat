//! One call-boundary real mean with source-ordered axes and explicit physical
//! classes. This route is distinct from the faster F32 moments/cache route.

use crate::{allocation::AllocationAcquisitionError, GpuTensorHandle, ProviderPrecision};

#[path = "ordered_real_mean/recipe.rs"]
mod recipe;
pub use recipe::{
    ProviderOrderedRealMeanControl, ProviderOrderedRealMeanLimits, ProviderOrderedRealMeanRecipe,
    ProviderOrderedRealMeanResources, ProviderOrderedRealMeanSource,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderRealMeanNaNMode {
    Include,
    Omit,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderOrderedRealMeanStep {
    axis: usize,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    input_precision: ProviderPrecision,
    output_precision: ProviderPrecision,
    input_bytes: u64,
    output_bytes: u64,
    uniform_bytes: u64,
    workgroups: u32,
}

impl ProviderOrderedRealMeanStep {
    pub fn new(
        axis: usize,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        input_precision: ProviderPrecision,
        output_precision: ProviderPrecision,
    ) -> Self {
        Self {
            axis,
            input_shape,
            output_shape,
            input_precision,
            output_precision,
            input_bytes: 0,
            output_bytes: 0,
            uniform_bytes: 0,
            workgroups: 0,
        }
    }

    pub fn with_physical_resources(
        mut self,
        input_bytes: u64,
        output_bytes: u64,
        uniform_bytes: u64,
        workgroups: u32,
    ) -> Self {
        self.input_bytes = input_bytes;
        self.output_bytes = output_bytes;
        self.uniform_bytes = uniform_bytes;
        self.workgroups = workgroups;
        self
    }

    pub fn axis(&self) -> usize {
        self.axis
    }
    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
    pub fn input_precision(&self) -> ProviderPrecision {
        self.input_precision
    }
    pub fn output_precision(&self) -> ProviderPrecision {
        self.output_precision
    }
    pub fn input_bytes(&self) -> u64 {
        self.input_bytes
    }
    pub fn output_bytes(&self) -> u64 {
        self.output_bytes
    }
    pub fn uniform_bytes(&self) -> u64 {
        self.uniform_bytes
    }
    pub fn workgroups(&self) -> u32 {
        self.workgroups
    }
}

/// A request-local physical proposal, not execution authority. The consuming
/// provider re-derives every step from the same live source and current limits.
pub struct ProviderOrderedRealMean {
    source: GpuTensorHandle,
    recipe: ProviderOrderedRealMeanRecipe,
}

impl ProviderOrderedRealMean {
    pub fn new(source: GpuTensorHandle, recipe: ProviderOrderedRealMeanRecipe) -> Self {
        Self { source, recipe }
    }

    pub fn source(&self) -> &GpuTensorHandle {
        &self.source
    }
    pub fn recipe(&self) -> &ProviderOrderedRealMeanRecipe {
        &self.recipe
    }
    pub fn steps(&self) -> &[ProviderOrderedRealMeanStep] {
        self.recipe.steps()
    }
    pub fn nan_mode(&self) -> ProviderRealMeanNaNMode {
        self.recipe.nan_mode()
    }
    pub fn requested_incremental_peak_bytes(&self) -> u64 {
        self.recipe.requested_incremental_peak_bytes()
    }
    pub fn validate_live_source(
        &self,
        source: &GpuTensorHandle,
    ) -> Result<bool, AllocationAcquisitionError> {
        self.source.allocation_lease().validate_live()?;
        source.allocation_lease().validate_live()?;
        Ok(self.source.view_identity() == source.view_identity())
    }
}

pub struct ProviderExecutedOrderedRealMean {
    pub output: GpuTensorHandle,
    pub requested_incremental_peak_bytes: u64,
    pub observed_incremental_peak_bytes: u64,
    pub observed_output_bytes: u64,
    pub selected_submissions: u32,
}
