//! Handle-free physical proposal for one ordered real mean call.

use crate::{ProviderPrecision, ProviderRepresentation};

use super::{ProviderOrderedRealMeanStep, ProviderRealMeanNaNMode};

/// Provider capability limits relevant to this recipe, independent of a
/// particular backend API. Live preparation compares current limits exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderOrderedRealMeanLimits {
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u32,
    pub max_uniform_buffer_binding_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroups_per_dimension: u32,
}

/// This contains no owner or allocation authority. The consuming provider
/// must authenticate a live view and rederive an identical recipe before any
/// effect. It is deliberately not serializable placement evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderOrderedRealMeanSource {
    pub representation: ProviderRepresentation,
    pub buffer_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderOrderedRealMeanControl {
    pub axes: Vec<usize>,
    pub output_precision: ProviderPrecision,
    pub nan_mode: ProviderRealMeanNaNMode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderOrderedRealMeanResources {
    pub limits: ProviderOrderedRealMeanLimits,
    pub steps: Vec<ProviderOrderedRealMeanStep>,
    pub incremental_peak_bytes: u64,
    /// Exact submissions issued by this provider route. The current WGPU
    /// implementation submits one encoder per ordered step; consumers must
    /// use this issued count rather than infer a generic dispatch ratio.
    pub queue_submissions: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderOrderedRealMeanRecipe {
    source: ProviderOrderedRealMeanSource,
    control: ProviderOrderedRealMeanControl,
    resources: ProviderOrderedRealMeanResources,
}

impl ProviderOrderedRealMeanRecipe {
    pub fn new(
        source: ProviderOrderedRealMeanSource,
        control: ProviderOrderedRealMeanControl,
        resources: ProviderOrderedRealMeanResources,
    ) -> Self {
        Self {
            source,
            control,
            resources,
        }
    }

    pub fn source(&self) -> &ProviderRepresentation {
        &self.source.representation
    }
    pub fn source_buffer_bytes(&self) -> u64 {
        self.source.buffer_bytes
    }
    pub fn axes(&self) -> &[usize] {
        &self.control.axes
    }
    pub fn output_precision(&self) -> ProviderPrecision {
        self.control.output_precision
    }
    pub fn nan_mode(&self) -> ProviderRealMeanNaNMode {
        self.control.nan_mode
    }
    pub fn limits(&self) -> ProviderOrderedRealMeanLimits {
        self.resources.limits
    }
    pub fn steps(&self) -> &[ProviderOrderedRealMeanStep] {
        &self.resources.steps
    }
    pub fn requested_incremental_peak_bytes(&self) -> u64 {
        self.resources.incremental_peak_bytes
    }
    pub fn queue_submissions(&self) -> u32 {
        self.resources.queue_submissions
    }
    pub fn source_facts(&self) -> &ProviderOrderedRealMeanSource {
        &self.source
    }
    pub fn control(&self) -> &ProviderOrderedRealMeanControl {
        &self.control
    }
    pub fn resources(&self) -> &ProviderOrderedRealMeanResources {
        &self.resources
    }
}
