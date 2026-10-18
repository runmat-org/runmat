//! Typed provider contract for one caller-supplied stochastic-evolution step.
//!
//! Random-stream ownership remains with Runtime. Providers receive already
//! materialized noise and may only execute the certified arithmetic recipe.

use crate::{
    ProviderElementType, ProviderLayout, ProviderRepresentation, ProviderResidency, ProviderStorage,
};
use runmat_execution::Digest;
use runmat_types::NumericClass;
use serde::{Deserialize, Serialize};

#[path = "stochastic_evolution/arithmetic.rs"]
mod arithmetic;
pub use arithmetic::{StochasticEvolutionArithmetic, StochasticEvolutionArithmeticClass};
#[path = "stochastic_evolution/progress.rs"]
mod progress;
pub use progress::{
    ProviderStochasticEvolutionPending, ProviderStochasticEvolutionProgress,
    ProviderStochasticEvolutionTileRequest,
};

pub const PROVIDER_STOCHASTIC_EVOLUTION_STEP_OPERATION: &str = "random.stochastic_evolution.step";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderStochasticEvolutionWorkload {
    pub plan_identity: Digest,
    pub first_arithmetic: StochasticEvolutionArithmetic,
    pub continuing_arithmetic: StochasticEvolutionArithmetic,
    pub steps: u32,
    pub noise_tile_elements: u64,
}

fn provider_element(class: NumericClass) -> Option<ProviderElementType> {
    match class {
        NumericClass::Single => Some(ProviderElementType::F32),
        NumericClass::Double => Some(ProviderElementType::F64),
        _ => None,
    }
}

/// Semantic whole-call facts supplied by Runtime before it draws or uploads
/// noise.  Tile geometry is deliberately absent: the physical provider owns
/// that choice.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderStochasticEvolutionWorkRequest {
    pub state: ProviderRepresentation,
    pub output: ProviderRepresentation,
    pub first_arithmetic: StochasticEvolutionArithmetic,
    pub continuing_arithmetic: StochasticEvolutionArithmetic,
    pub steps: u32,
}

impl ProviderStochasticEvolutionWorkRequest {
    pub fn validate(&self) -> Option<u64> {
        validate_semantic_work(
            &self.state,
            &self.output,
            self.first_arithmetic,
            self.continuing_arithmetic,
            self.steps,
        )
    }

    pub fn with_noise_tile(
        &self,
        noise_tile_elements: u64,
    ) -> Option<ProviderStochasticEvolutionPlanRequest> {
        let plan = ProviderStochasticEvolutionPlanRequest {
            state: self.state.clone(),
            output: self.output.clone(),
            first_arithmetic: self.first_arithmetic,
            continuing_arithmetic: self.continuing_arithmetic,
            steps: self.steps,
            noise_tile_elements,
        };
        plan.validate()?;
        Some(plan)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderStochasticEvolutionPlanRequest {
    pub state: ProviderRepresentation,
    pub output: ProviderRepresentation,
    pub first_arithmetic: StochasticEvolutionArithmetic,
    pub continuing_arithmetic: StochasticEvolutionArithmetic,
    pub steps: u32,
    pub noise_tile_elements: u64,
}

impl ProviderStochasticEvolutionPlanRequest {
    pub fn validate(&self) -> Option<u64> {
        let count = validate_semantic_work(
            &self.state,
            &self.output,
            self.first_arithmetic,
            self.continuing_arithmetic,
            self.steps,
        )?;
        if self.noise_tile_elements == 0 || self.noise_tile_elements > count {
            return None;
        }
        Some(count)
    }

    pub fn identity(&self) -> Option<Digest> {
        self.validate()?;
        Digest::sha256_serialized(self).ok()
    }

    pub fn workload(&self) -> Option<ProviderStochasticEvolutionWorkload> {
        Some(ProviderStochasticEvolutionWorkload {
            plan_identity: self.identity()?,
            first_arithmetic: self.first_arithmetic,
            continuing_arithmetic: self.continuing_arithmetic,
            steps: self.steps,
            noise_tile_elements: self.noise_tile_elements,
        })
    }

    pub fn arithmetic_for_step(&self, step: u32) -> Option<StochasticEvolutionArithmetic> {
        (step < self.steps).then_some(if step == 0 {
            self.first_arithmetic
        } else {
            self.continuing_arithmetic
        })
    }
}

fn validate_semantic_work(
    state: &ProviderRepresentation,
    output: &ProviderRepresentation,
    first_arithmetic: StochasticEvolutionArithmetic,
    continuing_arithmetic: StochasticEvolutionArithmetic,
    steps: u32,
) -> Option<u64> {
    if steps == 0
        || state.storage != ProviderStorage::DenseReal
        || output.storage != ProviderStorage::DenseReal
        || state.layout != ProviderLayout::ColumnMajorContiguous
        || output.layout != ProviderLayout::ColumnMajorContiguous
        || state.residency != ProviderResidency::Device
        || output.residency != ProviderResidency::Device
        || state.shape != output.shape
        || state.element_type != provider_element(first_arithmetic.state_class())?
        || continuing_arithmetic.state_class() != first_arithmetic.result_class()
        || continuing_arithmetic.result_class() != continuing_arithmetic.state_class()
        || continuing_arithmetic.noise_class() != first_arithmetic.noise_class()
        || continuing_arithmetic.drift() != first_arithmetic.drift()
        || continuing_arithmetic.scale() != first_arithmetic.scale()
        || output.element_type
            != provider_element(if steps == 1 {
                first_arithmetic.result_class()
            } else {
                continuing_arithmetic.result_class()
            })?
    {
        return None;
    }
    let count = state.checked_element_count()?;
    (count != 0).then_some(count)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderStochasticEvolutionPhysicalProposal {
    pub plan_identity: Digest,
    pub first_output_buffer_bytes: u64,
    pub continuing_output_buffer_bytes: u64,
    pub final_output_buffer_bytes: u64,
    /// Maximum physical bytes occupied by one caller-supplied noise tile.
    /// Runtime separately accounts the full logical noise value and its upload.
    pub max_noise_tile_buffer_bytes: u64,
    pub uniform_buffer_bytes_per_tile: u64,
    /// Operation-owned map fence used to establish the tile's semantic
    /// completion before progress is returned to Runtime.
    pub completion_fence_bytes_per_tile: u64,
    /// Canonical provider-observation fence retained until exact queue
    /// settlement. This is distinct from the semantic map fence above.
    pub observation_fence_bytes_per_tile: u64,
    pub completion_control_host_bytes: u64,
    pub authentication_host_bytes_per_tile: u64,
    /// Opaque provider completion owner retained until the exact submitted
    /// tile settles. Its referenced leases and claims are accounted by their
    /// existing owners rather than duplicated here.
    pub submission_retention_host_bytes_per_tile: u64,
    /// Shared acknowledgement state for the exact observed upload fence. One
    /// instance is live for the initial state or current noise-tile transfer.
    pub source_transfer_retirement_host_bytes: u64,
    /// Maximum request-owned host control backing for one exact observed
    /// submission. Stochastic tiles and selected source transfers retire
    /// serially, so only one such owner may be live at a time.
    pub observation_host_control_bytes_per_tile: u64,
    pub output_view_host_bytes: u64,
    pub tiles_per_step: u32,
    pub dispatches_per_step: u32,
    pub queue_submissions_per_step: u32,
    pub full_call_dispatches: u64,
    pub full_call_queue_submissions: u64,
    /// This primitive owns one provisional step output. Runtime separately
    /// accounts every published state/noise owner retained by the source.
    pub peak_provisional_output_buffers: u8,
    pub frozen_workgroup_size: u32,
    pub frozen_tile_capacity: u64,
}

impl ProviderStochasticEvolutionPhysicalProposal {
    pub fn output_buffer_bytes_for_step(self, step: u32) -> Option<u64> {
        if step == 0 {
            Some(self.first_output_buffer_bytes)
        } else {
            Some(self.continuing_output_buffer_bytes)
        }
    }

    pub fn validate_for(self, request: &ProviderStochasticEvolutionPlanRequest) -> bool {
        let Some(identity) = request.identity() else {
            return false;
        };
        let Some(count) = request.validate() else {
            return false;
        };
        let Some(final_output_bytes) = request.output.checked_byte_len() else {
            return false;
        };
        let Some(output_view_host_bytes) =
            crate::GpuTensorHandle::view_host_upper_bytes(request.output.shape.len())
        else {
            return false;
        };
        let Some(first_element) = provider_element(request.first_arithmetic.result_class()) else {
            return false;
        };
        let Some(continuing_element) =
            provider_element(request.continuing_arithmetic.result_class())
        else {
            return false;
        };
        let Some(noise_element) = provider_element(request.first_arithmetic.noise_class()) else {
            return false;
        };
        let Some(first_output_bytes) = count.checked_mul(first_element.byte_width()) else {
            return false;
        };
        let Some(continuing_output_bytes) = count.checked_mul(continuing_element.byte_width())
        else {
            return false;
        };
        let Some(max_noise_tile_bytes) = request
            .noise_tile_elements
            .checked_mul(noise_element.byte_width())
        else {
            return false;
        };
        let Some(tiles) = count
            .checked_add(request.noise_tile_elements - 1)
            .map(|n| n / request.noise_tile_elements)
        else {
            return false;
        };
        let Some(full_dispatches) = tiles.checked_mul(u64::from(request.steps)) else {
            return false;
        };
        identity == self.plan_identity
            && self.first_output_buffer_bytes == first_output_bytes
            && self.continuing_output_buffer_bytes == continuing_output_bytes
            && self.final_output_buffer_bytes == final_output_bytes
            && self.max_noise_tile_buffer_bytes == max_noise_tile_bytes
            && self.uniform_buffer_bytes_per_tile != 0
            && self.completion_fence_bytes_per_tile != 0
            && self.observation_fence_bytes_per_tile != 0
            && self.completion_control_host_bytes != 0
            && self.authentication_host_bytes_per_tile != 0
            && self.submission_retention_host_bytes_per_tile != 0
            && self.source_transfer_retirement_host_bytes != 0
            && self.observation_host_control_bytes_per_tile != 0
            && self.output_view_host_bytes == output_view_host_bytes
            && u64::from(self.tiles_per_step) == tiles
            && self.dispatches_per_step == self.tiles_per_step
            && self.queue_submissions_per_step == self.tiles_per_step
            && self.full_call_dispatches == full_dispatches
            && self.full_call_queue_submissions == full_dispatches
            && self.peak_provisional_output_buffers == 1
            && self.frozen_workgroup_size != 0
            && self.frozen_tile_capacity >= request.noise_tile_elements
    }
}

/// One immutable provider-selected plan and its exact physical proposal.
/// Runtime uses this same pair for feasibility, resource selection, noise
/// tiling, and execution; it never guesses a provider launch capacity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderStochasticEvolutionSelection {
    pub plan: ProviderStochasticEvolutionPlanRequest,
    pub proposal: ProviderStochasticEvolutionPhysicalProposal,
}

impl ProviderStochasticEvolutionSelection {
    pub fn validate_for(&self, work: &ProviderStochasticEvolutionWorkRequest) -> bool {
        self.plan.state == work.state
            && self.plan.output == work.output
            && self.plan.first_arithmetic == work.first_arithmetic
            && self.plan.continuing_arithmetic == work.continuing_arithmetic
            && self.plan.steps == work.steps
            && self.proposal.validate_for(&self.plan)
    }
}

#[cfg(test)]
#[path = "stochastic_evolution/tests.rs"]
mod tests;
