use super::*;
use crate::scalar_program::ScalarLiteral;
use crate::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider, HostTensorView,
    NumericElementType,
};
use crate::{GpuTensorHandle, GpuTensorStorage};

struct ProgressFixture {
    owner: TestAllocationOwner,
}

impl AccelProvider for ProgressFixture {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "stochastic progress fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unused")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unused") })
    }
}

fn representation(class: NumericClass) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type: provider_element(class).unwrap(),
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: vec![65, 1],
        residency: ProviderResidency::Device,
    }
}

#[test]
fn mixed_first_step_must_name_the_stable_continuing_recipe() {
    let first = StochasticEvolutionArithmetic::new(
        NumericClass::Double,
        NumericClass::Single,
        ScalarLiteral::F32Bits(0),
        ScalarLiteral::F32Bits(0),
    )
    .unwrap();
    assert_eq!(first.result_class(), NumericClass::Single);
    let continuing = StochasticEvolutionArithmetic::new(
        NumericClass::Single,
        NumericClass::Single,
        ScalarLiteral::F32Bits(0),
        ScalarLiteral::F32Bits(0),
    )
    .unwrap();
    let plan = ProviderStochasticEvolutionPlanRequest {
        state: representation(NumericClass::Double),
        output: representation(NumericClass::Single),
        first_arithmetic: first,
        continuing_arithmetic: continuing,
        steps: 3,
        noise_tile_elements: 64,
    };
    assert_eq!(plan.validate(), Some(65));
    assert_eq!(plan.arithmetic_for_step(0), Some(first));
    assert_eq!(plan.arithmetic_for_step(2), Some(continuing));
    let mut forged = plan.clone();
    forged.continuing_arithmetic = first;
    assert_eq!(forged.validate(), None);
}

#[test]
fn one_step_output_is_the_first_recipe_result() {
    let first = StochasticEvolutionArithmetic::new(
        NumericClass::Double,
        NumericClass::Single,
        ScalarLiteral::F32Bits(0),
        ScalarLiteral::F32Bits(0),
    )
    .unwrap();
    let continuing = StochasticEvolutionArithmetic::new(
        NumericClass::Single,
        NumericClass::Single,
        ScalarLiteral::F32Bits(0),
        ScalarLiteral::F32Bits(0),
    )
    .unwrap();
    let plan = ProviderStochasticEvolutionPlanRequest {
        state: representation(NumericClass::Double),
        output: representation(NumericClass::Single),
        first_arithmetic: first,
        continuing_arithmetic: continuing,
        steps: 1,
        noise_tile_elements: 65,
    };
    assert_eq!(plan.validate(), Some(65));
}

#[test]
fn physical_proposal_accounts_the_typed_noise_tile_and_exact_plan_identity() {
    let arithmetic = StochasticEvolutionArithmetic::new(
        NumericClass::Single,
        NumericClass::Double,
        ScalarLiteral::F32Bits(0),
        ScalarLiteral::F64Bits(0),
    )
    .unwrap();
    let plan = ProviderStochasticEvolutionPlanRequest {
        state: representation(NumericClass::Single),
        output: representation(NumericClass::Single),
        first_arithmetic: arithmetic,
        continuing_arithmetic: arithmetic,
        steps: 2,
        noise_tile_elements: 64,
    };
    let proposal = ProviderStochasticEvolutionPhysicalProposal {
        plan_identity: plan.identity().unwrap(),
        first_output_buffer_bytes: 65 * 4,
        continuing_output_buffer_bytes: 65 * 4,
        final_output_buffer_bytes: 65 * 4,
        max_noise_tile_buffer_bytes: 64 * 8,
        uniform_buffer_bytes_per_tile: 32,
        completion_fence_bytes_per_tile: 8,
        observation_fence_bytes_per_tile: 8,
        completion_control_host_bytes: 1,
        authentication_host_bytes_per_tile: 8 * std::mem::size_of::<usize>() as u64,
        submission_retention_host_bytes_per_tile: 1,
        source_transfer_retirement_host_bytes: 1,
        observation_host_control_bytes_per_tile: 1,
        output_view_host_bytes: crate::GpuTensorHandle::view_host_upper_bytes(2).unwrap(),
        tiles_per_step: 2,
        dispatches_per_step: 2,
        queue_submissions_per_step: 2,
        full_call_dispatches: 4,
        full_call_queue_submissions: 4,
        peak_provisional_output_buffers: 1,
        frozen_workgroup_size: 64,
        frozen_tile_capacity: 64,
    };
    assert!(proposal.validate_for(&plan));
    let work = ProviderStochasticEvolutionWorkRequest {
        state: plan.state.clone(),
        output: plan.output.clone(),
        first_arithmetic: plan.first_arithmetic,
        continuing_arithmetic: plan.continuing_arithmetic,
        steps: plan.steps,
    };
    assert_eq!(work.validate(), Some(65));
    let selected = ProviderStochasticEvolutionSelection {
        plan: plan.clone(),
        proposal,
    };
    assert!(selected.validate_for(&work));
    let mut wrong_noise_bound = proposal;
    wrong_noise_bound.max_noise_tile_buffer_bytes -= 1;
    assert!(!wrong_noise_bound.validate_for(&plan));
}

#[test]
fn progress_rejects_a_skipped_initial_prefix() {
    let provider: &'static ProgressFixture = Box::leak(Box::new(ProgressFixture {
        owner: TestAllocationOwner::new(),
    }));
    // SAFETY: the leaked fixture has a unique device namespace and remains
    // live for the process, matching the provider registry contract.
    unsafe { crate::register_device_provider(provider) };
    let source = provider
        .owner
        .allocate_zeroed(vec![65, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let noise = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let arithmetic = StochasticEvolutionArithmetic::new(
        NumericClass::Single,
        NumericClass::Double,
        ScalarLiteral::F32Bits(0),
        ScalarLiteral::F64Bits(0),
    )
    .unwrap();
    let plan = ProviderStochasticEvolutionPlanRequest {
        state: representation(NumericClass::Single),
        output: representation(NumericClass::Single),
        first_arithmetic: arithmetic,
        continuing_arithmetic: arithmetic,
        steps: 1,
        noise_tile_elements: 64,
    };
    let proposal = ProviderStochasticEvolutionPhysicalProposal {
        plan_identity: plan.identity().unwrap(),
        first_output_buffer_bytes: 65 * 4,
        continuing_output_buffer_bytes: 65 * 4,
        final_output_buffer_bytes: 65 * 4,
        max_noise_tile_buffer_bytes: 64 * 8,
        uniform_buffer_bytes_per_tile: 32,
        completion_fence_bytes_per_tile: 8,
        observation_fence_bytes_per_tile: 8,
        completion_control_host_bytes: 1,
        authentication_host_bytes_per_tile: 8 * std::mem::size_of::<usize>() as u64,
        submission_retention_host_bytes_per_tile: 1,
        source_transfer_retirement_host_bytes: 1,
        observation_host_control_bytes_per_tile: 1,
        output_view_host_bytes: crate::GpuTensorHandle::view_host_upper_bytes(2).unwrap(),
        tiles_per_step: 2,
        dispatches_per_step: 2,
        queue_submissions_per_step: 2,
        full_call_dispatches: 2,
        full_call_queue_submissions: 2,
        peak_provisional_output_buffers: 1,
        frozen_workgroup_size: 64,
        frozen_tile_capacity: 64,
    };
    let skipped = ProviderStochasticEvolutionTileRequest {
        plan: &plan,
        proposal: &proposal,
        source: &source.handle,
        noise: &noise.handle,
        step: 0,
        offset: 64,
        pending: None,
    };
    assert!(skipped.validate_for(provider).is_err());
}
