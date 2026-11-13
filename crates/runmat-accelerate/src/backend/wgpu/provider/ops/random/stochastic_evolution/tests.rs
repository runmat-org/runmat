use crate::WgpuProvider;
use runmat_accelerate_api::observation::{
    observe_provider_future, observe_provider_resource_future, ProviderOperationReceipt,
    ProviderResourcePhasePlan,
};
use runmat_accelerate_api::scalar_program::ScalarLiteral;
use runmat_accelerate_api::stochastic_evolution::{
    ProviderStochasticEvolutionPlanRequest, ProviderStochasticEvolutionProgress,
    ProviderStochasticEvolutionTileRequest, StochasticEvolutionArithmetic,
};
use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorView, ProviderElementType, ProviderLayout, ProviderPrecision,
    ProviderRepresentation, ProviderResidency, ProviderStorage,
};
use runmat_types::NumericClass;

fn operation_receipt(provider: &WgpuProvider) -> ProviderOperationReceipt {
    #[cfg(not(target_arch = "wasm32"))]
    {
        ProviderOperationReceipt::with_wgpu_progress(
            provider.observation_budget.clone(),
            provider
                .observation_progress
                .as_ref()
                .expect("actual WGPU provider has native progress")
                .clone(),
        )
    }
    #[cfg(target_arch = "wasm32")]
    {
        ProviderOperationReceipt::with_budget(provider.observation_budget.clone())
    }
}

async fn execute_tile<'a>(
    provider: &'a WgpuProvider,
    request: ProviderStochasticEvolutionTileRequest<'a>,
) -> ProviderStochasticEvolutionProgress {
    let phase_bytes = request
        .proposal
        .uniform_buffer_bytes_per_tile
        .checked_add(request.proposal.completion_fence_bytes_per_tile)
        .and_then(|bytes| bytes.checked_add(request.proposal.observation_fence_bytes_per_tile))
        .expect("selected tile resources fit u64");
    let operation = operation_receipt(provider);
    let resource_scope = operation
        .recorder()
        .begin_resource_scope(ProviderResourcePhasePlan::new([phase_bytes]).unwrap())
        .unwrap();
    resource_scope.begin_next_phase().unwrap();
    let progress = observe_provider_resource_future(
        Some(resource_scope.clone()),
        observe_provider_future(
            Some(operation.recorder()),
            provider.stochastic_evolution_tile_exec(request, None, None),
        ),
    )
    .await
    .expect("selected stochastic tile");
    resource_scope.finish().unwrap();
    progress
}

fn numeric_view<'a>(
    data: &'a HostNumericDataOwned,
    shape: &'a [usize],
) -> HostNumericTensorView<'a> {
    HostNumericTensorView {
        data: match data {
            HostNumericDataOwned::F32(values) => HostNumericDataView::F32(values),
            HostNumericDataOwned::F64(values) => HostNumericDataView::F64(values),
            _ => unreachable!(),
        },
        shape,
        storage: GpuTensorStorage::Real,
    }
}

async fn actual_tail() {
    #[cfg(not(target_arch = "wasm32"))]
    let provider = crate::backend::wgpu::provider::register_wgpu_provider(Default::default())
        .expect("actual WGPU provider required");
    #[cfg(target_arch = "wasm32")]
    let provider = crate::backend::wgpu::provider::register_wgpu_provider_async(Default::default())
        .await
        .expect("actual browser WebGPU provider required");
    assert_eq!(
        provider.stochastic_evolution_planning_host_workspace(2),
        runmat_execution::ResourceByteBound::Known(0)
    );
    assert_eq!(
        provider.stochastic_evolution_source_transfer_retirement_host_workspace(),
        runmat_execution::ResourceByteBound::Known(
            runmat_accelerate_api::observation::stochastic_evolution_source_transfer_retirement_host_upper_bytes()
                .expect("finite retirement control")
        )
    );
    let (state_class, noise_class, element, state, noise, drift, scale) = match provider.precision()
    {
        ProviderPrecision::F32 => (
            NumericClass::Single,
            NumericClass::Single,
            ProviderElementType::F32,
            HostNumericDataOwned::F32(vec![1.0; 65]),
            HostNumericDataOwned::F32((0..65).map(|index| index as f32 / 100.0).collect()),
            ScalarLiteral::F32Bits(0.1_f32.to_bits()),
            ScalarLiteral::F32Bits(0.2_f32.to_bits()),
        ),
        ProviderPrecision::F64 => (
            NumericClass::Single,
            NumericClass::Double,
            ProviderElementType::F32,
            HostNumericDataOwned::F32(vec![1.0; 65]),
            HostNumericDataOwned::F64((0..65).map(|index| index as f64 / 100.0).collect()),
            ScalarLiteral::F32Bits(0.1_f32.to_bits()),
            ScalarLiteral::F64Bits(0.2_f64.to_bits()),
        ),
    };
    let state_shape = [65, 1];
    let state_handle = provider
        .upload_numeric(&numeric_view(&state, &state_shape))
        .unwrap();
    let representation = ProviderRepresentation {
        element_type: element,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: vec![65, 1],
        residency: ProviderResidency::Device,
    };
    let arithmetic =
        StochasticEvolutionArithmetic::new(state_class, noise_class, drift, scale).unwrap();
    let plan = ProviderStochasticEvolutionPlanRequest {
        state: representation.clone(),
        output: representation,
        first_arithmetic: arithmetic,
        continuing_arithmetic: arithmetic,
        steps: 1,
        noise_tile_elements: 64,
    };
    let proposal = provider
        .propose_stochastic_evolution_step_exec(&plan)
        .expect("selected stochastic step route");
    assert_eq!(proposal.tiles_per_step, 2);
    let first_data = match &noise {
        HostNumericDataOwned::F32(values) => HostNumericDataOwned::F32(values[..64].to_vec()),
        HostNumericDataOwned::F64(values) => HostNumericDataOwned::F64(values[..64].to_vec()),
        _ => unreachable!(),
    };
    let first_shape = [64, 1];
    let first_noise = provider
        .upload_numeric(&numeric_view(&first_data, &first_shape))
        .unwrap();
    let first = execute_tile(
        provider,
        ProviderStochasticEvolutionTileRequest {
            plan: &plan,
            proposal: &proposal,
            source: &state_handle,
            noise: &first_noise,
            step: 0,
            offset: 0,
            pending: None,
        },
    )
    .await;
    assert!(!first.is_complete());
    let final_data = match &noise {
        HostNumericDataOwned::F32(values) => HostNumericDataOwned::F32(values[64..].to_vec()),
        HostNumericDataOwned::F64(values) => HostNumericDataOwned::F64(values[64..].to_vec()),
        _ => unreachable!(),
    };
    let final_shape = [1, 1];
    let final_noise = provider
        .upload_numeric(&numeric_view(&final_data, &final_shape))
        .unwrap();
    let completed = execute_tile(
        provider,
        ProviderStochasticEvolutionTileRequest {
            plan: &plan,
            proposal: &proposal,
            source: &state_handle,
            noise: &final_noise,
            step: 0,
            offset: 64,
            pending: Some(first.pending()),
        },
    )
    .await;
    assert!(completed.is_complete());
    let completed = completed
        .into_completed_output(provider, &plan, &proposal, 0)
        .unwrap();
    assert!(!completed.shares_allocation_with(&state_handle));
    let actual = provider.download_numeric(&completed).await.unwrap();
    let oracle = |noise: f64| {
        let drift = match drift {
            ScalarLiteral::F32Bits(bits) => f64::from(f32::from_bits(bits)),
            ScalarLiteral::F64Bits(bits) => f64::from_bits(bits),
        };
        let scale = match scale {
            ScalarLiteral::F32Bits(bits) => f64::from(f32::from_bits(bits)),
            ScalarLiteral::F64Bits(bits) => f64::from_bits(bits),
        };
        let scaled = arithmetic.scale_noise().evaluate(scale, noise).unwrap();
        let delta = arithmetic.drift_add().evaluate(drift, scaled).unwrap();
        let factor = match arithmetic.exponential_class() {
            NumericClass::Single => f64::from((delta as f32).exp()),
            NumericClass::Double => delta.exp(),
            _ => unreachable!(),
        };
        arithmetic.state_product().evaluate(1.0, factor).unwrap()
    };
    match (actual.data, noise) {
        (HostNumericDataOwned::F32(actual), HostNumericDataOwned::F32(noise)) => {
            for (actual, noise) in actual.iter().zip(noise) {
                assert!((f64::from(*actual) - oracle(f64::from(noise))).abs() <= 2.0e-6);
            }
        }
        (HostNumericDataOwned::F64(actual), HostNumericDataOwned::F64(noise)) => {
            for (actual, noise) in actual.iter().zip(noise) {
                assert!((*actual - oracle(noise)).abs() <= 2.0e-12);
            }
        }
        (HostNumericDataOwned::F32(actual), HostNumericDataOwned::F64(noise)) => {
            for (actual, noise) in actual.iter().zip(noise) {
                assert!((f64::from(*actual) - oracle(noise)).abs() <= 2.0e-6);
            }
        }
        other => panic!("unexpected stochastic output classes {other:?}"),
    }
    first_noise.release().unwrap();
    final_noise.release().unwrap();
    state_handle.release().unwrap();
    completed.release().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "run explicitly on actual WGPU hardware"]
fn actual_stochastic_step_uses_supplied_noise_and_completes_a_tail() {
    pollster::block_on(actual_tail());
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
async fn browser_stochastic_step_uses_supplied_noise_and_completes_a_tail() {
    actual_tail().await;
}
