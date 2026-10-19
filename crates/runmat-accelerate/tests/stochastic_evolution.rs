#![cfg(feature = "wgpu")]

use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
use runmat_accelerate_api::observation::{
    observe_provider_future, observe_provider_resource_future, ProviderObservationSupport,
    ProviderResourcePhasePlan,
};
use runmat_accelerate_api::scalar_program::ScalarLiteral;
use runmat_accelerate_api::stochastic_evolution::{
    ProviderStochasticEvolutionProgress, ProviderStochasticEvolutionTileRequest,
    ProviderStochasticEvolutionWorkRequest, StochasticEvolutionArithmetic,
    PROVIDER_STOCHASTIC_EVOLUTION_STEP_OPERATION,
};
use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorView, NumericElementType, ProviderElementType, ProviderFeasibilityQuery,
    ProviderLayout, ProviderOperationFamily, ProviderPrecision, ProviderRepresentation,
    ProviderResidency, ProviderStorage, ProviderWorkload,
};
use runmat_types::NumericClass;

#[derive(Clone, Copy)]
struct RealCase {
    class: NumericClass,
    element: ProviderElementType,
    physical: NumericElementType,
    drift: ScalarLiteral,
    scale: ScalarLiteral,
    tolerance: f64,
}

fn representation(case: RealCase, len: usize) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type: case.element,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: vec![u64::try_from(len).unwrap(), 1],
        residency: ProviderResidency::Device,
    }
}

fn host_values(case: RealCase, values: Vec<f64>) -> HostNumericDataOwned {
    match case.class {
        NumericClass::Single => {
            HostNumericDataOwned::F32(values.into_iter().map(|value| value as f32).collect())
        }
        NumericClass::Double => HostNumericDataOwned::F64(values),
        _ => unreachable!("real stochastic case"),
    }
}

fn host_view<'a>(data: &'a HostNumericDataOwned, shape: &'a [usize]) -> HostNumericTensorView<'a> {
    HostNumericTensorView {
        data: match data {
            HostNumericDataOwned::F32(values) => HostNumericDataView::F32(values),
            HostNumericDataOwned::F64(values) => HostNumericDataView::F64(values),
            _ => unreachable!("real stochastic payload"),
        },
        shape,
        storage: GpuTensorStorage::Real,
    }
}

async fn execute_tile<'a>(
    provider: &'a dyn AccelProvider,
    request: ProviderStochasticEvolutionTileRequest<'a>,
) -> ProviderStochasticEvolutionProgress {
    let workload = request.plan.workload().expect("selected workload");
    let query = ProviderFeasibilityQuery {
        operation: runmat_accelerate_api::ProviderOperationIdentity::new(
            PROVIDER_STOCHASTIC_EVOLUTION_STEP_OPERATION,
        ),
        family: ProviderOperationFamily::Library,
        inputs: vec![request.plan.state.clone()],
        outputs: vec![request.plan.output.clone()],
        workload: ProviderWorkload {
            stochastic_evolution: Some(workload),
            ..ProviderWorkload::default()
        },
    };
    let mut operation = match provider.begin_operation_observation(&query) {
        ProviderObservationSupport::Observed(receipt) => receipt,
        ProviderObservationSupport::Unsupported => {
            panic!("selected stochastic execution requires provider observation")
        }
    };
    operation.require_provider_execution(provider.device_id(), query.operation.clone());
    let phase_bytes = request
        .proposal
        .uniform_buffer_bytes_per_tile
        .checked_add(request.proposal.completion_fence_bytes_per_tile)
        .and_then(|bytes| bytes.checked_add(request.proposal.observation_fence_bytes_per_tile))
        .expect("selected tile resource bound");
    let scope = operation
        .recorder()
        .begin_resource_scope(ProviderResourcePhasePlan::new([phase_bytes]).unwrap())
        .unwrap();
    scope.begin_next_phase().unwrap();
    let result = observe_provider_resource_future(
        Some(scope.clone()),
        observe_provider_future(
            Some(operation.recorder()),
            provider.execute_stochastic_evolution_tile(request, None, None),
        ),
    )
    .await
    .expect("selected stochastic tile execution");
    scope.finish().unwrap();
    operation.publish().expect("publish selected tile");
    result
}

async fn run_zero_scale_case(provider: &dyn AccelProvider, case: RealCase) {
    const LEN: usize = 65;
    const STEPS: u32 = 4;
    let initial_values: Vec<f64> = (1..=LEN).map(|value| value as f64).collect();
    let initial_data = host_values(case, initial_values.clone());
    let mut state = provider
        .upload_numeric(&host_view(&initial_data, &[LEN, 1]))
        .expect("upload typed state");
    let arithmetic =
        StochasticEvolutionArithmetic::new(case.class, case.class, case.drift, case.scale)
            .expect("stable real arithmetic");
    let work = ProviderStochasticEvolutionWorkRequest {
        state: representation(case, LEN),
        output: representation(case, LEN),
        first_arithmetic: arithmetic,
        continuing_arithmetic: arithmetic,
        steps: STEPS,
    };
    let selection = provider
        .plan_stochastic_evolution(&work)
        .expect("provider planning")
        .expect("typed stochastic route");
    assert!(selection.validate_for(&work));
    assert_ne!(
        u64::try_from(LEN).unwrap() % u64::from(selection.proposal.frozen_workgroup_size),
        0,
        "the public numerical matrix must exercise a real dispatch tail"
    );

    for step in 0..STEPS {
        let mut offset = 0_u64;
        let mut progress: Option<ProviderStochasticEvolutionProgress> = None;
        while offset < LEN as u64 {
            let count = selection.plan.noise_tile_elements.min(LEN as u64 - offset);
            let count_usize = usize::try_from(count).unwrap();
            let noise_data = host_values(case, vec![7.0; count_usize]);
            let noise = provider
                .upload_numeric(&host_view(&noise_data, &[count_usize, 1]))
                .expect("upload supplied noise tile");
            let next = execute_tile(
                provider,
                ProviderStochasticEvolutionTileRequest {
                    plan: &selection.plan,
                    proposal: &selection.proposal,
                    source: &state,
                    noise: &noise,
                    step,
                    offset,
                    pending: progress.as_ref().map(|value| value.pending()),
                },
            )
            .await;
            noise.release().expect("release supplied noise tile");
            offset = next.pending().next_offset();
            progress = Some(next);
        }
        let next_state = progress
            .expect("at least one selected tile")
            .into_completed_output(provider, &selection.plan, &selection.proposal, step)
            .expect("complete selected step");
        assert_eq!(next_state.physical().element_type(), case.physical);
        assert_eq!(next_state.shape(), [LEN, 1]);
        state.release().expect("release prior state");
        state = next_state;
    }

    let gathered = provider
        .download_numeric(&state)
        .await
        .expect("download evolved state");
    let actual: Vec<f64> = match gathered.data {
        HostNumericDataOwned::F32(values) => values.into_iter().map(f64::from).collect(),
        HostNumericDataOwned::F64(values) => values,
        other => panic!("unexpected evolved class: {other:?}"),
    };
    for (index, value) in actual.into_iter().enumerate() {
        let mut expected = initial_values[index];
        for step in 0..STEPS {
            let recipe = selection
                .plan
                .arithmetic_for_step(step)
                .expect("selected arithmetic step");
            let scale = match recipe.scale() {
                ScalarLiteral::F32Bits(bits) => f64::from(f32::from_bits(bits)),
                ScalarLiteral::F64Bits(bits) => f64::from_bits(bits),
            };
            let drift = match recipe.drift() {
                ScalarLiteral::F32Bits(bits) => f64::from(f32::from_bits(bits)),
                ScalarLiteral::F64Bits(bits) => f64::from_bits(bits),
            };
            let scaled = recipe.scale_noise().evaluate(scale, 7.0).unwrap();
            let delta = recipe.drift_add().evaluate(drift, scaled).unwrap();
            let factor = match recipe.exponential_class() {
                NumericClass::Single => f64::from((delta as f32).exp()),
                NumericClass::Double => delta.exp(),
                _ => unreachable!("real stochastic exponential"),
            };
            expected = recipe.state_product().evaluate(expected, factor).unwrap();
        }
        assert!(
            (value - expected).abs() <= case.tolerance * expected.abs().max(1.0),
            "class {:?}, lane {index}: {value} != {expected}",
            case.class
        );
    }
    state.release().expect("release final state");
}

#[tokio::test]
async fn supplied_noise_stochastic_evolution_preserves_class_zero_scale_and_tail() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let single = RealCase {
        class: NumericClass::Single,
        element: ProviderElementType::F32,
        physical: NumericElementType::F32,
        drift: ScalarLiteral::F32Bits(0.05_f32.to_bits()),
        scale: ScalarLiteral::F32Bits(0.0_f32.to_bits()),
        tolerance: 2.0e-5,
    };
    run_zero_scale_case(provider, single).await;
    if provider.precision() == ProviderPrecision::F64 {
        run_zero_scale_case(
            provider,
            RealCase {
                class: NumericClass::Double,
                element: ProviderElementType::F64,
                physical: NumericElementType::F64,
                drift: ScalarLiteral::F64Bits(0.05_f64.to_bits()),
                scale: ScalarLiteral::F64Bits(0.0_f64.to_bits()),
                tolerance: 1.0e-10,
            },
        )
        .await;
    }
}
