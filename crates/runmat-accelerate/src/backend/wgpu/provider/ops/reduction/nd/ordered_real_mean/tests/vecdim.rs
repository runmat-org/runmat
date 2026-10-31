use super::*;
use runmat_accelerate_api::ProviderOrderedRealMeanStep;

#[test]
#[ignore = "requires an actual SHADER_F64 adapter and RUNMAT_WGPU_FORCE_PRECISION=f64"]
fn actual_ordered_vecdim_mean_keeps_double_scratch_then_publishes_single() {
    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("qualification requires a real WGPU adapter");
    assert_eq!(provider.precision, NumericPrecision::F64);
    let values: [f32; 40] = std::array::from_fn(|index| [1.0e8, 1.0, -1.0e8, 1.0][(index / 2) % 4]);
    let input = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&values),
            shape: &[2, 4, 5],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let weak_input = input.downgrade();
    let before_download = provider.telemetry_snapshot().download_bytes;
    let before_allocations = provider.allocations.record_count().unwrap();
    let (source, source_buffer_bytes) = provider.ordered_mean_live_source(&input).unwrap();
    let symbolic = provider
        .plan_ordered_real_mean(
            &source,
            source_buffer_bytes,
            &[1, 2],
            ProviderPrecision::F32,
            ProviderRealMeanNaNMode::Include,
        )
        .unwrap()
        .expect("live source and future source must share one derivation");
    let matched = provider
        .prepare_ordered_real_mean_against_recipe(&input, &symbolic)
        .unwrap()
        .expect("the same actual view must match the symbolic recipe");
    assert!(matched.validate_live_source(&input).unwrap());
    assert_eq!(matched.recipe(), &symbolic);
    assert_eq!(matched.steps(), symbolic.steps());
    drop(matched);
    let source_facts = ProviderOrderedRealMeanSource {
        representation: source.clone(),
        buffer_bytes: source_buffer_bytes,
    };
    let control = ProviderOrderedRealMeanControl {
        axes: symbolic.axes().to_vec(),
        output_precision: symbolic.output_precision(),
        nan_mode: symbolic.nan_mode(),
    };
    let resources = ProviderOrderedRealMeanResources {
        limits: symbolic.limits(),
        steps: symbolic.steps().to_vec(),
        incremental_peak_bytes: symbolic.requested_incremental_peak_bytes(),
        queue_submissions: symbolic.queue_submissions(),
    };
    let mut forged = Vec::new();

    let mut changed = source_facts.clone();
    changed.representation.element_type = ProviderElementType::F64;
    forged.push(ProviderOrderedRealMeanRecipe::new(
        changed,
        control.clone(),
        resources.clone(),
    ));
    let mut changed = source_facts.clone();
    changed.representation.shape = vec![2, 5, 4];
    forged.push(ProviderOrderedRealMeanRecipe::new(
        changed,
        control.clone(),
        resources.clone(),
    ));
    let mut changed = source_facts.clone();
    changed.buffer_bytes += 1;
    forged.push(ProviderOrderedRealMeanRecipe::new(
        changed,
        control.clone(),
        resources.clone(),
    ));

    let mut changed = control.clone();
    changed.nan_mode = ProviderRealMeanNaNMode::Omit;
    // A handle-free recipe proposes an operation; it cannot authenticate the
    // intent of a different call. Omit selects a distinct valid operation with
    // the same resources, whereas the physical mutations below are invalid.
    let alternate =
        ProviderOrderedRealMeanRecipe::new(source_facts.clone(), changed, resources.clone());
    assert_ne!(alternate, symbolic);
    let prepared_alternate = provider
        .prepare_ordered_real_mean_against_recipe(&input, &alternate)
        .unwrap()
        .expect("a distinct valid NaN mode has its own exact proposal");
    assert_eq!(prepared_alternate.recipe(), &alternate);
    drop(prepared_alternate);
    let mut changed = control.clone();
    changed.output_precision = ProviderPrecision::F64;
    forged.push(ProviderOrderedRealMeanRecipe::new(
        source_facts.clone(),
        changed,
        resources.clone(),
    ));
    let mut changed = control.clone();
    changed.axes = vec![1];
    forged.push(ProviderOrderedRealMeanRecipe::new(
        source_facts.clone(),
        changed,
        resources.clone(),
    ));

    let mut changed = resources.clone();
    changed.limits.max_buffer_size -= 1;
    forged.push(ProviderOrderedRealMeanRecipe::new(
        source_facts.clone(),
        control.clone(),
        changed,
    ));
    let mut changed = resources.clone();
    changed.queue_submissions += 1;
    forged.push(ProviderOrderedRealMeanRecipe::new(
        source_facts.clone(),
        control.clone(),
        changed,
    ));
    let first = &resources.steps[0];
    let step = |input_shape: Vec<usize>, output_precision, output_bytes, workgroups| {
        ProviderOrderedRealMeanStep::new(
            first.axis(),
            input_shape,
            first.output_shape().to_vec(),
            first.input_precision(),
            output_precision,
        )
        .with_physical_resources(
            first.input_bytes(),
            output_bytes,
            first.uniform_bytes(),
            workgroups,
        )
    };
    for changed_first in [
        step(
            first.input_shape().to_vec(),
            ProviderPrecision::F32,
            first.output_bytes(),
            first.workgroups(),
        ),
        step(
            vec![2, 5, 4],
            first.output_precision(),
            first.output_bytes(),
            first.workgroups(),
        ),
        step(
            first.input_shape().to_vec(),
            first.output_precision(),
            first.output_bytes() + 8,
            first.workgroups(),
        ),
        step(
            first.input_shape().to_vec(),
            first.output_precision(),
            first.output_bytes(),
            first.workgroups() + 1,
        ),
    ] {
        let mut changed = resources.clone();
        changed.steps[0] = changed_first;
        forged.push(ProviderOrderedRealMeanRecipe::new(
            source_facts.clone(),
            control.clone(),
            changed,
        ));
    }
    let mut changed = resources.clone();
    changed.incremental_peak_bytes += 1;
    forged.push(ProviderOrderedRealMeanRecipe::new(
        source_facts,
        control,
        changed,
    ));

    for (index, candidate) in forged.iter().enumerate() {
        assert!(
            provider
                .prepare_ordered_real_mean_against_recipe(&input, candidate)
                .unwrap()
                .is_none(),
            "mutated recipe {index} must be rejected"
        );
        assert!(provider
            .execute_ordered_real_mean(
                ProviderOrderedRealMean::new(input.clone(), candidate.clone(),),
                None
            )
            .is_err());
    }
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        before_allocations
    );
    let abandoned = provider
        .prepare_ordered_real_mean(
            &input,
            &[1, 2],
            ProviderPrecision::F32,
            ProviderRealMeanNaNMode::Include,
        )
        .unwrap()
        .unwrap();
    drop(abandoned);
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        before_allocations
    );
    input.allocation_lease().validate_live().unwrap();
    assert!(provider
        .prepare_ordered_real_mean(
            &input,
            &[2, 1],
            ProviderPrecision::F32,
            ProviderRealMeanNaNMode::Include,
        )
        .unwrap()
        .is_none());
    let foreign = WgpuProvider::new(WgpuProviderOptions::default()).unwrap();
    assert!(foreign
        .prepare_ordered_real_mean(
            &input,
            &[1, 2],
            ProviderPrecision::F32,
            ProviderRealMeanNaNMode::Include,
        )
        .is_err());
    let selected = provider
        .prepare_ordered_real_mean(
            &input,
            &[1, 2],
            ProviderPrecision::F32,
            ProviderRealMeanNaNMode::Include,
        )
        .unwrap()
        .expect("F64 owner must prepare both source-ordered axes");
    assert!(selected.validate_live_source(&input).unwrap());
    assert_eq!(selected.recipe(), &symbolic);
    assert_eq!(selected.steps().len(), 2);
    assert_eq!(
        selected.steps()[0].input_precision(),
        ProviderPrecision::F32
    );
    assert_eq!(
        selected.steps()[0].output_precision(),
        ProviderPrecision::F64
    );
    assert_eq!(selected.steps()[0].output_shape(), &[2, 1, 5]);
    assert_eq!(
        selected.steps()[1].input_precision(),
        ProviderPrecision::F64
    );
    assert_eq!(
        selected.steps()[1].output_precision(),
        ProviderPrecision::F32
    );
    assert_eq!(selected.steps()[1].output_shape(), &[2, 1, 1]);
    assert_eq!(selected.steps()[0].output_bytes(), 2 * 5 * 8);
    assert_eq!(selected.steps()[1].output_bytes(), 2 * 4);
    assert!(selected.requested_incremental_peak_bytes() >= 2 * 5 * 8 + 2 * 4);
    let executed = provider.execute_ordered_real_mean(selected, None).unwrap();
    assert_eq!(executed.selected_submissions, 2);
    assert_eq!(
        executed.observed_incremental_peak_bytes,
        executed.requested_incremental_peak_bytes
    );
    assert_eq!(executed.output.shape(), &[2, 1, 1]);
    assert_eq!(
        executed.output.physical().element_type(),
        NumericElementType::F32
    );
    assert_eq!(
        provider.telemetry_snapshot().download_bytes,
        before_download
    );
    input.allocation_lease().validate_live().unwrap();
    let output = block_on(provider.download_numeric_exec(&executed.output)).unwrap();
    let HostNumericDataOwned::F32(data) = output.data else {
        panic!("selected call must publish native Single");
    };
    assert_eq!(data, vec![0.5, 0.5]);
    let first_call = provider
        .prepare_ordered_real_mean(
            &input,
            &[1],
            ProviderPrecision::F32,
            ProviderRealMeanNaNMode::Include,
        )
        .unwrap()
        .unwrap();
    assert_eq!(first_call.steps().len(), 1);
    assert_eq!(
        first_call.steps()[0].output_precision(),
        ProviderPrecision::F32
    );
    let first_call = provider
        .execute_ordered_real_mean(first_call, None)
        .unwrap();
    let second_call = provider
        .prepare_ordered_real_mean(
            &first_call.output,
            &[2],
            ProviderPrecision::F32,
            ProviderRealMeanNaNMode::Include,
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        second_call.steps()[0].input_precision(),
        ProviderPrecision::F32
    );
    assert_eq!(
        second_call.steps()[0].output_precision(),
        ProviderPrecision::F32
    );
    let second_call = provider
        .execute_ordered_real_mean(second_call, None)
        .unwrap();
    assert_eq!(second_call.output.shape(), &[2, 1, 1]);
    assert!(!second_call
        .output
        .shares_allocation_with(&first_call.output));
    drop(executed);
    input.allocation_lease().validate_live().unwrap();
    drop(input);
    assert!(weak_input.acquire().is_err());
}

#[test]
#[ignore = "requires an actual SHADER_F64 adapter and RUNMAT_WGPU_FORCE_PRECISION=f64"]
fn future_mixed_power_output_supplies_provider_owned_variance_extent() {
    use runmat_accelerate_api::power_service::{
        execute_resident_scalar_power, plan_scalar_power_for_dense_output,
        prepare_resident_scalar_power_against_plan,
    };
    use runmat_accelerate_api::{
        PowerResultDomain, PowerScalarControl, ProviderOperationIdentity, RealPowerOperand,
    };

    let provider =
        crate::backend::wgpu::provider::register_wgpu_provider(WgpuProviderOptions::default())
            .expect("qualification requires a real WGPU adapter");
    assert_eq!(provider.precision(), ProviderPrecision::F64);
    let values = [1.25_f32; 40];
    let input = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&values),
            shape: &[2, 4, 5],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let (representation, _) = provider.ordered_mean_live_source(&input).unwrap();
    let square = plan_scalar_power_for_dense_output(
        provider,
        representation,
        PowerScalarControl::DoubleBits(2.0f64.to_bits()),
        RealPowerOperand::Base,
        &[1, 1],
        ProviderOperationIdentity::new(runmat_builtins::POWER_CATALOG_ENTRY.identity.name),
    )
    .unwrap()
    .expect("original Double square must have a selected provider route");
    assert!(matches!(
        square.mode(),
        runmat_accelerate_api::RealPowerArithmeticMode::MixedDoubleThenSingle { .. }
    ));
    let projected_bytes = provider
        .plan_power_output_buffer_bytes(&square, PowerResultDomain::Real)
        .unwrap()
        .expect("the selected provider must project its own output allocator");
    assert_eq!(projected_bytes, 160);
    let squared = square.physical_work(PowerResultDomain::Real).unwrap();
    let variance = provider
        .plan_ordered_real_mean(
            &squared.output,
            projected_bytes,
            &[1, 2],
            ProviderPrecision::F32,
            ProviderRealMeanNaNMode::Include,
        )
        .unwrap()
        .expect("variance consumes the selected POWER publication class and extent");
    assert_eq!(variance.source(), &squared.output);
    assert_eq!(variance.source_buffer_bytes(), projected_bytes);
    assert_eq!(
        variance.steps()[0].output_precision(),
        ProviderPrecision::F64
    );
    assert_eq!(
        variance.steps()[1].output_precision(),
        ProviderPrecision::F32
    );
    let prepared = prepare_resident_scalar_power_against_plan(
        provider,
        &input,
        &square,
        PowerScalarControl::DoubleBits(2.0f64.to_bits()),
        RealPowerOperand::Base,
        &[1, 1],
        ProviderOperationIdentity::new(runmat_builtins::POWER_CATALOG_ENTRY.identity.name),
    )
    .expect("actual source must revalidate the exact symbolic square");
    let actual_square = block_on(execute_resident_scalar_power(prepared))
        .expect("actual mixed square on the selected provider");
    assert_eq!(
        actual_square.handle.physical().requested_bytes(),
        projected_bytes
    );
    let actual_variance = provider
        .prepare_ordered_real_mean_against_recipe(&actual_square.handle, &variance)
        .unwrap()
        .expect("future variance recipe must match the produced square view");
    assert_eq!(actual_variance.recipe(), &variance);
    let mean = provider
        .execute_ordered_real_mean(actual_variance, None)
        .unwrap();
    assert_eq!(mean.output.shape(), &[2, 1, 1]);
    assert_eq!(
        mean.output.physical().element_type(),
        NumericElementType::F32
    );
    input.allocation_lease().validate_live().unwrap();
}
