use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, GpuTensorStorage, HostTensorView,
    NumericElementType, ProviderExecutionFuture, ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

fn fixture_ordered_recipe(
    source: &GpuTensorHandle,
    steps: Vec<runmat_accelerate_api::ProviderOrderedRealMeanStep>,
    nan_mode: ProviderRealMeanNaNMode,
    incremental_peak_bytes: u64,
) -> runmat_accelerate_api::ProviderOrderedRealMeanRecipe {
    use runmat_accelerate_api::{
        ProviderElementType, ProviderLayout, ProviderOrderedRealMeanControl,
        ProviderOrderedRealMeanLimits, ProviderOrderedRealMeanRecipe,
        ProviderOrderedRealMeanResources, ProviderOrderedRealMeanSource, ProviderRepresentation,
        ProviderResidency, ProviderStorage,
    };

    let element_type = match runmat_accelerate_api::handle_precision(source)
        .expect("fixture source has a real physical class")
    {
        ProviderPrecision::F32 => ProviderElementType::F32,
        ProviderPrecision::F64 => ProviderElementType::F64,
    };
    let axes = steps.iter().map(|step| step.axis()).collect();
    let output_precision = steps
        .last()
        .expect("fixture has a selected axis")
        .output_precision();
    let queue_submissions = u32::try_from(steps.len()).expect("test steps fit u32");
    ProviderOrderedRealMeanRecipe::new(
        ProviderOrderedRealMeanSource {
            representation: ProviderRepresentation {
                element_type,
                storage: ProviderStorage::DenseReal,
                layout: ProviderLayout::ColumnMajorContiguous,
                shape: source.shape().iter().map(|&dim| dim as u64).collect(),
                residency: ProviderResidency::Device,
            },
            buffer_bytes: source.physical().requested_bytes(),
        },
        ProviderOrderedRealMeanControl {
            axes,
            output_precision,
            nan_mode,
        },
        ProviderOrderedRealMeanResources {
            limits: ProviderOrderedRealMeanLimits {
                max_buffer_size: u64::MAX,
                max_storage_buffer_binding_size: u32::MAX,
                max_uniform_buffer_binding_size: u32::MAX,
                max_compute_invocations_per_workgroup: u32::MAX,
                max_compute_workgroup_size_x: u32::MAX,
                max_compute_workgroups_per_dimension: u32::MAX,
            },
            steps,
            incremental_peak_bytes,
            queue_submissions,
        },
    )
}

struct Provider(TestAllocationOwner);
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }
    fn device_info(&self) -> String {
        "mean reduction publication fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unused upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unused download") })
    }
    fn fill(
        &self,
        shape: &[usize],
        value: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<GpuTensorHandle> = (|| {
            Ok(self
                .0
                .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
                    data: runmat_accelerate_api::HostNumericDataOwned::F64(vec![
                        value;
                        if shape.contains(&0) {
                            0
                        } else {
                            shape.iter().product()
                        }
                    ]),
                    shape: shape.to_vec(),
                    storage: GpuTensorStorage::Real,
                })?
                .handle)
        })();
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
}

#[test]
fn class_adaptation_unsupported_is_terminal_only_after_a_completed_reduction() {
    let provider: &'static Provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .0
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let reduced = provider
        .0
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let before = futures::executor::block_on(adapt_output_class(
        provider,
        &input.handle,
        reduced.handle.clone(),
        &OutputTemplate::Native,
        ClassAdaptationStage::BeforeReduction,
    ));
    assert!(matches!(before, Err(MeanProviderFailure::Unsupported)));
    let after = futures::executor::block_on(adapt_output_class(
        provider,
        &input.handle,
        reduced.handle,
        &OutputTemplate::Native,
        ClassAdaptationStage::AfterReduction,
    ));
    let Err(MeanProviderFailure::Execution(error)) = after else {
        panic!("completed reduction must not authorize numerical replay")
    };
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(error.source.is_some(), "typed provider cause must survive");
    input.handle.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn scalar_vecdim_keeps_the_original_signed_zero_allocation() {
    let provider: &'static Provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .0
        .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
            data: runmat_accelerate_api::HostNumericDataOwned::F32(vec![-0.0]),
            shape: vec![1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let result = futures::executor::block_on(execute(
        provider,
        &input.handle,
        &MeanAxes::Vec(vec![1, 2]),
        ReductionNaN::Include,
        &OutputTemplate::Native,
    ))
    .unwrap()
    .unwrap();
    assert!(result.shares_allocation_with(&input.handle));
    assert_eq!(result.shape(), &[1, 1]);
    assert_eq!(input.probe.retirements(), 0);
}

struct OrderedFailureProvider {
    owner: TestAllocationOwner,
    fallback_calls: AtomicUsize,
}

impl AccelProvider for OrderedFailureProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "ordered mean effect boundary fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unused upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unused download") })
    }
    fn fill(
        &self,
        _: &[usize],
        _: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<GpuTensorHandle> = Err(anyhow::anyhow!("unused fill"));
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
    fn prepare_ordered_real_mean(
        &self,
        source: &GpuTensorHandle,
        _: &[usize],
        _: ProviderPrecision,
        nan_mode: ProviderRealMeanNaNMode,
    ) -> anyhow::Result<Option<runmat_accelerate_api::ProviderOrderedRealMean>> {
        let step = runmat_accelerate_api::ProviderOrderedRealMeanStep::new(
            0,
            vec![2, 2],
            vec![1, 2],
            ProviderPrecision::F32,
            ProviderPrecision::F64,
        )
        .with_physical_resources(16, 16, 16, 1);
        Ok(Some(runmat_accelerate_api::ProviderOrderedRealMean::new(
            source.clone(),
            fixture_ordered_recipe(source, vec![step], nan_mode, 32),
        )))
    }
    fn execute_ordered_real_mean(
        &self,
        _: runmat_accelerate_api::ProviderOrderedRealMean,
        _admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        runmat_accelerate_api::ProviderExecutedOrderedRealMean,
    > {
        Err(runmat_accelerate_api::ProviderExecutionError::from(
            runmat_accelerate_api::unsupported_provider_operation("second selected axis failed"),
        )
        .after_effect())
    }
    fn reduce_mean_nd<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a [usize],
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        self.fallback_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let result: anyhow::Result<GpuTensorHandle> =
                async move { anyhow::bail!("selected route must not replay") }.await;
            result.map(Some).map_err(|source| {
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
            })
        })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn prepared_ordered_mean_failure_after_effect_cannot_replay_old_nd_route() {
    let provider: &'static OrderedFailureProvider = Box::leak(Box::new(OrderedFailureProvider {
        owner: TestAllocationOwner::new(),
        fallback_calls: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let failure = futures::executor::block_on(execute(
        provider,
        &input.handle,
        &MeanAxes::Vec(vec![1, 2]),
        ReductionNaN::Include,
        &OutputTemplate::Native,
    ))
    .unwrap_err();
    assert_eq!(failure.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(failure.source.is_some());
    assert_eq!(provider.fallback_calls.load(Ordering::SeqCst), 0);
    assert_eq!(input.probe.retirements(), 0);
    input.handle.allocation_lease().validate_live().unwrap();
}

struct OrderedPublicationProvider(TestAllocationOwner);

impl AccelProvider for OrderedPublicationProvider {
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }
    fn device_info(&self) -> String {
        "ordered mean publication fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unused upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unused download") })
    }
    fn fill(
        &self,
        _: &[usize],
        _: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<GpuTensorHandle> = Err(anyhow::anyhow!("unused fill"));
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
    fn prepare_ordered_real_mean(
        &self,
        source: &GpuTensorHandle,
        axes: &[usize],
        output_precision: ProviderPrecision,
        nan_mode: ProviderRealMeanNaNMode,
    ) -> anyhow::Result<Option<runmat_accelerate_api::ProviderOrderedRealMean>> {
        if source.shape() != [1, 1, 2] || axes != [2] {
            return Ok(None);
        }
        let input_precision = runmat_accelerate_api::handle_precision(source)
            .expect("fixture source has a real physical class");
        let output_bytes = match output_precision {
            ProviderPrecision::F32 => 4,
            ProviderPrecision::F64 => 8,
        };
        let step = runmat_accelerate_api::ProviderOrderedRealMeanStep::new(
            2,
            source.shape().to_vec(),
            vec![1, 1, 1],
            input_precision,
            output_precision,
        )
        .with_physical_resources(source.physical().requested_bytes(), output_bytes, 16, 1);
        Ok(Some(runmat_accelerate_api::ProviderOrderedRealMean::new(
            source.clone(),
            fixture_ordered_recipe(source, vec![step], nan_mode, output_bytes + 16),
        )))
    }
    fn execute_ordered_real_mean(
        &self,
        selected: runmat_accelerate_api::ProviderOrderedRealMean,
        _admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        runmat_accelerate_api::ProviderExecutedOrderedRealMean,
    > {
        let precision = selected.steps()[0].output_precision();
        let data = match precision {
            ProviderPrecision::F32 => runmat_accelerate_api::HostNumericDataOwned::F32(vec![4.0]),
            ProviderPrecision::F64 => runmat_accelerate_api::HostNumericDataOwned::F64(vec![4.0]),
        };
        let output = self
            .0
            .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
                data,
                shape: vec![1, 1, 1],
                storage: GpuTensorStorage::Real,
            })
            .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
        let requested_incremental_peak_bytes = selected.requested_incremental_peak_bytes();
        Ok(runmat_accelerate_api::ProviderExecutedOrderedRealMean {
            output: output.handle,
            requested_incremental_peak_bytes,
            observed_incremental_peak_bytes: requested_incremental_peak_bytes,
            observed_output_bytes: selected.steps()[0].output_bytes(),
            selected_submissions: 1,
        })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn default_dim_and_vec_one_element_means_publish_the_shared_public_shape() {
    let provider: &'static OrderedPublicationProvider = Box::leak(Box::new(
        OrderedPublicationProvider(TestAllocationOwner::new()),
    ));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let controls = [
        Vec::new(),
        vec![Value::Num(3.0)],
        vec![Value::Tensor(
            Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap(),
        )],
    ];
    for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
        for axes in &controls {
            let data = match precision {
                ProviderPrecision::F32 => {
                    runmat_accelerate_api::HostNumericDataOwned::F32(vec![3.0, 5.0])
                }
                ProviderPrecision::F64 => {
                    runmat_accelerate_api::HostNumericDataOwned::F64(vec![3.0, 5.0])
                }
            };
            let input = provider
                .0
                .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
                    data,
                    shape: vec![1, 1, 2],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap();
            let mut arguments = axes.clone();
            arguments.push(Value::from("native"));
            let result = futures::executor::block_on(mean_builtin(
                Value::GpuTensor(input.handle.clone()),
                crate::builtins::common::test_support::builtin_arguments(arguments.clone()),
            ))
            .unwrap();
            let Value::GpuTensor(result) = result else {
                panic!("successful resident mean must retain the selected owner");
            };
            assert_eq!(result.shape(), &[1, 1]);
            assert_eq!(
                runmat_accelerate_api::handle_precision(&result),
                Some(precision)
            );
            assert!(!result.shares_allocation_with(&input.handle));
            input.handle.allocation_lease().validate_live().unwrap();
            assert_eq!(input.probe.retirements(), 0);

            let host = match precision {
                ProviderPrecision::F32 => {
                    Value::Tensor(Tensor::from_f32(vec![3.0, 5.0], vec![1, 1, 2]).unwrap())
                }
                ProviderPrecision::F64 => {
                    Value::Tensor(Tensor::new(vec![3.0, 5.0], vec![1, 1, 2]).unwrap())
                }
            };
            let host = futures::executor::block_on(mean_builtin(
                host,
                crate::builtins::common::test_support::builtin_arguments(arguments),
            ))
            .unwrap();
            match (precision, host) {
                (ProviderPrecision::F32, Value::Tensor(tensor)) => {
                    assert_eq!(tensor.shape(), &[1, 1]);
                    assert_eq!(tensor.materialize_f64(), vec![4.0]);
                }
                (ProviderPrecision::F64, Value::Num(value)) => assert_eq!(value, 4.0),
                other => panic!("unexpected one-element host mean: {other:?}"),
            }
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn empty_mean_keeps_non_scalar_shape_and_nan_publication() {
    assert_eq!(mean_public_shape(&[1, 0, 2], 0), vec![1, 0, 2]);
    for single in [false, true] {
        let input = if single {
            Value::Tensor(Tensor::from_f32(Vec::new(), vec![1, 0, 2]).unwrap())
        } else {
            Value::Tensor(Tensor::new(Vec::new(), vec![1, 0, 2]).unwrap())
        };
        let output = futures::executor::block_on(mean_builtin(
            input,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
        ))
        .unwrap();
        let Value::Tensor(output) = output else {
            panic!("empty-source mean with two output elements remains a tensor");
        };
        assert_eq!(output.shape(), &[1, 1, 2]);
        assert!(output.materialize_f64().iter().all(|value| value.is_nan()));
    }
}

#[test]
fn publication_rejects_wrong_class_shape_and_physical_alias_and_retires_output() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .0
        .allocate_zeroed(vec![2, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    for (shape, class) in [
        (vec![1, 2], NumericElementType::F64),
        (vec![2, 1], NumericElementType::F32),
    ] {
        let output = provider
            .0
            .allocate_zeroed(shape, class, GpuTensorStorage::Real)
            .unwrap();
        assert!(admit_result(
            provider,
            &input.handle,
            output.handle,
            &MeanAxes::Dim(1),
            &[0],
            Some(ProviderPrecision::F32),
        )
        .is_err());
        assert_eq!(output.probe.retirements(), 1);
    }
    let alias = input
        .handle
        .with_view(input.handle.view().clone().with_shape(vec![1, 2]))
        .unwrap();
    assert!(admit_result(
        provider,
        &input.handle,
        alias,
        &MeanAxes::Dim(1),
        &[0],
        Some(ProviderPrecision::F32)
    )
    .is_err());
    assert_eq!(input.probe.retirements(), 0);
    let output = provider
        .0
        .allocate_zeroed(vec![1, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let accepted = admit_result(
        provider,
        &input.handle,
        output.handle,
        &MeanAxes::Dim(1),
        &[0],
        Some(ProviderPrecision::F32),
    )
    .unwrap();
    assert_eq!(accepted.shape(), &[1, 2]);
    let identity = admit_result(
        provider,
        &accepted,
        accepted.clone(),
        &MeanAxes::Dim(1),
        &[0],
        Some(ProviderPrecision::F32),
    )
    .unwrap();
    assert!(identity.shares_allocation_with(&accepted));
    drop(identity);
    drop(accepted);
    assert_eq!(output.probe.retirements(), 1);
}

#[test]
fn empty_nd_strategy_preserves_unreduced_dimensions_without_permute() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .0
        .allocate_zeroed(
            vec![0, 3, 2],
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )
        .unwrap();
    // This provider has no permute hook. Empty shape resolution precedes it.
    let output =
        futures::executor::block_on(reduce_mean_vecdim_nd_gpu(provider, &input.handle, &[1]))
            .unwrap();
    assert_eq!(output.shape(), &[1, 3, 2]);
    assert_eq!(input.probe.retirements(), 0);
    let huge = u32::MAX as usize;
    let empty = provider
        .0
        .allocate_zeroed(
            vec![huge, huge, 2, 0],
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let output =
        futures::executor::block_on(reduce_mean_vecdim_nd_gpu(provider, &empty.handle, &[1]))
            .unwrap();
    assert_eq!(output.shape(), &[1, huge, 2, 0]);
    // Reducing the only zero extent would create an unrepresentable nonempty
    // output; reject its geometry before asking this fixture to allocate it.
    assert!(matches!(
        futures::executor::block_on(reduce_mean_vecdim_nd_gpu(provider, &empty.handle, &[4])),
        Err(MeanProviderFailure::Execution(_))
    ));
}

#[test]
fn omitnan_single_uses_double_reduction_then_restores_requested_class_resident() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let values = [16_777_216.0_f32, 1.0, 1.0, f32::NAN];
        let input = provider
            .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                data: runmat_accelerate_api::HostNumericDataView::F32(&values),
                shape: &[4, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        for (controls, expected_precision) in [
            (vec![Value::from("omitnan")], ProviderPrecision::F32),
            (
                vec![Value::from("omitnan"), Value::from("double")],
                ProviderPrecision::F64,
            ),
        ] {
            provider.reset_telemetry();
            let result = futures::executor::block_on(mean_builtin(
                Value::GpuTensor(input.clone()),
                crate::builtins::common::test_support::builtin_arguments(controls),
            ))
            .unwrap();
            let Value::GpuTensor(output) = result else {
                panic!("mean should remain resident")
            };
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(expected_precision)
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            let host =
                crate::builtins::common::test_support::gather(Value::GpuTensor(output)).unwrap();
            assert_eq!(host.materialize_f64(), vec![5_592_406.0]);
        }
        input.allocation_lease().validate_live().unwrap();
    });
}

#[test]
fn omitnan_stage_rejects_wrong_precision_and_borrowed_alias_before_division() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let original = provider
        .0
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let wrong = provider
        .0
        .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    assert!(admit_double_omitnan_stage(
        provider,
        &original.handle,
        wrong.handle,
        &[1, 1],
        OmitNanStageAlias::Fresh,
        &[&original.handle],
    )
    .is_err());
    assert_eq!(wrong.probe.retirements(), 1);

    let borrowed = provider
        .0
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    assert!(admit_double_omitnan_stage(
        provider,
        &original.handle,
        borrowed.handle.clone(),
        &[1, 1],
        OmitNanStageAlias::Fresh,
        &[&borrowed.handle],
    )
    .is_err());
    assert_eq!(borrowed.probe.retirements(), 0);
    original.handle.allocation_lease().validate_live().unwrap();
    borrowed.handle.allocation_lease().validate_live().unwrap();
}

#[test]
fn omitnan_singleton_double_stage_accepts_only_its_current_identity_view() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let original = provider
        .0
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let current = provider
        .0
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let earlier_same_branch = current.handle.clone();
    let accepted = admit_double_omitnan_stage(
        provider,
        &original.handle,
        current.handle.clone(),
        &[1, 1],
        OmitNanStageAlias::SingletonIdentity(&current.handle),
        &[&current.handle, &earlier_same_branch],
    )
    .unwrap();
    assert!(accepted.shares_allocation_with(&current.handle));
    assert_eq!(current.probe.retirements(), 0);

    let sibling = provider
        .0
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let sibling_alias = admit_double_omitnan_stage(
        provider,
        &original.handle,
        sibling.handle.clone(),
        &[1, 1],
        OmitNanStageAlias::SingletonIdentity(&current.handle),
        &[&current.handle, &sibling.handle],
    );
    assert!(matches!(
        sibling_alias,
        Err(MeanProviderFailure::Execution(_))
    ));
    assert_eq!(sibling.probe.retirements(), 0);

    let rejected = admit_double_omitnan_stage(
        provider,
        &original.handle,
        original.handle.clone(),
        &[1, 1],
        OmitNanStageAlias::SingletonIdentity(&current.handle),
        &[&original.handle, &current.handle],
    );
    assert!(matches!(rejected, Err(MeanProviderFailure::Execution(_))));
    assert_eq!(original.probe.retirements(), 0);
    original.handle.allocation_lease().validate_live().unwrap();
}

#[derive(Clone, Copy)]
enum MalformedPreparation {
    AliasedMap,
    WrongMaskClass,
}

struct PreparationProvider {
    owner: TestAllocationOwner,
    mode: MalformedPreparation,
    mask_calls: AtomicUsize,
    reduction_calls: AtomicUsize,
}

impl AccelProvider for PreparationProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "mean preparation admission fixture".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unexpected download") })
    }

    fn map_nan_to_zero(
        &self,
        input: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        match self.mode {
            MalformedPreparation::AliasedMap => Ok(input.clone()),
            MalformedPreparation::WrongMaskClass => Ok(self
                .owner
                .allocate_zeroed(
                    input.shape().to_vec(),
                    NumericElementType::F32,
                    GpuTensorStorage::Real,
                )?
                .handle),
        }
    }

    fn not_nan_mask(&self, input: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        self.mask_calls.fetch_add(1, Ordering::SeqCst);
        Ok(self
            .owner
            .allocate_zeroed(
                input.shape().to_vec(),
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )?
            .handle)
    }

    fn reduce_sum_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduction_calls.fetch_add(1, Ordering::SeqCst);
            Err(
                runmat_accelerate_api::unsupported_provider_operation("unexpected reduction")
                    .into(),
            )
        })
    }
}

#[test]
fn malformed_nan_preparation_stops_before_reduction_or_later_provider_hooks() {
    for mode in [
        MalformedPreparation::AliasedMap,
        MalformedPreparation::WrongMaskClass,
    ] {
        let provider: &'static PreparationProvider = Box::leak(Box::new(PreparationProvider {
            owner: TestAllocationOwner::new(),
            mode,
            mask_calls: AtomicUsize::new(0),
            reduction_calls: AtomicUsize::new(0),
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        let result = futures::executor::block_on(execute(
            provider,
            &input.handle,
            &MeanAxes::Dim(1),
            ReductionNaN::Omit,
            &OutputTemplate::Default,
        ));
        assert!(result.is_err());
        assert_eq!(provider.reduction_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            provider.mask_calls.load(Ordering::SeqCst),
            usize::from(matches!(mode, MalformedPreparation::WrongMaskClass))
        );
        input.handle.allocation_lease().validate_live().unwrap();
    }
}
