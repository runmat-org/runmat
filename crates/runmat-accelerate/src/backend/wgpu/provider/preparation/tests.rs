use super::*;
use runmat_accelerate_api::{
    ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity, ProviderRepresentation,
    ProviderStorage,
};

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn preparation_key_warmth_uses_the_actual_route_cache() {
    let key = 47;
    assert!(!preparation_key_is_warm(key, None, false, true));
    assert!(preparation_key_is_warm(key, None, true, false));
    assert!(preparation_key_is_warm(key, Some(key), false, true));
    assert!(!preparation_key_is_warm(key, Some(key), true, false));
}

#[test]
#[ignore = "requires an actual shader-f64 WGPU adapter; execute explicitly for qualification"]
fn sum_estimate_and_preparation_follow_the_same_selected_256_or_512_workgroup() {
    use runmat_accelerate_api::{AccelProvider, ProviderSumWorkload, ProviderWorkload};
    use runmat_types::{NumericClass, NumericDomain, ReductionNaN, SumOutputClass};

    let mut provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    assert!(
        provider
            .adapter_limits
            .max_compute_invocations_per_workgroup
            >= 512
    );
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(runmat_builtins::SUM_CATALOG_ENTRY.identity.name),
        family: ProviderOperationFamily::Reduction,
        inputs: vec![ProviderRepresentation {
            element_type: ProviderElementType::I64,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![2, 2, 2],
            residency: ProviderResidency::Host,
        }],
        outputs: vec![],
        workload: ProviderWorkload {
            elements: Some(8),
            sum: Some(ProviderSumWorkload {
                invocation: runmat_types::SumInvocation::Sum,
                axes: vec![1],
                missing: ReductionNaN::Include,
                output: SumOutputClass::Double,
                input_class: NumericClass::Int64,
                input_domain: NumericDomain::Real,
                accumulation_class: NumericClass::Double,
                publication_class: NumericClass::Double,
            }),
            ..Default::default()
        },
    };
    provider.workgroup_config.scalar = 256;
    let geometry_256 = provider.sum_geometry().unwrap();
    let key_256 = provider.nd_sum_pipeline_key();
    let preparation_256 = provider.sum_preparation_snapshot(&query).unwrap();
    provider.workgroup_config.scalar = 512;
    let geometry_512 = provider.sum_geometry().unwrap();
    let key_512 = provider.nd_sum_pipeline_key();
    let preparation_512 = provider.sum_preparation_snapshot(&query).unwrap();
    for (selected, geometry) in [(256, geometry_256), (512, geometry_512)] {
        let expected = crate::backend::wgpu::config::elementwise_chunk_capacity(selected).unwrap();
        assert_eq!(geometry.nd_chunk_elements, expected);
        assert_eq!(geometry.cast_chunk_elements, expected);
        assert_eq!(geometry.integer_cast_chunk_elements, expected);
    }
    assert_ne!(
        geometry_256.nd_chunk_elements,
        geometry_512.nd_chunk_elements
    );
    assert_ne!(key_256, key_512);
    assert_ne!(preparation_256.identity, preparation_512.identity);
}

#[test]
#[ignore = "requires an actual shader-f64 WGPU adapter; execute explicitly for qualification"]
fn actual_integer_sum_conversion_preparation_tracks_its_distinct_pipeline() {
    use runmat_accelerate_api::{
        HostIntegerDataView, HostIntegerTensorView, HostNumericDataOwned, ProviderSumWorkload,
        ProviderWorkload,
    };
    use runmat_types::{NumericClass, NumericDomain, ReductionNaN, SumOutputClass};

    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(runmat_builtins::SUM_CATALOG_ENTRY.identity.name),
        family: ProviderOperationFamily::Reduction,
        inputs: vec![ProviderRepresentation {
            element_type: ProviderElementType::I64,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![2, 3],
            residency: ProviderResidency::Host,
        }],
        outputs: vec![],
        workload: ProviderWorkload {
            elements: Some(6),
            sum: Some(ProviderSumWorkload {
                invocation: runmat_types::SumInvocation::Sum,
                axes: vec![0],
                missing: ReductionNaN::Include,
                output: SumOutputClass::Double,
                input_class: NumericClass::Int64,
                input_domain: NumericDomain::Real,
                accumulation_class: NumericClass::Double,
                publication_class: NumericClass::Double,
            }),
            ..Default::default()
        },
    };
    let before = provider.sum_preparation_snapshot(&query).unwrap();
    assert_eq!(before.state, Some(CandidatePreparationState::Cold));
    assert!(before.cold_pipelines >= 1);
    let input = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[1, 2, 3, 4, 5, 6]),
            shape: &[2, 3],
        })
        .unwrap();
    let output = futures::executor::block_on(provider.reduce_sum_dim(&input, 0)).unwrap();
    assert_eq!(
        futures::executor::block_on(provider.download_numeric(&output))
            .unwrap()
            .data,
        HostNumericDataOwned::F64(vec![3.0, 7.0, 11.0])
    );
    let after = provider.sum_preparation_snapshot(&query).unwrap();
    assert_eq!(after.identity, before.identity);
    assert_eq!(after.state, Some(CandidatePreparationState::Warm));
    assert_eq!(after.cold_pipelines, 0);
    output.release().unwrap();
    input.release().unwrap();

    let mut nd_query = query.clone();
    nd_query.inputs[0].shape = vec![2, 2, 2];
    nd_query.workload.elements = Some(8);
    nd_query.workload.sum.as_mut().unwrap().axes = vec![1];
    let nd_before = provider.sum_preparation_snapshot(&nd_query).unwrap();
    assert_eq!(nd_before.state, Some(CandidatePreparationState::Cold));
    let nd_input = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[1, 2, 3, 4, 5, 6, 7, 8]),
            shape: &[2, 2, 2],
        })
        .unwrap();
    let nd_output = futures::executor::block_on(provider.reduce_sum_dim(&nd_input, 1)).unwrap();
    assert_eq!(nd_output.shape(), &[2, 1, 2]);
    assert_eq!(
        futures::executor::block_on(provider.download_numeric(&nd_output))
            .unwrap()
            .data,
        HostNumericDataOwned::F64(vec![4.0, 6.0, 12.0, 14.0]),
    );
    let nd_after = provider.sum_preparation_snapshot(&nd_query).unwrap();
    assert_eq!(nd_after.identity, nd_before.identity);
    assert_eq!(nd_after.state, Some(CandidatePreparationState::Warm));
    assert_eq!(nd_after.cold_pipelines, 0);
    assert!(provider
        .validated_fusion_cache
        .lock()
        .unwrap()
        .contains_key(&provider.nd_sum_pipeline_key()));
    assert!(!provider
        .fused_pipeline_cache
        .lock()
        .unwrap()
        .contains_key(&provider.nd_sum_pipeline_key()));
    nd_output.release().unwrap();
    nd_input.release().unwrap();
}

#[test]
#[ignore = "requires an actual shader-f64 WGPU adapter; execute explicitly for qualification"]
fn actual_omit_sum_preparation_covers_map_and_reduction_pipelines() {
    use runmat_accelerate_api::{
        GpuTensorStorage, HostNumericDataView, HostNumericTensorView, ProviderSumWorkload,
        ProviderWorkload,
    };
    use runmat_types::{NumericClass, NumericDomain, ReductionNaN, SumOutputClass};

    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(runmat_builtins::SUM_CATALOG_ENTRY.identity.name),
        family: ProviderOperationFamily::Reduction,
        inputs: vec![ProviderRepresentation {
            element_type: ProviderElementType::F64,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![2, 3],
            residency: ProviderResidency::Host,
        }],
        outputs: vec![],
        workload: ProviderWorkload {
            elements: Some(6),
            sum: Some(ProviderSumWorkload {
                invocation: runmat_types::SumInvocation::Sum,
                axes: vec![0],
                missing: ReductionNaN::Omit,
                output: SumOutputClass::Double,
                input_class: NumericClass::Double,
                input_domain: NumericDomain::Real,
                accumulation_class: NumericClass::Double,
                publication_class: NumericClass::Double,
            }),
            ..Default::default()
        },
    };
    let before = provider.sum_preparation_snapshot(&query).unwrap();
    assert_eq!(before.state, Some(CandidatePreparationState::Cold));
    assert!(before.cold_pipelines >= 1);
    let nan_shader = crate::backend::wgpu::shaders::nan::map_source(NumericPrecision::F64);
    let nan_recipe = PipelineRecipe::fusion(
        &nan_shader,
        1,
        FusionPipelineLayout::Single,
        provider.workgroup_config.scalar,
    );
    let nan_key = nan_recipe.key();
    let validated_only = futures::executor::block_on(provider.prepare_validated_fusion_pipeline(
        &nan_recipe,
        1,
        FusionPipelineLayout::Single,
    ))
    .expect("scoped ordinary pair for the identical shader recipe");
    assert_eq!(validated_only.key, nan_key);
    assert!(provider
        .validated_fusion_cache
        .lock()
        .unwrap()
        .contains_key(&nan_key));
    assert!(!provider
        .fused_pipeline_cache
        .lock()
        .unwrap()
        .contains_key(&nan_key));
    let still_cold = provider.sum_preparation_snapshot(&query).unwrap();
    assert_eq!(still_cold.identity, before.identity);
    assert_eq!(still_cold.state, Some(CandidatePreparationState::Cold));
    assert_eq!(still_cold.cold_pipelines, before.cold_pipelines);
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0]),
            shape: &[2, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let mapped = provider.map_nan_to_zero(&input).unwrap();
    let reduced = futures::executor::block_on(provider.reduce_sum_dim(&mapped, 0)).unwrap();
    let after = provider.sum_preparation_snapshot(&query).unwrap();
    assert_eq!(after.identity, before.identity);
    assert_eq!(after.state, Some(CandidatePreparationState::Warm));
    assert_eq!(after.cold_pipelines, 0);
    assert!(after.warm_pipelines >= 2);
    reduced.release().unwrap();
    mapped.release().unwrap();
    input.release().unwrap();
}

#[test]
#[ignore = "requires an actual shader-f64 WGPU adapter; execute explicitly for qualification"]
fn actual_single_omit_sum_keeps_exact_class_and_value_across_all_stages() {
    use runmat_accelerate_api::{
        GpuTensorStorage, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
        ProviderSumWorkload, ProviderWorkload,
    };
    use runmat_types::{NumericClass, NumericDomain, ReductionNaN, SumOutputClass};

    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(runmat_builtins::SUM_CATALOG_ENTRY.identity.name),
        family: ProviderOperationFamily::Reduction,
        inputs: vec![ProviderRepresentation {
            element_type: ProviderElementType::F32,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![2, 3],
            residency: ProviderResidency::Host,
        }],
        outputs: vec![],
        workload: ProviderWorkload {
            elements: Some(6),
            sum: Some(ProviderSumWorkload {
                invocation: runmat_types::SumInvocation::Sum,
                axes: vec![0],
                missing: ReductionNaN::Omit,
                output: SumOutputClass::Native,
                input_class: NumericClass::Single,
                input_domain: NumericDomain::Real,
                accumulation_class: NumericClass::Double,
                publication_class: NumericClass::Single,
            }),
            ..Default::default()
        },
    };
    let before = provider.sum_preparation_snapshot(&query).unwrap();
    assert_eq!(before.state, Some(CandidatePreparationState::Cold));
    assert!(before.cold_pipelines >= 3);
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, f32::NAN, 3.0, 4.0, 5.0, 6.0]),
            shape: &[2, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let widened =
        futures::executor::block_on(provider.floating_cast_exec(&input, NumericPrecision::F64))
            .unwrap();
    let mapped = provider.map_nan_to_zero(&widened).unwrap();
    let reduced = futures::executor::block_on(provider.reduce_sum_dim(&mapped, 0)).unwrap();
    let published =
        futures::executor::block_on(provider.floating_cast_exec(&reduced, NumericPrecision::F32))
            .unwrap();
    assert_eq!(
        futures::executor::block_on(provider.download_numeric(&published))
            .unwrap()
            .data,
        HostNumericDataOwned::F32(vec![1.0, 7.0, 11.0])
    );
    let after = provider.sum_preparation_snapshot(&query).unwrap();
    assert_eq!(after.identity, before.identity);
    assert_eq!(after.state, Some(CandidatePreparationState::Warm));
    assert_eq!(after.cold_pipelines, 0);
    published.release().unwrap();
    reduced.release().unwrap();
    mapped.release().unwrap();
    widened.release().unwrap();
    input.release().unwrap();
}

fn single_query(source: ProviderElementType, shape: &[u64]) -> ProviderFeasibilityQuery {
    ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(
            runmat_builtins::SINGLE_CATALOG_ENTRY.identity.name,
        ),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![ProviderRepresentation {
            element_type: source,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            residency: ProviderResidency::Device,
            shape: shape.to_vec(),
        }],
        outputs: vec![],
        workload: runmat_accelerate_api::ProviderWorkload {
            preparation_route: ProviderPreparationProgram::FloatingCast {
                target: runmat_accelerate_api::ProviderPrecision::F32,
            }
            .route(),
            ..Default::default()
        },
    }
}

#[test]
#[ignore = "requires actual WGPU device; execute explicitly for qualification"]
fn actual_single_alias_and_empty_inventory_are_ready_even_when_cache_is_locked() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    let lock = provider.fused_pipeline_cache.lock().unwrap();
    for shape in [&[1, 8][..], &[0, 2, 3][..]] {
        let query = single_query(ProviderElementType::F32, shape);
        let mut unresolved = query.clone();
        unresolved.workload.preparation_route = None;
        let program = ProviderPreparationProgram::FloatingCast {
            target: runmat_accelerate_api::ProviderPrecision::F32,
        };
        assert!(provider
            .preparation_snapshot(&unresolved, program)
            .is_none());
        let cost_query = runmat_accelerate_api::ProviderCostQuery {
            operation: unresolved,
            context: Default::default(),
            preparation: CandidatePreparationState::Ready,
            required_upload_bytes: 0,
            required_download_bytes: 0,
            downstream_materialization: false,
        };
        assert!(
            provider.estimate_cost(&cost_query).is_none(),
            "unknown adaptation is not a zero-cost alias"
        );
        let snapshot = provider
            .preparation_snapshot(
                &query,
                ProviderPreparationProgram::FloatingCast {
                    target: runmat_accelerate_api::ProviderPrecision::F32,
                },
            )
            .unwrap();
        assert_eq!(snapshot.state, Some(CandidatePreparationState::Ready));
        assert_eq!((snapshot.cold_pipelines, snapshot.warm_pipelines), (0, 0));
    }
    drop(lock);
}

#[test]
#[ignore = "requires an actual shader-f64 adapter; execute explicitly for qualification"]
fn actual_single_cast_execution_populates_exact_preparation_key() {
    use runmat_accelerate_api::{GpuTensorStorage, HostNumericDataView, HostNumericTensorView};
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let query = single_query(ProviderElementType::F64, &[1, 2]);
    let program = ProviderPreparationProgram::FloatingCast {
        target: runmat_accelerate_api::ProviderPrecision::F32,
    };
    let cold = provider.preparation_snapshot(&query, program).unwrap();
    assert_eq!(cold.state, Some(CandidatePreparationState::Cold));
    assert_eq!(cold.cold_pipelines, 1);
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.25, 16_777_217.0]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let output =
        futures::executor::block_on(provider.floating_cast_exec(&input, NumericPrecision::F32))
            .unwrap();
    let warm = provider.preparation_snapshot(&query, program).unwrap();
    assert_eq!(warm.identity, cold.identity);
    assert_eq!(warm.state, Some(CandidatePreparationState::Warm));
    assert_eq!((warm.cold_pipelines, warm.warm_pipelines), (0, 1));
    input.release().unwrap();
    output.release().unwrap();
}

#[test]
#[ignore = "requires actual WGPU device; execute explicitly for qualification"]
fn actual_cache_queries_are_read_only_and_lazy_hits_reuse_exact_pipeline() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    let shader = "// preparation qualification\n@compute @workgroup_size(@WG@) fn main() {}";
    let value = ProviderRepresentation {
        element_type: ProviderElementType::from(AccelProvider::precision(&provider)),
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        residency: ProviderResidency::Device,
        shape: vec![1, 8],
    };
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("fusion.elementwise"),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![value.clone()],
        outputs: vec![value],
        workload: Default::default(),
    };
    let source = ProviderPreparationProgram::Elementwise { shader };
    let count = provider.fused_pipeline_cache.lock().unwrap().len();
    let cold = provider.preparation_snapshot(&query, source).unwrap();
    assert_eq!(cold.state, Some(CandidatePreparationState::Cold));
    assert_eq!(cold.cold_pipelines, 1);
    assert_eq!(provider.fused_pipeline_cache.lock().unwrap().len(), count);
    let recipe = PipelineRecipe::fusion(
        shader,
        1,
        FusionPipelineLayout::Single,
        provider.workgroup_config.scalar,
    );
    let (bindings, first, second) = provider
        .with_emission(|emission, _| -> anyhow::Result<_> {
            let bindings = provider.cached_fusion_bind_group_layout(emission, 1)?;
            let first = provider.recipe_pipeline(
                emission,
                &recipe,
                &bindings,
                "preparation-qualification",
            )?;
            let second = provider.recipe_pipeline(
                emission,
                &recipe,
                &bindings,
                "preparation-qualification",
            )?;
            Ok((bindings, first, second))
        })
        .unwrap()
        .unwrap();
    assert!(first.ptr_eq(&second));
    // A legacy shared-cache hit is not validation authority for the ordinary
    // fused hook. Seed its exact scoped, validated pair before claiming warm.
    let validated = futures::executor::block_on(provider.prepare_validated_fusion_pipeline(
        &recipe,
        1,
        FusionPipelineLayout::Single,
    ))
    .expect("actual scoped validation");
    assert_eq!(validated.key, recipe.key());
    let warm = provider.preparation_snapshot(&query, source).unwrap();
    assert_eq!(warm.identity, cold.identity);
    assert_eq!(warm.state, Some(CandidatePreparationState::Warm));
    assert_eq!(warm.cold_pipelines, 0);
    let lock = provider.fused_pipeline_cache.lock().unwrap();
    assert_eq!(
        provider.preparation_snapshot(&query, source).unwrap().state,
        None
    );
    drop(lock);
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = provider.fused_pipeline_cache.lock().unwrap();
        panic!("injected cache poison");
    }));
    assert!(poisoned.is_err());
    assert_eq!(
        provider.preparation_snapshot(&query, source).unwrap().state,
        None
    );
    use runmat_accelerate_api::observation::{
        with_provider_observation, ProviderObservationFailure, ProviderObservationStatus,
        ProviderOperationReceipt,
    };
    let mut receipt = ProviderOperationReceipt::new();
    let uncached = with_provider_observation(Some(receipt.recorder()), || {
        provider
            .with_emission(|emission, _| {
                provider.recipe_pipeline(
                    emission,
                    &recipe,
                    &bindings,
                    "poisoned-cache-qualification",
                )
            })
            .unwrap()
            .unwrap()
    });
    assert!(!first.ptr_eq(&uncached));
    assert!(matches!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::ProviderFailed)
    ));
}
