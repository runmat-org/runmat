use super::*;
use runmat_accelerate_api::{
    ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity, ProviderRepresentation,
    ProviderStorage,
};

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
    let output = provider
        .floating_cast_exec(&input, NumericPrecision::F32)
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
    let bindings = provider.cached_fusion_bind_group_layout(1);
    let first = provider.recipe_pipeline(&recipe, &bindings, "preparation-qualification");
    let second = provider.recipe_pipeline(&recipe, &bindings, "preparation-qualification");
    assert!(std::sync::Arc::ptr_eq(&first, &second));
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
        provider.recipe_pipeline(&recipe, &bindings, "poisoned-cache-qualification")
    });
    assert!(!std::sync::Arc::ptr_eq(&first, &uncached));
    assert!(matches!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::ProviderFailed)
    ));
}
