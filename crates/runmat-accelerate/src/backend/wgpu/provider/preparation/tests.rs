use super::*;
use runmat_accelerate_api::{
    ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity, ProviderRepresentation,
    ProviderStorage,
};

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
