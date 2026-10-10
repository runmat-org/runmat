//! Deterministic cast authority/lifetime seams, not GPU numerical evidence.
use super::*;
use std::sync::atomic::Ordering;

fn cast(target: ProviderPrecision) -> ProviderRealElementwiseOperation {
    ProviderRealElementwiseOperation::FloatingCast { target }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn target_projection_and_zero_dispatch_alias_are_separate_authorities() {
    let fixture = new_fixture();
    let alias = ProviderRealElementwiseRecipe::plan(fixture, request(cast(ProviderPrecision::F32)))
        .unwrap()
        .unwrap();
    let convert =
        ProviderRealElementwiseRecipe::plan(fixture, request(cast(ProviderPrecision::F64)))
            .unwrap()
            .unwrap();
    let sqrt = ProviderRealElementwiseRecipe::plan(
        fixture,
        request(ProviderRealElementwiseOperation::Sqrt),
    )
    .unwrap()
    .unwrap();
    assert_eq!(alias.output().element_type, ProviderElementType::F32);
    assert_eq!(convert.output().element_type, ProviderElementType::F64);
    assert_eq!(convert.output().shape, alias.output().shape);
    assert_eq!(
        alias.physical().output_route,
        ProviderRealElementwiseOutputRoute::SameClassFloatingCastAlias
    );
    assert_eq!(alias.physical().requested_incremental_peak_bytes(), Some(0));
    assert_eq!(alias.physical().pipeline_key, None);
    assert_ne!(alias.identity().unwrap(), convert.identity().unwrap());
    assert_ne!(alias.identity().unwrap(), sqrt.identity().unwrap());
    let source = input(fixture);
    let published = ready(alias.execute(&source.handle)).unwrap();
    assert!(published.output.shares_allocation_with(&source.handle));
    assert_eq!(
        published.observed,
        ProviderRealElementwiseObserved {
            output_buffer_bytes: Some(8),
            ..Default::default()
        }
    );
    drop(published);
    assert_eq!(source.probe.retirements(), 0);
    let fresh = ready(convert.execute(&source.handle)).unwrap();
    assert_eq!(
        fresh.output.physical().element_type(),
        NumericElementType::F64
    );
    assert_eq!(fresh.output.physical().requested_bytes(), 16);
    assert!(!fresh.output.shares_allocation_with(&source.handle));
    drop(fresh);
    assert_eq!(fixture.produced.lock().unwrap()[0].retirements(), 1);
    source.handle.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn conversion_alias_wrong_class_shape_and_receipts_are_terminal_without_retiring_source() {
    for mode in [1, 2, 3, 5, 6, 7, 8, 9, 10] {
        let fixture = new_fixture();
        fixture.mode.store(mode, Ordering::Relaxed);
        let recipe =
            ProviderRealElementwiseRecipe::plan(fixture, request(cast(ProviderPrecision::F64)))
                .unwrap()
                .unwrap();
        let source = input(fixture);
        let error = ready(recipe.execute_protected(&source.handle, &[&source.handle]))
            .err()
            .unwrap();
        assert!(error.is_after_effect());
        if mode == 3 {
            assert!(std::error::Error::source(&error)
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .is_some());
            assert!(error.to_string().contains("original hook cause"));
        }
        assert_eq!(source.probe.retirements(), 0);
        source.handle.allocation_lease().validate_live().unwrap();
        if mode != 2 {
            assert_eq!(fixture.produced.lock().unwrap()[0].retirements(), 1);
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn alias_requires_exact_route_and_source_and_owned_cancellation_keeps_borrowed_live() {
    use std::{
        future::Future,
        task::{Context, Poll},
    };
    for mode in [11, 12] {
        let fixture = new_fixture();
        fixture.mode.store(mode, Ordering::Relaxed);
        assert!(ProviderRealElementwiseRecipe::plan(
            fixture,
            request(cast(ProviderPrecision::F32))
        )
        .unwrap()
        .is_none());
    }
    let fixture = new_fixture();
    let recipe =
        ProviderRealElementwiseRecipe::plan(fixture, request(cast(ProviderPrecision::F64)))
            .unwrap()
            .unwrap();
    let foreign = new_fixture();
    let replacement = input(foreign);
    assert!(ready(recipe.execute(&replacement.handle)).is_err());
    assert_eq!(replacement.probe.retirements(), 0);
    fixture.mode.store(4, Ordering::Relaxed);
    let source = input(fixture);
    drop(Box::pin(recipe.execute(&source.handle)));
    assert!(fixture.produced.lock().unwrap().is_empty());
    let protected = [&replacement.handle];
    let mut future = Box::pin(recipe.execute_protected(&source.handle, &protected));
    assert!(matches!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(std::task::Waker::noop())),
        Poll::Pending
    ));
    drop(future);
    assert_eq!(fixture.produced.lock().unwrap()[0].retirements(), 1);
    assert_eq!(source.probe.retirements(), 0);
    assert_eq!(replacement.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn arbitrary_protected_alias_is_not_source_alias_permission() {
    let fixture = new_fixture();
    let source = input(fixture);
    let protected = input(fixture);
    *fixture.protected_alias.lock().unwrap() = Some(protected.handle.clone());
    for target in [ProviderPrecision::F32, ProviderPrecision::F64] {
        let recipe = ProviderRealElementwiseRecipe::plan(fixture, request(cast(target)))
            .unwrap()
            .unwrap();
        assert!(
            ready(recipe.execute_protected(&source.handle, &[&protected.handle]))
                .err()
                .unwrap()
                .is_after_effect()
        );
        assert_eq!(source.probe.retirements(), 0);
        assert_eq!(protected.probe.retirements(), 0);
        protected.handle.allocation_lease().validate_live().unwrap();
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn live_extent_and_rederived_alias_recipe_are_checked_before_hook() {
    let fixture = new_fixture();
    let source = input(fixture);
    let mut description = request(cast(ProviderPrecision::F32));
    description.source_buffer_bytes += 4;
    let wrong_extent = ProviderRealElementwiseRecipe::plan(fixture, description)
        .unwrap()
        .unwrap();
    assert!(ready(wrong_extent.execute(&source.handle)).is_err());
    let alias = ProviderRealElementwiseRecipe::plan(fixture, request(cast(ProviderPrecision::F32)))
        .unwrap()
        .unwrap();
    fixture.mode.store(12, Ordering::Relaxed);
    assert!(ready(alias.execute(&source.handle)).is_err());
    assert!(fixture.produced.lock().unwrap().is_empty());
    assert_eq!(source.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn downcast_uses_target_extent_and_dead_or_logical_sources_never_reach_hook() {
    let fixture = new_fixture();
    let mut description = request(cast(ProviderPrecision::F32));
    description.source.element_type = ProviderElementType::F64;
    description.source_buffer_bytes = 16;
    let down = ProviderRealElementwiseRecipe::plan(fixture, description)
        .unwrap()
        .unwrap();
    assert_eq!(down.physical().output_buffer_bytes, 8);
    let source = fixture
        .owner
        .allocate_zeroed(
            vec![2, 1, 1],
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let output = ready(down.execute(&source.handle)).unwrap();
    assert_eq!(
        output.output.physical().element_type(),
        NumericElementType::F32
    );
    drop(output);
    source.probe.mark_unavailable();
    assert!(ready(down.execute(&source.handle)).is_err());
    assert_eq!(fixture.produced.lock().unwrap().len(), 1);
    let alias = ProviderRealElementwiseRecipe::plan(fixture, request(cast(ProviderPrecision::F32)))
        .unwrap()
        .unwrap();
    let logical = fixture
        .owner
        .allocate_logical_zeroed(vec![2, 1, 1], ProviderPrecision::F32)
        .unwrap();
    assert!(ready(alias.execute(&logical.handle)).is_err());
    assert_eq!(fixture.produced.lock().unwrap().len(), 1);
    assert_eq!(logical.probe.retirements(), 0);
}
