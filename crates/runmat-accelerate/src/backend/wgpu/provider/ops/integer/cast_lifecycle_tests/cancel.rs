use super::*;
#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires an actual native WGPU adapter; run explicitly in cast qualification"]
async fn dropping_unpolled_integer_cast_does_not_prepare_or_allocate() {
    let provider = owner().await;
    let input = provider
        .upload_integer_exec(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[i64::MIN, 9_007_199_254_740_993]),
            shape: &[1, 2],
        })
        .unwrap();
    let count = provider.allocations.record_count().unwrap();
    let cache = provider.validated_fusion_cache.lock().unwrap().len();
    drop(provider.cast_to_integer_exec(&input, IntegerElementType::U64));
    assert_eq!(provider.allocations.record_count().unwrap(), count);
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    input.allocation_lease().validate_live().unwrap();
    let output = provider
        .cast_to_integer_exec(&input, IntegerElementType::U64)
        .await
        .unwrap();
    assert_eq!(
        provider.download_integer_exec(&output).await.unwrap().data,
        HostIntegerDataOwned::U64(vec![0, 9_007_199_254_740_993])
    );
    output.release().unwrap();
    input.release().unwrap();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn dropping_pending_browser_integer_cast_retains_borrowed_source() {
    use std::future::Future;
    let provider = owner().await;
    let input = provider
        .upload_integer_exec(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[i64::MIN, 9_007_199_254_740_993]),
            shape: &[1, 2],
        })
        .unwrap();
    let count = provider.allocations.record_count().unwrap();
    let mut future = Box::pin(provider.cast_to_integer_exec(&input, IntegerElementType::U64));
    let pending = std::future::poll_fn(|context| {
        std::task::Poll::Ready(future.as_mut().poll(context).is_pending())
    })
    .await;
    assert!(pending, "cold browser scope preparation must suspend");
    drop(future);
    assert_eq!(provider.allocations.record_count().unwrap(), count);
    assert!(provider.validated_fusion_cache.lock().unwrap().is_empty());
    input.allocation_lease().validate_live().unwrap();
    let output = provider
        .cast_to_integer_exec(&input, IntegerElementType::U64)
        .await
        .unwrap();
    assert_eq!(
        provider.download_integer_exec(&output).await.unwrap().data,
        HostIntegerDataOwned::U64(vec![0, 9_007_199_254_740_993])
    );
    output.release().unwrap();
    input.release().unwrap();
}
