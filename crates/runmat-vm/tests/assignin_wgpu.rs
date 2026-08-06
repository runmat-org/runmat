#[path = "support/mod.rs"]
mod test_helpers;

#[test]
#[cfg(feature = "wgpu")]
fn assignin_wgpu_preserves_exact_resident_integer_value() {
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};

    if register_wgpu_provider(WgpuProviderOptions::default()).is_err() {
        return;
    }
    let vars = test_helpers::execute_source(
        "base = uint64(9007199254740992); source_gpu = gpuArray([base + uint64(1) intmax('uint64')]); assignin('base','assigned_gpu',source_gpu); copied = evalin('base','gather(assigned_gpu)');",
    )
    .expect("compiled assignin WGPU transfer");
    let resident_ids = vars
        .iter()
        .filter_map(|value| match value {
            runmat_builtins::Value::GpuTensor(handle) => Some(handle.buffer_id),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        resident_ids.iter().any(|id| resident_ids
            .iter()
            .filter(|candidate| *candidate == id)
            .count()
            >= 2),
        "assignin should retain the same WGPU buffer, got {resident_ids:?}"
    );
    assert!(vars.iter().any(|value| {
        matches!(
            value,
            runmat_builtins::Value::Tensor(tensor)
                if tensor.integer_storage()
                    == Some(&runmat_builtins::IntegerStorage::U64(vec![
                        9_007_199_254_740_993,
                        u64::MAX,
                    ]))
        )
    }));
}
