#[path = "support/mod.rs"]
mod test_helpers;

#[test]
#[cfg(feature = "wgpu")]
#[ignore = "requires an actual WGPU adapter; unavailable hardware fails when explicitly run"]
fn assignin_wgpu_preserves_exact_resident_integer_value() {
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};

    register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual WGPU adapter required for assignin qualification");
    let vars = test_helpers::execute_source(
        "base = uint64(9007199254740992); source_gpu = gpuArray([base + uint64(1) intmax('uint64')]); assignin('base','assigned_gpu',source_gpu); copied = evalin('base','gather(assigned_gpu)');",
    )
    .expect("compiled assignin WGPU transfer");
    let resident_handles = vars
        .iter()
        .filter_map(|value| match value {
            runmat_value::Value::GpuTensor(handle) => Some(handle),
            _ => None,
        })
        .collect::<Vec<_>>();
    let resident_ids = resident_handles
        .iter()
        .map(|handle| handle.allocation_identity())
        .collect::<Vec<_>>();
    assert!(
        resident_handles
            .iter()
            .enumerate()
            .any(|(index, handle)| resident_handles[index + 1..]
                .iter()
                .any(|candidate| handle.shares_allocation_with(candidate))),
        "assignin should retain the same WGPU buffer, got {resident_ids:?}"
    );
    assert!(vars.iter().any(|value| {
        matches!(
            value,
            runmat_value::Value::Tensor(tensor)
                if tensor.integer_storage()
                    == Some(&runmat_value::IntegerStorage::U64(vec![
                        9_007_199_254_740_993,
                        u64::MAX,
                    ]))
        )
    }));
}
