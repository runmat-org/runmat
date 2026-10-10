use super::*;

fn provider() -> WgpuProvider {
    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("ownership qualification requires an actual WGPU adapter");
    if let Ok(requested) = std::env::var("RUNMAT_WGPU_FORCE_PRECISION") {
        let actual = match provider.precision {
            NumericPrecision::F32 => "f32",
            NumericPrecision::F64 => "f64",
        };
        assert_eq!(requested.to_ascii_lowercase(), actual);
    }
    provider
}

fn upload(provider: &WgpuProvider, data: &[f64], shape: &[usize]) -> GpuTensorHandle {
    provider
        .upload_exec(&HostTensorView { data, shape })
        .unwrap()
}

fn download(provider: &WgpuProvider, handle: &GpuTensorHandle) -> Vec<f64> {
    futures::executor::block_on(provider.download_exec(handle))
        .unwrap()
        .data
}

#[test]
#[ignore = "requires an actual WGPU adapter; run explicitly with --ignored"]
fn published_syrk_result_survives_repeated_calls_and_retires_with_last_alias() {
    let provider = provider();
    let a = upload(&provider, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = upload(&provider, &[2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let first = provider.syrk_exec(&a).unwrap();
    let alias = first.clone();
    let weak = first.allocation_lease().downgrade();
    assert_eq!(download(&provider, &first), [5.0, 11.0, 11.0, 25.0]);
    let second = provider.syrk_exec(&b).unwrap();
    assert!(!first.shares_allocation_with(&second));
    assert!(!runmat_accelerate_api::WgpuBufferOwner::ptr_eq(
        &provider.get_entry_raw(&first).unwrap().buffer,
        &provider.get_entry_raw(&second).unwrap().buffer,
    ));
    assert_eq!(download(&provider, &second), [4.0, 0.0, 0.0, 9.0]);
    assert_eq!(download(&provider, &first), [5.0, 11.0, 11.0, 25.0]);
    drop(first);
    assert!(weak.acquire().is_some());
    assert_eq!(download(&provider, &alias), [5.0, 11.0, 11.0, 25.0]);
    drop(alias);
    assert!(weak.acquire().is_none());
    assert_eq!(download(&provider, &a), [1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[ignore = "requires an actual WGPU adapter; run explicitly with --ignored"]
fn chunked_matmul_preserves_first_partial_and_previous_result() {
    let provider = provider();
    // Eight unequal partials cross the actual production chunking threshold.
    // Reusing the first accumulator as scratch incorrectly replaces its first
    // contribution with the second; constant inputs would conceal that bug.
    let k = 65_536usize;
    let mut lhs = Vec::with_capacity(2 * k);
    for column in 0..k {
        let value = (column / 8192 + 1) as f64;
        lhs.extend([value, 2.0 * value]);
    }
    let mut rhs = vec![1.0; k];
    rhs.resize(2 * k, 2.0);
    let a = upload(&provider, &lhs, &[2, k]);
    let b = upload(&provider, &rhs, &[k, 2]);
    let first = provider.matmul_exec(&a, &b).unwrap();
    let sum = (1..=8).sum::<usize>() as f64 * 8192.0;
    let expected = [sum, 2.0 * sum, 2.0 * sum, 4.0 * sum];
    assert_eq!(download(&provider, &first), expected);
    let alias = first.clone();
    let weak = first.allocation_lease().downgrade();
    let negative = rhs.iter().map(|value| -value).collect::<Vec<_>>();
    let negative_b = upload(&provider, &negative, &[k, 2]);
    let second = provider.matmul_exec(&a, &negative_b).unwrap();
    assert!(!first.shares_allocation_with(&second));
    assert_eq!(download(&provider, &second), expected.map(|value| -value));
    assert_eq!(download(&provider, &first), expected);
    assert_eq!(download(&provider, &a), lhs);
    assert_eq!(download(&provider, &b), rhs);
    drop(first);
    assert!(weak.acquire().is_some());
    assert_eq!(download(&provider, &alias), expected);
    drop(alias);
    assert!(weak.acquire().is_none());
}
