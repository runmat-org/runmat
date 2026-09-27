use super::*;

fn provider() -> WgpuProvider {
    WgpuProvider::new(WgpuProviderOptions::default()).expect("required WGPU adapter")
}

fn upload(provider: &WgpuProvider, data: &[f64]) -> GpuTensorHandle {
    provider
        .upload_exec(&HostTensorView {
            data,
            shape: &[2, 2],
        })
        .unwrap()
}

fn download(provider: &WgpuProvider, handle: &GpuTensorHandle) -> Vec<f64> {
    futures::executor::block_on(provider.download_exec(handle))
        .unwrap()
        .data
}

#[test]
#[ignore = "requires provisioned WGPU adapter; run with --ignored"]
fn published_qr_and_cholesky_outputs_survive_next_factorization() {
    let provider = provider();
    let first = upload(&provider, &[2.0, 0.0, 0.0, 3.0]);
    let second = upload(&provider, &[4.0, 0.0, 0.0, 5.0]);
    let (q1, r1, inverse1) = provider
        .qr_factor_device(&first, 2, 2, "first", true)
        .unwrap();
    let inverse1 = inverse1.unwrap();
    let before_r = download(&provider, &r1);
    let before_inverse = download(&provider, &inverse1);
    assert_eq!(before_r, vec![2.0, 0.0, 0.0, 3.0]);
    let (_, r2, inverse2) = provider
        .qr_factor_device(&second, 2, 2, "second", true)
        .unwrap();
    assert!(!r1.shares_allocation_with(&r2));
    assert!(!inverse1.shares_allocation_with(inverse2.as_ref().unwrap()));
    assert_eq!(download(&provider, &r1), before_r);
    assert_eq!(download(&provider, &inverse1), before_inverse);
    assert_eq!(download(&provider, &q1), vec![1.0, 0.0, 0.0, 1.0]);
    assert_eq!(download(&provider, &first), vec![2.0, 0.0, 0.0, 3.0]);

    let spd1 = upload(&provider, &[4.0, 0.0, 0.0, 9.0]);
    let spd2 = upload(&provider, &[16.0, 0.0, 0.0, 25.0]);
    let (factor1, inv1) = provider
        .chol_factor_spd_device(&spd1, 2, "first-chol")
        .unwrap();
    let before_factor = download(&provider, &factor1);
    let before_inverse = download(&provider, &inv1);
    assert_eq!(before_factor, vec![2.0, 0.0, 0.0, 3.0]);
    let (factor2, inv2) = provider
        .chol_factor_spd_device(&spd2, 2, "second-chol")
        .unwrap();
    assert!(!factor1.shares_allocation_with(&factor2));
    assert!(!inv1.shares_allocation_with(&inv2));
    assert_eq!(download(&provider, &factor1), before_factor);
    assert_eq!(download(&provider, &inv1), before_inverse);
}

#[test]
#[ignore = "requires provisioned WGPU adapter; run with --ignored"]
fn power_step_does_not_mutate_borrowed_rhs_or_its_alias() {
    let provider = provider();
    let lhs = upload(&provider, &[2.0, 0.0, 0.0, 3.0]);
    let values = [1.0, 1.0, 0.0, 1.0];
    let rhs = upload(&provider, &values);
    let alias = provider.reshape_exec(&rhs, &[4, 1]).unwrap();
    let result = futures::executor::block_on(provider.matmul_power_step_exec(
        &lhs,
        &rhs,
        &runmat_accelerate_api::PowerStepEpilogue { epsilon: 0.0 },
    ))
    .unwrap();
    assert!(!result.shares_allocation_with(&rhs));
    assert_eq!(download(&provider, &rhs), values);
    assert_eq!(download(&provider, &alias), values);
    let expected = [2.0 / 13.0_f64.sqrt(), 3.0 / 13.0_f64.sqrt(), 0.0, 1.0];
    for (actual, expected) in download(&provider, &result).into_iter().zip(expected) {
        assert!((actual - expected).abs() < 2.0e-6);
    }
}
