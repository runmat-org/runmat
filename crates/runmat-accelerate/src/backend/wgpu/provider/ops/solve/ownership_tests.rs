use super::*;
use runmat_accelerate_api::{HostNumericDataOwned, HostNumericDataView, HostNumericTensorView};

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
fn real_linalg_fallback_preserves_physical_source_class() {
    let provider = provider();
    for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
        let single = [3.0_f32, 0.0, 1.0, 2.0];
        let double = [3.0_f64, 0.0, 1.0, 2.0];
        let data = match precision {
            ProviderPrecision::F32 => HostNumericDataView::F32(&single),
            ProviderPrecision::F64 => HostNumericDataView::F64(&double),
        };
        let source = provider
            .upload_numeric_exec(&HostNumericTensorView {
                data,
                shape: &[2, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        if provider.precision() == ProviderPrecision::F64 {
            let before = provider.telemetry_snapshot();
            let allocations = provider.allocations.record_count().unwrap();
            let error = provider
                .qr_factor_device(&source, 2, 2, "unsupported-double-owner", true)
                .unwrap_err();
            assert!(runmat_accelerate_api::is_unsupported_provider_operation(
                &error
            ));
            assert_eq!(provider.telemetry_snapshot(), before);
            assert_eq!(provider.allocations.record_count().unwrap(), allocations);
        }
        let inverse = futures::executor::block_on(provider.inv_exec(&source)).unwrap();
        let norm = futures::executor::block_on(provider.norm_exec(&source, ProviderNormOrder::Fro))
            .unwrap();
        let pinv = futures::executor::block_on(
            provider.pinv_exec(&source, ProviderPinvOptions { tolerance: None }),
        )
        .unwrap();
        let lu = futures::executor::block_on(provider.lu_exec(&source)).unwrap();
        let solved = futures::executor::block_on(provider.mldivide_exec(&source, &source)).unwrap();
        for output in [&inverse, &norm, &pinv, &lu.lower, &lu.upper, &solved] {
            assert_eq!(
                runmat_accelerate_api::handle_precision(output),
                Some(precision)
            );
            assert!(!source.shares_allocation_with(output));
            let readback =
                futures::executor::block_on(provider.download_numeric_exec(output)).unwrap();
            assert!(matches!(
                (precision, readback.data),
                (ProviderPrecision::F32, HostNumericDataOwned::F32(_))
                    | (ProviderPrecision::F64, HostNumericDataOwned::F64(_))
            ));
        }
        let inv = futures::executor::block_on(provider.download_numeric_exec(&inverse)).unwrap();
        let inv = match inv.data {
            HostNumericDataOwned::F32(values) => {
                values.into_iter().map(f64::from).collect::<Vec<_>>()
            }
            HostNumericDataOwned::F64(values) => values,
            other => panic!("unexpected inverse storage: {other:?}"),
        };
        assert!((3.0 * inv[0] + inv[1] - 1.0).abs() < 1e-5);
        assert!((2.0 * inv[1]).abs() < 1e-5);
        assert!((3.0 * inv[2] + inv[3]).abs() < 1e-5);
        assert!((2.0 * inv[3] - 1.0).abs() < 1e-5);
        let solved = futures::executor::block_on(provider.download_numeric_exec(&solved)).unwrap();
        let solved = match solved.data {
            HostNumericDataOwned::F32(values) => {
                values.into_iter().map(f64::from).collect::<Vec<_>>()
            }
            HostNumericDataOwned::F64(values) => values,
            other => panic!("unexpected solved storage: {other:?}"),
        };
        for (actual, expected) in solved.into_iter().zip([1.0, 0.0, 0.0, 1.0]) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }
    let complex = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[2.0, 1.0]),
            shape: &[1, 1],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    let inv_error = futures::executor::block_on(provider.inv_exec(&complex)).unwrap_err();
    let pinv_error = futures::executor::block_on(
        provider.pinv_exec(&complex, ProviderPinvOptions { tolerance: None }),
    )
    .unwrap_err();
    assert!(inv_error.is_unsupported());
    assert!(pinv_error.is_unsupported());
    assert_eq!(
        runmat_accelerate_api::handle_storage(&complex),
        GpuTensorStorage::ComplexInterleaved
    );
}

#[test]
#[ignore = "requires provisioned WGPU adapter; run with --ignored"]
fn host_factor_hooks_preserve_native_class_and_complex_eigen_spectrum() {
    // Register exactly this physical owner for the public Runtime fallback.
    // This fixture-only static lifetime is the existing registry contract.
    let provider: &'static WgpuProvider = Box::leak(Box::new(provider()));
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
        let single = [4.0_f32, 0.0, 0.0, 9.0];
        let double = [4.0_f64, 0.0, 0.0, 9.0];
        let data = match precision {
            ProviderPrecision::F32 => HostNumericDataView::F32(&single),
            ProviderPrecision::F64 => HostNumericDataView::F64(&double),
        };
        let source = provider
            .upload_numeric_exec(&HostNumericTensorView {
                data,
                shape: &[2, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let chol = futures::executor::block_on(provider.chol_exec(&source, false)).unwrap();
        assert_eq!(chol.info, 0);
        assert_eq!(
            runmat_accelerate_api::handle_precision(&chol.factor),
            Some(precision)
        );

        let qr = futures::executor::block_on(provider.qr_exec(
            &source,
            ProviderQrOptions {
                economy: false,
                column_order: ProviderQrColumnOrder::Unpivoted,
                outputs: ProviderQrOutputs::Qr,
            },
        ))
        .unwrap();
        let ProviderQrResult::Qr { q, r } = qr else {
            panic!("unpivoted QR returned a different result variant");
        };
        assert_eq!(runmat_accelerate_api::handle_precision(&q), Some(precision));
        assert_eq!(runmat_accelerate_api::handle_precision(&r), Some(precision));
        let pivoted = futures::executor::block_on(provider.qr_exec(
            &source,
            ProviderQrOptions {
                economy: false,
                column_order: ProviderQrColumnOrder::Pivoted,
                outputs: ProviderQrOutputs::Qrp(runmat_accelerate_api::ProviderQrPivot::Matrix),
            },
        ))
        .unwrap();
        let ProviderQrResult::Qrp { q, r, permutation } = pivoted else {
            panic!("pivoted QR returned a different result variant");
        };
        assert_eq!(runmat_accelerate_api::handle_precision(&q), Some(precision));
        assert_eq!(runmat_accelerate_api::handle_precision(&r), Some(precision));
        assert_eq!(
            runmat_accelerate_api::handle_precision(&permutation),
            Some(ProviderPrecision::F64)
        );

        let single_rotation = [0.0_f32, 1.0, -1.0, 0.0];
        let double_rotation = [0.0_f64, 1.0, -1.0, 0.0];
        let rotation_data = match precision {
            ProviderPrecision::F32 => HostNumericDataView::F32(&single_rotation),
            ProviderPrecision::F64 => HostNumericDataView::F64(&double_rotation),
        };
        let rotation = provider
            .upload_numeric_exec(&HostNumericTensorView {
                data: rotation_data,
                shape: &[2, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let hook = futures::executor::block_on(runmat_accelerate_api::AccelProvider::eig(
            provider, &rotation, false,
        ))
        .unwrap_err();
        assert!(hook.is_unsupported());
        let eig = futures::executor::block_on(
            runmat_runtime::builtins::math::linalg::factor::eig::evaluate_in_context(
                Some(&runtime),
                Value::GpuTensor(rotation.clone()),
                runmat_runtime::call::FunctionArgumentView::from_slice(&[]),
                false,
            ),
        )
        .unwrap();
        for value in [eig.eigenvalues(), eig.diagonal_matrix(), eig.right()] {
            let Value::GpuTensor(output) = value else {
                panic!("public eig lost resident factor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(precision)
            );
            assert_eq!(
                runmat_accelerate_api::handle_storage(&output),
                GpuTensorStorage::ComplexInterleaved
            );
            assert!(!output.shares_allocation_with(&rotation));
        }
        let Value::GpuTensor(values) = eig.eigenvalues() else {
            unreachable!()
        };
        let spectrum =
            futures::executor::block_on(provider.download_numeric_exec(&values)).unwrap();
        match spectrum.data {
            HostNumericDataOwned::F32(values) => {
                assert_eq!(values.len(), 4);
                assert!(values
                    .chunks_exact(2)
                    .all(|pair| pair[0].abs() < 1e-5 && (pair[1].abs() - 1.0).abs() < 1e-5));
            }
            HostNumericDataOwned::F64(values) => {
                assert_eq!(values.len(), 4);
                assert!(values
                    .chunks_exact(2)
                    .all(|pair| pair[0].abs() < 1e-10 && (pair[1].abs() - 1.0).abs() < 1e-10));
            }
            _ => panic!("eig changed the native floating class"),
        }
    }
}

#[test]
#[ignore = "requires provisioned F32 WGPU adapter; RUNMAT_WGPU_FORCE_PRECISION=f32"]
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
