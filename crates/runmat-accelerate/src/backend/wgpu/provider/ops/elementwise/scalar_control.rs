use anyhow::Result;
use runmat_accelerate_api::{unsupported_provider_operation, ProviderPrecision};

use crate::backend::wgpu::types::ScalarOpCode;

/// An F32 uniform cannot carry an arbitrary Double control. Mixed Single and
/// Double host arithmetic evaluates in Double and converts once to Single;
/// rounding the control first can change the result. NaN payloads are not
/// part of the scalar-binary contract, but signed zero and infinities remain
/// exact. This excludes only the proven lossy-control path; it does not claim
/// that every F32 shader operation matches Double-intermediate host rounding.
/// Max also declines a lossy ABI control: this owner does not erase the
/// original scalar class to invent a mixed-class numerical certificate.
pub(super) fn admit(
    operation: ScalarOpCode,
    precision: ProviderPrecision,
    scalar: f64,
) -> Result<()> {
    if precision == ProviderPrecision::F32
        && matches!(
            operation,
            ScalarOpCode::Add
                | ScalarOpCode::Sub
                | ScalarOpCode::RSub
                | ScalarOpCode::Mul
                | ScalarOpCode::Div
                | ScalarOpCode::RDiv
                | ScalarOpCode::Max
        )
        && !scalar.is_nan()
        && f64::from(scalar as f32).to_bits() != scalar.to_bits()
    {
        return Err(unsupported_provider_operation(
            "scalar binary operation requires a losslessly representable F32 control",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        is_unsupported_provider_operation, same_provider_instance, AccelProvider, GpuTensorHandle,
        GpuTensorStorage, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
        NumericElementType,
    };

    use crate::backend::wgpu::provider::{
        register_wgpu_provider, WgpuProvider, WgpuProviderOptions,
    };

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn max_f32_control_rejects_class_erasure_without_claiming_mixed_support() {
        let lossy = 1.0 + f64::EPSILON;
        let error = admit(ScalarOpCode::Max, ProviderPrecision::F32, lossy).unwrap_err();
        assert!(is_unsupported_provider_operation(&error));
        assert!(admit(ScalarOpCode::Max, ProviderPrecision::F64, lossy).is_ok());
        for scalar in [0.0, -0.0, 3.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert!(admit(ScalarOpCode::Max, ProviderPrecision::F32, scalar).is_ok());
        }
    }

    struct ScalarBoundaryCase {
        operation: ScalarOpCode,
        scalar: f64,
        host_operation: fn(f64, f64) -> f64,
    }

    #[test]
    fn f32_binary_controls_reject_known_lossy_double_before_provider_work() {
        let below_one = f64::from(f32::from_bits(0x3f7f_ffff));
        let half_ulp = f64::from(f32::EPSILON) / 4.0;
        let cases = [
            ScalarBoundaryCase {
                operation: ScalarOpCode::Add,
                scalar: half_ulp - f64::EPSILON,
                host_operation: |a, s| a + s,
            },
            ScalarBoundaryCase {
                operation: ScalarOpCode::Sub,
                scalar: 1.0 + 2.0f64.powi(-24),
                host_operation: |a, s| a - s,
            },
            ScalarBoundaryCase {
                operation: ScalarOpCode::RSub,
                scalar: 1.0 + 2.0f64.powi(-24),
                host_operation: |a, s| s - a,
            },
            ScalarBoundaryCase {
                operation: ScalarOpCode::Mul,
                scalar: 1.0 + 2.0f64.powi(-24),
                host_operation: |a, s| a * s,
            },
            ScalarBoundaryCase {
                operation: ScalarOpCode::Div,
                scalar: 1.0 + 2.0f64.powi(-24),
                host_operation: |a, s| a / s,
            },
            ScalarBoundaryCase {
                operation: ScalarOpCode::RDiv,
                scalar: 1.0 - 2.0f64.powi(-25),
                host_operation: |a, s| s / a,
            },
        ];
        for ScalarBoundaryCase {
            operation,
            scalar,
            host_operation,
        } in cases
        {
            let host = host_operation(below_one, scalar) as f32;
            let prematurely_rounded = host_operation(below_one, f64::from(scalar as f32)) as f32;
            assert_ne!(host.to_bits(), prematurely_rounded.to_bits());
            let error = admit(operation, ProviderPrecision::F32, scalar)
                .expect_err("F32 uniform would erase the Double control");
            assert!(is_unsupported_provider_operation(&error));
            assert!(admit(operation, ProviderPrecision::F64, scalar).is_ok());
        }
        for operation in [
            ScalarOpCode::Add,
            ScalarOpCode::Sub,
            ScalarOpCode::RSub,
            ScalarOpCode::Mul,
            ScalarOpCode::Div,
            ScalarOpCode::RDiv,
        ] {
            for scalar in [half_ulp, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
                assert!(admit(operation, ProviderPrecision::F32, scalar).is_ok());
            }
        }
    }

    async fn read_and_retire(
        provider: &'static WgpuProvider,
        input: &GpuTensorHandle,
        output: GpuTensorHandle,
    ) -> f32 {
        assert_eq!(output.shape(), input.shape());
        assert_eq!(output.physical().element_type(), NumericElementType::F32);
        assert_eq!(output.physical().storage(), GpuTensorStorage::Real);
        assert!(!output.shares_allocation_with(input));
        let owner =
            runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle(&output)
                .expect("output has a registered exact provider");
        assert!(same_provider_instance(owner, provider));
        output.allocation_lease().validate_live().unwrap();
        let HostNumericDataOwned::F32(values) =
            provider.download_numeric(&output).await.unwrap().data
        else {
            panic!("scalar Add changed the native Single class")
        };
        assert_eq!(values.len(), 1);
        let retired = output.downgrade();
        drop(output);
        assert!(retired.acquire().is_err(), "owned output view remains live");
        input.allocation_lease().validate_live().unwrap();
        values[0]
    }

    #[tokio::test]
    #[ignore = "requires an actual F32 WGPU adapter"]
    async fn f32_binary_controls_reject_before_queue_or_allocation() {
        let provider = register_wgpu_provider(WgpuProviderOptions::default())
            .expect("required actual WGPU adapter");
        assert_eq!(provider.precision(), ProviderPrecision::F32);
        let below_one = f32::from_bits(0x3f7f_ffff);
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[below_one]),
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            })
            .expect("native Single input");
        let half_ulp = f64::from(f32::EPSILON) / 4.0;
        let before_telemetry = provider.telemetry_snapshot();
        let before_allocations = provider.allocations.record_count().unwrap();
        for (operation, scalar) in [
            (ScalarOpCode::Add, half_ulp - f64::EPSILON),
            (ScalarOpCode::Add, half_ulp + f64::EPSILON),
            (ScalarOpCode::Sub, 1.0 + 2.0f64.powi(-24)),
            (ScalarOpCode::RSub, 1.0 + 2.0f64.powi(-24)),
            (ScalarOpCode::Mul, 1.0 + 2.0f64.powi(-24)),
            (ScalarOpCode::Div, 1.0 + 2.0f64.powi(-24)),
            (ScalarOpCode::RDiv, 1.0 - 2.0f64.powi(-25)),
        ] {
            let error = provider
                .scalar_op_exec(operation, &input, scalar)
                .expect_err("lossy control must not enter an F32 kernel");
            assert!(is_unsupported_provider_operation(&error));
            assert_eq!(provider.telemetry_snapshot(), before_telemetry);
            assert_eq!(
                provider.allocations.record_count().unwrap(),
                before_allocations
            );
            input.allocation_lease().validate_live().unwrap();
        }
        let exact = provider
            .scalar_add(&input, half_ulp)
            .expect("exactly representable control remains resident");
        assert_eq!(read_and_retire(provider, &input, exact).await, 1.0);

        let negative_zero = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[-0.0]),
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let signed = provider.scalar_add(&negative_zero, -0.0).unwrap();
        assert_eq!(
            read_and_retire(provider, &negative_zero, signed)
                .await
                .to_bits(),
            (-0.0f32).to_bits()
        );
        for (scalar, expected_nan) in [(f64::INFINITY, false), (f64::NAN, true)] {
            let output = provider.scalar_add(&input, scalar).unwrap();
            let value = read_and_retire(provider, &input, output).await;
            assert_eq!(value.is_nan(), expected_nan);
            if !expected_nan {
                assert_eq!(value, f32::INFINITY);
            }
        }
    }
}
