use crate::builtins::common::{gpu_helpers, test_support};
use crate::dispatcher::call_builtin;
use runmat_accelerate_api::{
    same_provider_instance, AccelProvider, GpuHandleProvenance, GpuTensorStorage,
    HostNumericDataOwned, NumericElementType, ProviderPrecision,
};
use runmat_value::{NumericDType, Tensor, Value};

#[derive(Clone, Copy)]
enum ScalarOracle {
    Add,
    Sub,
    ReverseSub,
    Mul,
    Div,
    ReverseDiv,
}

#[derive(Clone, Copy)]
enum ResultCarrier {
    Host,
    ExplicitResident,
}

impl ScalarOracle {
    fn host(self, resident: f64, scalar: f64) -> f32 {
        (match self {
            Self::Add => resident + scalar,
            Self::Sub => resident - scalar,
            Self::ReverseSub => scalar - resident,
            Self::Mul => resident * scalar,
            Self::Div => resident / scalar,
            Self::ReverseDiv => scalar / resident,
        }) as f32
    }
}

#[test]
#[ignore = "requires an actual F32 WGPU adapter"]
fn mixed_single_resident_double_scalar_uses_host_oracle_in_both_orders() {
    let _guard = test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("required actual WGPU adapter");
    assert_eq!(provider.precision(), ProviderPrecision::F32);

    let below_one = f32::from_bits(0x3f7f_ffff);
    let source = Tensor::from_f32(vec![below_one; 2], vec![1, 2]).unwrap();
    let resident = gpu_helpers::upload_tensor(provider, &source)
        .unwrap()
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    assert!(runmat_accelerate_api::handle_is_explicit(&resident));

    let near_one_above = 1.0 + 2.0f64.powi(-24);
    let near_one_below = 1.0 - 2.0f64.powi(-25);
    let half_ulp_below = f64::from(f32::EPSILON) / 4.0 - f64::EPSILON;
    let cases = [
        (
            "plus",
            ResultCarrier::Host,
            (half_ulp_below, ScalarOracle::Add),
            (half_ulp_below, ScalarOracle::Add),
        ),
        (
            "minus",
            ResultCarrier::Host,
            (near_one_above, ScalarOracle::Sub),
            (near_one_above, ScalarOracle::ReverseSub),
        ),
        (
            "times",
            ResultCarrier::Host,
            (near_one_above, ScalarOracle::Mul),
            (near_one_above, ScalarOracle::Mul),
        ),
        (
            "rdivide",
            ResultCarrier::ExplicitResident,
            (near_one_above, ScalarOracle::Div),
            (near_one_below, ScalarOracle::ReverseDiv),
        ),
        (
            "ldivide",
            ResultCarrier::ExplicitResident,
            (near_one_below, ScalarOracle::ReverseDiv),
            (near_one_above, ScalarOracle::Div),
        ),
    ];
    for (name, carrier, right, left) in cases {
        for reverse in [false, true] {
            let (scalar, operation) = if reverse { left } else { right };
            let operands = if reverse {
                [Value::Num(scalar), Value::GpuTensor(resident.clone())]
            } else {
                [Value::GpuTensor(resident.clone()), Value::Num(scalar)]
            };
            let expected = operation.host(f64::from(below_one), scalar);
            let result = call_builtin(name, &operands).unwrap();
            let output_weak = match (carrier, result.value()) {
                (ResultCarrier::Host, Value::Tensor(output)) => {
                    assert_eq!(output.shape(), &[1, 2]);
                    assert_eq!(output.numeric_dtype(), NumericDType::F32);
                    assert_eq!(output.materialize_f64(), vec![f64::from(expected); 2]);
                    None
                }
                (ResultCarrier::ExplicitResident, Value::GpuTensor(output)) => {
                    assert!(runmat_accelerate_api::handle_is_explicit(output));
                    assert_eq!(output.device_id(), provider.device_id());
                    assert_eq!(output.shape(), &[1, 2]);
                    assert_eq!(output.physical().element_type(), NumericElementType::F32);
                    assert_eq!(output.physical().storage(), GpuTensorStorage::Real);
                    assert!(!runmat_accelerate_api::handle_is_logical(output));
                    assert!(!runmat_accelerate_api::handle_is_transposed(output));
                    assert!(!output.shares_allocation_with(&resident));
                    let owner = gpu_helpers::exact_provider_for_handle(output)
                        .expect("restored division output has an exact owner");
                    assert!(same_provider_instance(owner, provider));
                    output.allocation_lease().validate_live().unwrap();
                    let downloaded =
                        futures::executor::block_on(provider.download_numeric(output)).unwrap();
                    let HostNumericDataOwned::F32(values) = downloaded.data else {
                        panic!("restored division changed native Single class")
                    };
                    assert_eq!(values, vec![expected; 2]);
                    Some(output.downgrade())
                }
                (ResultCarrier::Host, _) => panic!("host arithmetic changed its output carrier"),
                (ResultCarrier::ExplicitResident, _) => {
                    panic!("explicit division failed to restore its output carrier")
                }
            };
            drop(result);
            if let Some(weak) = output_weak {
                assert!(weak.acquire().is_err(), "restored output remains live");
            }
            resident.allocation_lease().validate_live().unwrap();
        }
    }
}
