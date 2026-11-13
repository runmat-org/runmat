//! Legacy dispatch retains exact constants through the shared typed owner.
use super::*;

fn divide_plan(control: f64) -> FusionGroupPlan {
    let mut source = plan();
    source.operations[0] = FusionOp::Primitive {
        op: PrimitiveOp::ElemDiv,
        inputs: vec![0, 1],
        output: Some(2),
    };
    source.inputs = vec![0];
    source.stack_pattern = vec![0];
    source.const_values.insert(1, Value::Num(control));
    source
}

fn invocation(source: &FusionGroupPlan) -> FusionExecutionRequest<'_> {
    FusionExecutionRequest {
        plan: source,
        inputs: vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap(),
        )],
        placement: None,
        runtime: None,
    }
}

#[test]
fn ordinary_reduction_constants_use_sealed_ieee_words_and_preserve_shadowing() {
    runmat_gc::gc_test_context(|| {
        for control in [
            -0.0_f64,
            f64::from_bits(0x7ff8_0000_0000_0042),
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            let source = divide_plan(control);
            assert!(crate::fusion::reduction_emission::structural_support(
                &source
            ));
            let request = invocation(&source);
            let bindings = AdmittedBindings::new(&request).unwrap();
            let selected = admitted_static(&request, &bindings, ProviderPrecision::F64)
                .unwrap()
                .unwrap();
            assert!(selected.emission().is_some());
            let shader = source
                .prepare_reduction_shader(
                    ProviderPrecision::F64,
                    runmat_accelerate_api::ReductionFlavor::Sum,
                    selected.emission(),
                )
                .unwrap();
            let expected = format!(
                "runmat_f64_from_words(0x{:08x}u, 0x{:08x}u)",
                control.to_bits() as u32,
                (control.to_bits() >> 32) as u32
            );
            assert!(shader.source().contains(&expected));
            assert!(shader.source().contains("f64(v) / f64(control1)"));
            assert_eq!(
                shader.requirements(),
                runmat_accelerate_api::ProviderShaderRequirements::F64IeeeWords
            );
            assert!(!shader.source().contains("f64(-0)"));
            #[cfg(feature = "wgpu")]
            crate::wgsl::validation_tests::validate_generated(
                "sealed-ordinary-reduction-constant",
                shader.source(),
                64,
                8,
                crate::wgsl::validation_tests::RequiredCaps::Float64Int64,
            );
            let mut changed = source.clone();
            changed.const_values.insert(1, Value::Num(0.0));
            assert!(selected
                .validate(&changed, &request.inputs, ProviderPrecision::F64, 2, 1)
                .is_err());
        }
        // Unused constants add neither dependencies nor formatted tokens.
        let mut source = divide_plan(f64::NAN);
        source.inputs.push(1);
        let mut request = invocation(&source);
        request.inputs.push(Value::Tensor(
            Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap(),
        ));
        let bindings = AdmittedBindings::new(&request).unwrap();
        let selected = admitted_static(&request, &bindings, ProviderPrecision::F64)
            .unwrap()
            .unwrap();
        let shader = source
            .prepare_reduction_shader(
                ProviderPrecision::F64,
                runmat_accelerate_api::ReductionFlavor::Sum,
                selected.emission(),
            )
            .unwrap();
        assert!(
            !shader.source().contains("control1"),
            "invocation input wins same-ID captured constant"
        );
        source.const_values.remove(&1);
        source.const_values.insert(99, Value::Num(f64::NAN));
        source.const_values.insert(100, Value::Num(f64::INFINITY));
        let mut request = invocation(&source);
        request.inputs.push(Value::Tensor(
            Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap(),
        ));
        let bindings = AdmittedBindings::new(&request).unwrap();
        let selected = admitted_static(&request, &bindings, ProviderPrecision::F64)
            .unwrap()
            .expect("selected producer ignores unused constants");
        let shader = source
            .prepare_reduction_shader(
                ProviderPrecision::F64,
                runmat_accelerate_api::ReductionFlavor::Sum,
                selected.emission(),
            )
            .unwrap();
        assert!(!shader.source().contains("control99"));
        #[cfg(feature = "wgpu")]
        crate::wgsl::validation_tests::validate_generated(
            "unused-reduction-constants",
            shader.source(),
            64,
            8,
            crate::wgsl::validation_tests::RequiredCaps::Float64Int64,
        );
    });
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[tokio::test]
#[ignore = "requires actual F64 WGPU owner; ordinary scalar constant numerical qualification"]
async fn actual_f64_ordinary_reduction_constants_match_host_oracle() {
    use crate::backend::wgpu::provider::{WgpuProvider, WgpuProviderOptions};
    use runmat_accelerate_api::HostNumericDataOwned;
    let provider = Box::leak(Box::new(
        WgpuProvider::new_async(WgpuProviderOptions::default())
            .await
            .expect("actual F64 WGPU owner required"),
    ));
    assert_eq!(
        provider.precision(),
        ProviderPrecision::F64,
        "never substitute F32 hardware"
    );
    // SAFETY: actual facade and its unique allocation owner live forever.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    for control in [
        -0.0_f64,
        f64::from_bits(0x7ff8_0000_0000_0042),
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let source = divide_plan(control);
        let result =
            execute_reduction_with_shape_placed(invocation(&source), 2, 1, &[1, 1], 64).unwrap();
        let FusionExecutionOutcome::Executed(Value::GpuTensor(output)) = result else {
            panic!("ordinary synchronous reduction must execute")
        };
        assert_eq!(output.physical().element_type(), NumericElementType::F64);
        assert_eq!(output.shape(), &[1, 1]);
        let downloaded = provider.download_numeric(&output).await.unwrap();
        let HostNumericDataOwned::F64(actual) = downloaded.data else {
            panic!("Double output required")
        };
        let expected = [1.0_f64, 2.0]
            .into_iter()
            .map(|value| value / control)
            .sum::<f64>();
        if expected.is_nan() {
            assert!(actual[0].is_nan());
        } else {
            assert_eq!(actual[0], expected);
        }
        output.release().unwrap();
    }
}
