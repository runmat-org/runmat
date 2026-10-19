//! Required capable-device Double folded reduction gate. The ordinary F32
//! suite remains separately runnable on adapters without F64 IEEE shaders.

use super::*;

#[test]
#[ignore = "requires actual SHADER_F64 and SHADER_INT64 WGPU hardware; RUNMAT_WGPU_FORCE_PRECISION=f64"]
fn actual_vm_folded_double_sum_keeps_f64_storage_and_full_call_feedback() {
    runmat_gc::gc_test_context(|| {
        assert_eq!(
            std::env::var("RUNMAT_WGPU_FORCE_PRECISION").as_deref(),
            Ok("f64"),
            "run this required Double hardware gate in a separate F64-selected process"
        );
        let provider = register_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual F64 WGPU adapter required; unavailable hardware is a failed gate");
        assert_eq!(
            runmat_accelerate_api::AccelProvider::precision(provider),
            runmat_accelerate_api::ProviderPrecision::F64,
            "F32 provider fallback cannot qualify the Double route"
        );
        assert!(
            runmat_accelerate_api::AccelProvider::capability_snapshot(provider)
                .element_types
                .contains(&runmat_accelerate_api::ProviderElementType::F64),
            "selected provider must advertise F64 storage"
        );
        assert!(
            runmat_accelerate_api::AccelProvider::supports_shader_requirements(
                provider,
                runmat_accelerate_api::ProviderShaderRequirements::F64IeeeWords,
            ),
            "selected device must enable both SHADER_F64 and SHADER_INT64"
        );

        for decoded_function in [false, true] {
            block_on(typed_elementwise_call::assert_original_sum_full_call(
                decoded_function,
            ));
        }

        // The first source element exceeds F32's exact-integer range. Its
        // 128-row Double sum has low bits that an F32 output cannot retain.
        let bytecode = test_helpers::compile_source(
            "x = gpuArray(reshape(16777217:16793600,128,128)); y = sum(x+x); z = gather(y);",
        )
        .expect("compile authentic folded Double Sum");
        assert_compiled_folded_graph(&bytecode, 1);
        let expected = (0..128_u64)
            .map(|column| 256.0 * (16_777_217 + 128 * column) as f64 + 128.0 * 127.0)
            .collect::<Vec<_>>();
        assert_ne!(expected[0], f64::from(expected[0] as f32));
        let vars = assert_actual_folded_full_call_shape_with_dtype(
            &bytecode,
            &[1, 128],
            &expected,
            NumericDType::F64,
        );
        for (name, shape) in [("x", &[128, 128][..]), ("y", &[1, 128][..])] {
            let slot = bytecode
                .var_names
                .iter()
                .find_map(|(slot, variable)| (variable == name).then_some(*slot))
                .expect("Double source and result have published VM slots");
            let Value::GpuTensor(handle) = &vars[slot] else {
                panic!("Double {name} must remain an authentic resident handle");
            };
            handle
                .allocation_lease()
                .validate_live()
                .expect("Double source and output retain live physical owners");
            assert_eq!(handle.shape(), shape);
            assert_eq!(
                handle.physical().element_type(),
                runmat_accelerate_api::NumericElementType::F64
            );
            assert_eq!(
                handle.physical().storage(),
                runmat_accelerate_api::GpuTensorStorage::Real
            );
        }
    });
}
