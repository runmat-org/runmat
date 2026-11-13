use super::*;

#[test]
fn resident_integer_controls_do_not_round_through_binary64() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let exact = if usize::BITS == 64 {
            u64::MAX
        } else {
            u64::from(u32::MAX)
        };
        let control = Tensor::new_integer(IntegerStorage::U64(vec![exact]), vec![1, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &control).unwrap();
        for kind in [TopKKind::Max, TopKKind::Min] {
            let k = futures::executor::block_on(parse_k(kind, &Value::GpuTensor(handle.clone())))
                .unwrap();
            assert_eq!(k, usize::MAX);
            let args = futures::executor::block_on(parse_topk_args(
                kind,
                crate::call::FunctionArgumentView::from_slice(&[
                    Value::Num(1.),
                    Value::GpuTensor(handle.clone()),
                ]),
            ))
            .unwrap();
            assert_eq!(args.dim, Some(usize::MAX));
        }
        handle.release().unwrap();
    });
}

#[test]
fn topk_gpu_extension_gates_before_gather_and_preserves_residency() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX, 0]),
            vec![3, 1],
        )
        .expect("integer input");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = futures::executor::block_on(maxk_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::Num(2.0)],
            ))
            .expect_err("strict GPU gate");
            assert_eq!(
                err.identifier(),
                Some("RunMat:compatibility:MaxkGpuInputExtension")
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let _outputs =
                crate::output_count::push_output_count(Some(2)).expect("output-count scope");
            let Value::OutputList(outputs) = futures::executor::block_on(maxk_builtin(
                Value::GpuTensor(handle),
                vec![Value::Num(2.0)],
            ))
            .expect("GPU extension") else {
                panic!("expected outputs");
            };
            assert!(matches!(outputs[0], Value::GpuTensor(_)));
            assert!(matches!(outputs[1], Value::GpuTensor(_)));
            let values = crate::builtins::common::test_support::gather(outputs[0].clone())
                .expect("gather values");
            assert_eq!(
                values.into_numeric_storage().unwrap(),
                NumericStorage::U64(vec![u64::MAX, u64::MAX - 1])
            );
            let indices = crate::builtins::common::test_support::gather(outputs[1].clone())
                .expect("gather indices");
            assert_eq!(self::values(&indices), vec![2.0, 1.0]);
        }

        let logical = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[0, 1, 0],
                shape: &[3, 1],
            })
            .expect("logical upload");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = futures::executor::block_on(mink_builtin(
            Value::GpuTensor(logical),
            vec![Value::Num(1.0)],
        ))
        .expect_err("strict logical GPU gate");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:MinkGpuInputExtension")
        );
    });
}
