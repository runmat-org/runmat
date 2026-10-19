use super::*;
use runmat_value::NumericScalar;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn exact_float_replay_preserves_ragged_empty_values_and_retained_claim() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (_restore, selected) = selected_path(b"1,,3\n,5\n");
    for (dtype, empty) in [
        (NumericDType::F64, None),
        (NumericDType::F32, None),
        (NumericDType::F64, Some(-7.25)),
        (NumericDType::F32, Some(-7.25)),
    ] {
        let mut source = block_on(selected.open_bounded_input())
            .expect("bounded opened source")
            .expect("direct memory snapshot");
        let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
        let options = ReadMatrixOptions {
            delimiter: Some(Delimiter::Char(',')),
            output_template: OutputTemplate::Numeric(dtype),
            empty_value: empty.map(NumericScalar::F64),
            ..ReadMatrixOptions::default()
        };
        let geometry = block_on(bounded::scan_utf8(
            &mut source,
            &mut cooperation,
            &selected,
            &options,
            bounded::ScanResources::default(),
        ))
        .expect("whole-source validation");
        let live = Arc::new(AtomicUsize::new(0));
        let permit = ExactOutputTestPermit(live.clone());
        let tensor = block_on(bounded::fill_utf8(
            &mut source,
            &geometry,
            &mut cooperation,
            &selected,
            &options,
            bounded::FillResources {
                output: Some(&permit),
                ..Default::default()
            },
        ))
        .expect("exact ragged output");
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.numeric_dtype(), dtype);
        let bytes = runmat_value::ExactHostTensor2dAllocation::requested_bytes(dtype, 2, 3)
            .expect("same physical output owner requested layout");
        assert_eq!(live.load(Ordering::SeqCst), bytes);
        let expected = [
            1.0,
            empty.unwrap_or(f64::NAN),
            empty.unwrap_or(f64::NAN),
            5.0,
            3.0,
            empty.unwrap_or(f64::NAN),
        ];
        let actual = match dtype {
            NumericDType::F64 => tensor.as_f64_slice().unwrap().to_vec(),
            NumericDType::F32 => tensor
                .as_f32_slice()
                .unwrap()
                .iter()
                .map(|value| f64::from(*value))
                .collect(),
            _ => unreachable!("floating cases only"),
        };
        for (actual, expected) in actual.iter().zip(expected) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(*actual, expected);
            }
        }
        let shape_pointer = tensor.shape().as_ptr();
        let payload_pointer = unsafe { tensor.host_buffer().foreign_data_pointer() };
        let output = super::super::super::output::finalize_output(tensor, &options)
            .expect("same-class final presentation");
        let Value::Tensor(output) = output else {
            panic!("numeric output remains a tensor")
        };
        assert_eq!(output.shape().as_ptr(), shape_pointer);
        assert_eq!(
            unsafe { output.host_buffer().foreign_data_pointer() },
            payload_pointer
        );
        assert_eq!(live.load(Ordering::SeqCst), bytes);
        drop(output);
        assert_eq!(live.load(Ordering::SeqCst), 0);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn exact_output_claim_releases_on_changed_source_digest() {
    let _lock = runmat_filesystem::provider_override_lock();
    let directory = tempfile::tempdir().expect("isolated numeric-text source");
    let file = directory.path().join("numbers.csv");
    std::fs::write(&file, b"1,2\n3,4\n").expect("initial payload");
    let _restore =
        runmat_filesystem::replace_provider(Arc::new(runmat_filesystem::NativeFsProvider));
    let selected = TabularFsOwner::capture(Some(&file))
        .expect("captured provider")
        .path(file.clone())
        .expect("selected source");
    let mut opened = block_on(selected.open_bounded_input())
        .expect("direct opened source")
        .expect("bounded source");
    let options = ReadMatrixOptions {
        delimiter: Some(Delimiter::Char(',')),
        output_template: OutputTemplate::Numeric(NumericDType::U64),
        ..ReadMatrixOptions::default()
    };
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let geometry = block_on(bounded::scan_utf8(
        &mut opened,
        &mut cooperation,
        &selected,
        &options,
        bounded::ScanResources::default(),
    ))
    .expect("first pass");
    std::fs::write(&file, b"9,8\n7,6\n").expect("same-shape mutation");
    let live = Arc::new(AtomicUsize::new(0));
    let permit = ExactOutputTestPermit(live.clone());
    let error = block_on(bounded::fill_utf8(
        &mut opened,
        &geometry,
        &mut cooperation,
        &selected,
        &options,
        bounded::FillResources {
            output: Some(&permit),
            ..Default::default()
        },
    ))
    .expect_err("changed bytes cannot publish the charged output");
    assert!(error
        .message()
        .contains("changed between validation and output"));
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn exact_output_claim_releases_on_pending_drop_and_cancellation() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (_restore, selected) = selected_path(b"1,2\n");
    let mut opened = block_on(selected.open_bounded_input())
        .expect("bounded opened source")
        .expect("direct memory snapshot");
    let options = ReadMatrixOptions {
        delimiter: Some(Delimiter::Char(',')),
        output_template: OutputTemplate::Numeric(NumericDType::F64),
        ..ReadMatrixOptions::default()
    };
    let mut scan_cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let geometry = block_on(bounded::scan_utf8(
        &mut opened,
        &mut scan_cooperation,
        &selected,
        &options,
        bounded::ScanResources::default(),
    ))
    .expect("first pass");
    let live = Arc::new(AtomicUsize::new(0));
    let permit = ExactOutputTestPermit(live.clone());
    let source = CancellationSource::new();
    let runtime =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), source.token());
    let claimed =
        runmat_value::ExactHostTensor2dAllocation::requested_bytes(NumericDType::F64, 1, 2)
            .expect("same physical output owner requested layout");
    for cancel in [false, true] {
        let dropped_turn = Rc::new(Cell::new(false));
        let dropped_turn_owner = dropped_turn.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("nonzero")),
            source.token(),
            move || -> crate::context::RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
                Box::pin(PendingTurn(dropped_turn_owner.clone()))
            },
        );
        let mut cooperation = TabularCooperation::with_test_budget(&runtime, BUILTIN_NAME, budget);
        let mut call = Box::pin(bounded::fill_utf8(
            &mut opened,
            &geometry,
            &mut cooperation,
            &selected,
            &options,
            bounded::FillResources {
                output: Some(&permit),
                ..Default::default()
            },
        ));
        let waker = futures::task::noop_waker();
        assert!(matches!(
            call.as_mut().poll(&mut Context::from_waker(&waker)),
            Poll::Pending
        ));
        assert_eq!(live.load(Ordering::SeqCst), claimed);
        if cancel {
            source.cancel(runmat_execution::CancellationReason::User);
            let error = block_on(call).expect_err("cancelled replay cannot publish");
            assert!(error
                .identifier()
                .is_some_and(|id| id.ends_with(":Cancelled")));
        } else {
            drop(call);
        }
        assert!(dropped_turn.get(), "pending service turn was released");
        assert_eq!(live.load(Ordering::SeqCst), 0);
    }
}
