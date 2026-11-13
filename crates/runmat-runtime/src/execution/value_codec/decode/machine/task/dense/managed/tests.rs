use super::*;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::rc::Rc;

struct Parallel(ManagedHostEffectLedger);
impl RuntimeParallelService for Parallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            cpu_millicores_available: 1_000,
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

fn runtime(capacity: u64) -> (RuntimeContext, ManagedHostEffectLedger, CancellationSource) {
    let ledger = ManagedHostEffectLedger::new_with_epoch(capacity, 1);
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
    );
    (runtime, ledger, cancellation)
}

fn decode(source: &DenseValue, runtime: &RuntimeContext) -> Value {
    let mut task = dense_task(source, runtime).unwrap();
    while !task.is_complete() {
        task.advance(1).unwrap();
    }
    task.finish().unwrap()
}

fn dense_task<'a>(
    source: &'a DenseValue,
    runtime: &'a RuntimeContext,
) -> Result<DenseTask<'a>, ValueCodecError> {
    let control = ControlOwner::new(Some(runtime))?;
    DenseTask::new_with_runtime(source, control.path("$")?, Some(runtime))
}

#[test]
fn managed_native_array_classes_retain_backing_until_last_alias() {
    // Exercise every dense class, including wide integers and NaN payloads.
    let classes = [
        ElementType::Logical,
        ElementType::F32,
        ElementType::F64,
        ElementType::I8,
        ElementType::I16,
        ElementType::I32,
        ElementType::I64,
        ElementType::U8,
        ElementType::U16,
        ElementType::U32,
        ElementType::U64,
        ElementType::ComplexF32,
        ElementType::ComplexF64,
        ElementType::ComplexI8,
        ElementType::ComplexI16,
        ElementType::ComplexI32,
        ElementType::ComplexI64,
        ElementType::ComplexU8,
        ElementType::ComplexU16,
        ElementType::ComplexU32,
        ElementType::ComplexU64,
    ];
    for element_type in classes {
        let width = element_width(element_type);
        let bytes = match element_type {
            ElementType::Logical => vec![1],
            ElementType::F32 => 0x7fa0_0042_u32.to_le_bytes().to_vec(),
            ElementType::F64 => 0x7ff0_0000_0000_0042_u64.to_le_bytes().to_vec(),
            ElementType::ComplexF32 => {
                [0x7fa0_0042_u32.to_le_bytes(), 0x8000_0000_u32.to_le_bytes()].concat()
            }
            ElementType::ComplexF64 => [
                0x7ff0_0000_0000_0042_u64.to_le_bytes(),
                0x8000_0000_0000_0000_u64.to_le_bytes(),
            ]
            .concat(),
            _ => vec![0xff; width],
        };
        let source = DenseValue {
            element_type,
            shape: vec![1, 1],
            little_endian_data: bytes.clone(),
        };
        let (runtime, ledger, _) = runtime(1 << 20);
        let value = decode(&source, &runtime);
        let payload = crate::execution::value_codec::encode_inline_value(&value).unwrap();
        let runmat_execution::value::ValuePayload::Inline(payload) = payload else {
            panic!("inline result")
        };
        let runmat_execution::value::InlineValue::Dense(result) = *payload else {
            panic!("dense result")
        };
        assert_eq!(result.element_type, element_type);
        assert_eq!(result.little_endian_data, bytes);
        let retained = ledger.snapshot().live_reserved_bytes;
        assert!(retained > 0);
        let alias = value.clone();
        drop(value);
        assert_eq!(ledger.snapshot().live_reserved_bytes, retained);
        drop(alias);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

        if !is_complex_integer(element_type) {
            let source = runmat_execution::value::SparseValue {
                element_type,
                rows: 2,
                columns: 1,
                column_offsets: vec![0, 1],
                row_indices: vec![1],
                little_endian_data: bytes,
            };
            let control = ControlOwner::new(Some(&runtime)).unwrap();
            let mut task = super::super::super::sparse::SparseTask::new(
                &source,
                control.path("$").unwrap(),
                Some(&runtime),
            )
            .unwrap();
            let value = loop {
                if let (_, Some(value)) = task.advance(1).unwrap() {
                    break value;
                }
            };
            drop(task);
            drop(control);
            let payload = crate::execution::value_codec::encode_inline_value(&value).unwrap();
            let runmat_execution::value::ValuePayload::Inline(payload) = payload else {
                panic!("inline result")
            };
            let runmat_execution::value::InlineValue::Sparse(result) = *payload else {
                panic!("sparse result")
            };
            assert_eq!(result, source);
            let retained = ledger.snapshot().live_reserved_bytes;
            assert!(retained > 0);
            let alias = value.clone();
            drop(value);
            assert_eq!(ledger.snapshot().live_reserved_bytes, retained);
            drop(alias);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        }
    }
}

#[test]
fn managed_dense_denial_and_cancelled_partial_release_without_publication() {
    let source = DenseValue {
        element_type: ElementType::U64,
        shape: vec![1, 3],
        little_endian_data: vec![0xff; 24],
    };
    let (denied, ledger, _) = runtime(1);
    let scalar = runmat_execution::value::ValuePayload::Inline(Box::new(
        runmat_execution::value::InlineValue::Logical(true),
    ));
    assert!(matches!(
        DecodeMachine::new_with_runtime(&scalar, "$", None, Some(&denied)),
        Err(ValueCodecError::Allocation { .. })
    ));
    assert!(matches!(
        dense_task(&source, &denied),
        Err(ValueCodecError::Allocation { .. })
    ));
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let (runtime, ledger, cancellation) = runtime(1 << 20);
    let mut task = dense_task(&source, &runtime).unwrap();
    task.advance(2).unwrap(); // shape
    task.advance(1).unwrap(); // first native element, exact backing now live
    assert!(ledger.snapshot().live_reserved_bytes > 0);
    cancellation.cancel(CancellationReason::User);
    assert!(matches!(
        task.advance(1),
        Err(ValueCodecError::Allocation {
            source: HostAllocationError::Cancelled,
            ..
        })
    ));
    drop(task);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn cooperative_context_ingress_publishes_the_same_retained_dense_backing() {
    let (runtime, ledger, _) = runtime(1 << 20);
    {
        let scalar = runmat_execution::value::ValuePayload::Inline(Box::new(
            runmat_execution::value::InlineValue::Logical(true),
        ));
        let mut machine = DecodeMachine::new_with_runtime(&scalar, "$", None, Some(&runtime))
            .expect("admit decoder before its first control block");
        assert!(ledger.snapshot().live_reserved_bytes > 0);
        for _ in 0..super::super::super::super::TASK_CHUNK_CAPACITY * 2 {
            machine
                .tasks
                .push(Task::Decode {
                    payload: &scalar,
                    path: machine.control.path("$").unwrap(),
                })
                .unwrap();
        }
        let path = machine.control.path("$.nested").unwrap();
        let path = path.extended(format_args!(".field[{}]", 64)).unwrap();
        assert_eq!(&*path, "$.nested.field[64]");
        let retained = ledger.snapshot().live_reserved_bytes;
        runtime.invalidate_host_effect_session_authority();
        assert!(matches!(
            path.extended(format_args!(".after_rotation")),
            Err(ValueCodecError::Allocation {
                source: HostAllocationError::StaleAuthority,
                ..
            })
        ));
        assert_eq!(
            ledger.snapshot().live_reserved_bytes,
            retained,
            "a stale owner cannot allocate using a freshly captured receipt"
        );
        drop(machine);
        assert!(
            ledger.snapshot().live_reserved_bytes > 0,
            "path retains its own physical control claim"
        );
        drop(path);
    }
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let payload = runmat_execution::value::ValuePayload::Inline(Box::new(
        runmat_execution::value::InlineValue::Dense(DenseValue {
            element_type: ElementType::U64,
            shape: vec![1, 2],
            little_endian_data: [u64::MAX.to_le_bytes(), (1_u64 << 63).to_le_bytes()].concat(),
        }),
    ));
    let result = futures::executor::block_on(
        crate::execution::value_codec::decode_inline_values_cooperatively_in_context(
            &[payload],
            runmat_execution::value::ValueLimits::default(),
            None,
            None,
            &runtime,
        ),
    )
    .expect("canonical context decode");
    let alias = result.values()[0].clone();
    assert!(ledger.snapshot().live_reserved_bytes > 0);
    drop(result);
    assert!(
        ledger.snapshot().live_reserved_bytes > 0,
        "returned payload retains its own claim independently of the final argument collection"
    );
    drop(alias);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    // Aggregate destinations retain their actual child/text/shape/header
    // owners too, not just the decoder's temporary control allocations.
    for input in [
        Value::Cell(runmat_value::CellArray::new(vec![Value::Num(7.0)], 1, 1).unwrap()),
        Value::function_handle("named_destination".into()),
        Value::Closure(runmat_value::Closure::new(
            "captured_destination".into(),
            None,
            runmat_types::FunctionSourceRole::Anonymous,
            Some("retained_source.m".into()),
            vec![Value::Num(9.0)],
            None,
        )),
        Value::MException(runmat_value::MException::from_owned_fields(
            "RunMat:Decoded".into(),
            "public exception".into(),
            runmat_value::MException::stack_from_unbounded(vec!["frame".into()]),
            None,
        )),
    ] {
        let payload = crate::execution::value_codec::encode_inline_value(&input).unwrap();
        let result = futures::executor::block_on(
            crate::execution::value_codec::decode_inline_values_cooperatively_in_context(
                &[payload],
                runmat_execution::value::ValueLimits::default(),
                None,
                None,
                &runtime,
            ),
        )
        .expect("admit original aggregate destination");
        let alias = result.values()[0].clone();
        assert_eq!(alias, input);
        match &alias {
            Value::Cell(cell) => {
                assert!(cell.data_owner().is_exact());
                assert!(cell.shape.is_admitted());
            }
            Value::Closure(closure) => assert!(closure.captures.is_exact()),
            _ => {}
        }
        drop(result);
        assert!(ledger.snapshot().live_reserved_bytes > 0);
        drop(alias);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn dropping_suspended_managed_decode_retires_partial_native_backing() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use std::{
        cell::Cell,
        future::Future,
        num::NonZeroU64,
        pin::Pin,
        task::{Context, Poll},
    };

    struct PendingTurn(Rc<Cell<usize>>);
    impl Future for PendingTurn {
        type Output = Result<(), ExecutionServiceError>;
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }
    impl Drop for PendingTurn {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    let elements = crate::execution::value_codec::decode::DECODE_WORK_QUANTUM.get() * 4;
    let payload_bytes = elements * std::mem::size_of::<u64>();
    // The native payload is larger than all pre-payload managed control
    // allocations together, so its live claim is distinguishable from a mere
    // stage reservation or final argument-collection allocation.
    let control_bytes = collection_block_retained_bytes(Layout::new::<ManagedDense<'_>>()).unwrap()
        + HostTypedWorkspace::<usize>::quoted_bytes(2).unwrap()
        + crate::context::host_allocation::FrameCollectionGate::retained_bytes().unwrap()
        + collection_block_retained_bytes(
            Layout::array::<Task<'_>>(super::super::super::super::TASK_CHUNK_CAPACITY).unwrap(),
        )
        .unwrap()
        + collection_block_retained_bytes(Layout::array::<u8>(1).unwrap()).unwrap()
        + u64::try_from(runmat_value::OutputListStorage::requested_bytes(1).unwrap()).unwrap();
    assert!(control_bytes < u64::try_from(payload_bytes).unwrap());
    let payload = runmat_execution::value::ValuePayload::Inline(Box::new(
        runmat_execution::value::InlineValue::Dense(DenseValue {
            element_type: ElementType::U64,
            shape: vec![1, elements as u64],
            little_endian_data: vec![0xff; payload_bytes],
        }),
    ));
    let (runtime, ledger, cancellation) = runtime(1 << 20);
    let observed = Rc::new(Cell::new(false));
    let observed_in_turn = observed.clone();
    let dropped_turns = Rc::new(Cell::new(0));
    let dropped_in_turn = dropped_turns.clone();
    let observed_ledger = ledger.clone();
    let mut budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        cancellation.token(),
        move || {
            if observed_ledger.snapshot().live_claimed_bytes >= payload_bytes as u64 {
                observed_in_turn.set(true);
                Box::pin(PendingTurn(dropped_in_turn.clone()))
            } else {
                // Validation and pre-payload turns must not suspend the test.
                Box::pin(std::future::ready(Ok(())))
            }
        },
    );
    let mut future = Box::pin(
        crate::execution::value_codec::decode_inline_values_cooperatively_in_context(
            std::slice::from_ref(&payload),
            runmat_execution::value::ValueLimits::default(),
            None,
            Some(&mut budget),
            &runtime,
        ),
    );
    let waker = futures::task::noop_waker();
    // Four work quanta of source elements cannot finish before the first
    // post-allocation turn; no RootedValues result has been published.
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    assert!(
        observed.get(),
        "suspended only after native claims became live"
    );
    assert!(ledger.snapshot().live_claimed_bytes >= payload_bytes as u64);
    drop(future); // Do not advance or cancel the decoder to trigger cleanup.
    assert_eq!(dropped_turns.get(), 1);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn managed_decode_member_roots_gc_handle_and_releases_after_cancelled_drop() {
    runmat_gc::gc_test_context(|| {
        let (runtime, ledger, cancellation) = runtime(1 << 20);
        let initial = runmat_gc::gc_allocate_rooted(Value::string("decoded handle"))
            .expect("managed payload");
        let handle = initial.handle();
        let member = crate::execution::RootedValueMember::root_in_context(
            Value::HandleObject(runmat_value::HandleRef {
                class_name: "DecodedHandle".into(),
                target: handle,
                valid: true,
            }),
            &runtime,
        )
        .expect("admitted construction roots");
        initial.unroot().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        assert!(ledger.snapshot().live_reserved_bytes > 0);
        let roots_only = ledger.snapshot().live_claimed_bytes;
        let mut children = crate::execution::RootedValueChildren::in_context(&runtime).unwrap();
        let value = member.into_child(&mut children).unwrap();
        assert!(
            ledger.snapshot().live_claimed_bytes > roots_only,
            "actual nonempty child roots admit their retained slot/gate metadata"
        );
        let member = crate::execution::RootedValueMember::aggregate(value, children).unwrap();
        cancellation.cancel(CancellationReason::User);
        drop(member);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    });
}
