use super::*;
use crate::execution::CpuCooperationPolicy;
use futures::task::noop_waker;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::value::{
    CallableForm, CallableOwner, CallableValue, DenseValue, DurableStructArray, ElementType,
    InlineValue, StructField,
};
use runmat_execution::CancellationReason;
use runmat_types::{FunctionSourceRole, ProgramFunctionId, SourceId};
use std::cell::Cell;
use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Context, Poll};

fn large_dense(elements: usize) -> ValuePayload {
    ValuePayload::Inline(Box::new(InlineValue::Dense(DenseValue {
        element_type: ElementType::F64,
        shape: vec![1, elements as u64],
        little_endian_data: (0..elements)
            .flat_map(|index| (index as f64).to_bits().to_le_bytes())
            .collect(),
    })))
}

#[test]
fn synchronous_and_never_yield_frontends_are_bit_exact() {
    let payload = ValuePayload::Inline(Box::new(InlineValue::Cell {
        shape: vec![1, 2],
        values: vec![
            ValuePayload::Inline(Box::new(InlineValue::F64Bits(0x7ff8_0000_0000_0042))),
            large_dense(1025),
        ],
    }));
    let expected = decode_inline_value(&payload).expect("synchronous decode");
    let source = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let observed_turns = turns.clone();
    let mut budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::NeverYield,
        source.token(),
        move || {
            observed_turns.set(observed_turns.get() + 1);
            Box::pin(async { Ok(()) })
        },
    );
    let actual = futures::executor::block_on(decode_inline_values_cooperatively(
        std::slice::from_ref(&payload),
        ValueLimits::default(),
        None,
        Some(&mut budget),
    ))
    .expect("cooperative decode");
    assert_eq!(turns.get(), 0);
    let [Value::Cell(actual)] = actual.values() else {
        panic!("cooperative result should preserve the cell");
    };
    let Value::Cell(expected) = expected else {
        panic!("synchronous result should preserve the cell");
    };
    let (Value::Num(actual_nan), Value::Num(expected_nan)) =
        (&actual.data()[0], &expected.data()[0])
    else {
        panic!("first cell member should preserve the NaN scalar");
    };
    assert_eq!(actual_nan.to_bits(), expected_nan.to_bits());
    assert_eq!(actual.data()[1], expected.data()[1]);
}

#[test]
fn durable_batch_uses_one_shared_validation_budget_and_one_grammar() {
    let payloads = vec![
        ValuePayload::Inline(Box::new(InlineValue::String("four".into()))),
        ValuePayload::Inline(Box::new(InlineValue::String("more".into()))),
    ];
    let limits = ValueLimits {
        max_text_bytes: 7,
        ..ValueLimits::default()
    };
    assert!(matches!(
        futures::executor::block_on(decode_durable_values_cooperatively(&payloads, limits, None)),
        Err(CooperativeValueCodecError::Value(
            ValueCodecError::Invalid { .. }
        ))
    ));

    let ordered = ValuePayload::Inline(Box::new(InlineValue::DurableOrderedStruct(vec![
        StructField {
            name: "answer".into(),
            value: ValuePayload::Inline(Box::new(InlineValue::I32(42))),
        },
    ])));
    let expected = decode_durable_inline_value(&ordered).expect("synchronous durable decode");
    let actual = futures::executor::block_on(decode_durable_values_cooperatively(
        std::slice::from_ref(&ordered),
        ValueLimits::default(),
        None,
    ))
    .expect("cooperative durable decode");
    assert_eq!(actual.values(), std::slice::from_ref(&expected));
}

#[test]
fn synchronous_and_cooperative_frontends_preserve_decoder_errors() {
    let payload = ValuePayload::Inline(Box::new(InlineValue::Null));
    let expected = decode_inline_value(&payload).expect_err("null is not a runtime value");
    let actual = futures::executor::block_on(decode_inline_values_cooperatively(
        std::slice::from_ref(&payload),
        ValueLimits::default(),
        None,
        None,
    ))
    .expect_err("cooperative decoder must reject null");
    assert_eq!(actual, CooperativeValueCodecError::Value(expected));
}

#[test]
fn pending_turn_drop_retires_partial_numeric_machine_without_replay() {
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

    let payload = large_dense(DECODE_WORK_QUANTUM.get() * 2);
    let source = CancellationSource::new();
    let dropped = Rc::new(Cell::new(0));
    let dropped_turn = dropped.clone();
    let mut budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || Box::pin(PendingTurn(dropped_turn.clone())),
    );
    let mut future = Box::pin(decode_inline_values_cooperatively(
        std::slice::from_ref(&payload),
        ValueLimits {
            max_inline_bytes: (payload_size(&payload) as u64).saturating_add(64),
            max_elements: (DECODE_WORK_QUANTUM.get() as u64) * 3,
            ..ValueLimits::default()
        },
        None,
        Some(&mut budget),
    ));
    let waker = noop_waker();
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    drop(future);
    assert_eq!(dropped.get(), 1);
}

#[test]
fn complex_shape_admission_uses_selected_budget_and_retires_pending_turn() {
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

    // The first machine quantum creates the dense task and copies all but one
    // shape entry. The second copies that final entry and then traverses the
    // owned shape through ComplexTensorAdmission, so its selected turn cannot
    // be reached by payload copying alone.
    let payload = ValuePayload::Inline(Box::new(InlineValue::Dense(DenseValue {
        element_type: ElementType::ComplexF64,
        shape: std::iter::once(0)
            .chain(std::iter::repeat_n(1, DECODE_WORK_QUANTUM.get() - 1))
            .collect(),
        little_endian_data: Vec::new(),
    })));
    let source = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let observed_turns = turns.clone();
    let dropped = Rc::new(Cell::new(0));
    let dropped_turn = dropped.clone();
    let mut budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(
            NonZeroU64::new(DECODE_WORK_QUANTUM.get() as u64).unwrap(),
        ),
        source.token(),
        move || {
            let turn = observed_turns.get() + 1;
            observed_turns.set(turn);
            if turn == 1 {
                Box::pin(std::future::ready(Ok(())))
            } else {
                Box::pin(PendingTurn(dropped_turn.clone()))
            }
        },
    );
    let payloads = [payload];
    let mut cooperation = Some(&mut budget);
    // Enter below canonical payload validation so validation work cannot be
    // mistaken for the two decoder turns asserted by this admission witness.
    let mut future = Box::pin(decode_validated_values(
        &payloads,
        None,
        &mut cooperation,
        None,
    ));
    let waker = noop_waker();
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    assert_eq!(turns.get(), 2);
    source.cancel(CancellationReason::User);
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Ready(Err(CooperativeValueCodecError::Execution(
            ExecutionServiceError::Cancelled
        )))
    ));
    drop(future);
    assert_eq!(dropped.get(), 1);
}

#[test]
fn struct_array_admission_uses_selected_budget_and_retires_pending_turn() {
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

    let payload = ValuePayload::Inline(Box::new(InlineValue::DurableStructArray(
        DurableStructArray {
            shape: std::iter::once(0)
                .chain(std::iter::repeat_n(1, DECODE_WORK_QUANTUM.get() - 1))
                .collect(),
            fields: Vec::new(),
        },
    )));
    let source = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let observed_turns = turns.clone();
    let dropped = Rc::new(Cell::new(0));
    let dropped_turn = dropped.clone();
    let mut budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(
            NonZeroU64::new(DECODE_WORK_QUANTUM.get() as u64).unwrap(),
        ),
        source.token(),
        move || {
            let turn = observed_turns.get() + 1;
            observed_turns.set(turn);
            if turn == 1 {
                Box::pin(std::future::ready(Ok(())))
            } else {
                Box::pin(PendingTurn(dropped_turn.clone()))
            }
        },
    );
    let payloads = [payload];
    let mut cooperation = Some(&mut budget);
    // Enter below canonical payload validation. The first turn reaches the
    // durable task's shape copy; the second reaches StructArrayAdmission, so
    // the Pending turn cannot be attributed to validation work.
    let mut future = Box::pin(decode_validated_values(
        &payloads,
        None,
        &mut cooperation,
        None,
    ));
    let waker = noop_waker();
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    assert_eq!(turns.get(), 2);
    source.cancel(CancellationReason::User);
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Ready(Err(CooperativeValueCodecError::Execution(
            ExecutionServiceError::Cancelled
        )))
    ));
    drop(future);
    assert_eq!(dropped.get(), 1);
}

#[test]
fn selected_cancellation_is_typed_and_drops_the_pending_turn() {
    let payload = large_dense(DECODE_WORK_QUANTUM.get() * 2);
    let source = CancellationSource::new();
    let mut budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        || Box::pin(std::future::pending::<Result<(), ExecutionServiceError>>()),
    );
    let limits = ValueLimits {
        max_inline_bytes: (payload_size(&payload) as u64).saturating_add(64),
        max_elements: (DECODE_WORK_QUANTUM.get() as u64) * 3,
        ..ValueLimits::default()
    };
    let mut future = Box::pin(decode_inline_values_cooperatively(
        std::slice::from_ref(&payload),
        limits,
        None,
        Some(&mut budget),
    ));
    let waker = noop_waker();
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    source.cancel(CancellationReason::User);
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Ready(Err(CooperativeValueCodecError::Execution(
            ExecutionServiceError::Cancelled
        )))
    ));
}

#[test]
fn completed_scalar_members_share_one_batch_quantum_and_drop_pending_turn() {
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

    let payloads = (0..DECODE_WORK_QUANTUM.get())
        .map(|index| ValuePayload::Inline(Box::new(InlineValue::I32(index as i32))))
        .collect::<Vec<_>>();
    let source = CancellationSource::new();
    let dropped = Rc::new(Cell::new(0));
    let dropped_turn = dropped.clone();
    let mut budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(
            NonZeroU64::new(DECODE_WORK_QUANTUM.get() as u64).unwrap(),
        ),
        source.token(),
        move || Box::pin(PendingTurn(dropped_turn.clone())),
    );
    let mut future = Box::pin(decode_inline_values_cooperatively(
        &payloads,
        ValueLimits::default(),
        None,
        Some(&mut budget),
    ));
    let waker = noop_waker();
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    drop(future);
    assert_eq!(dropped.get(), 1);
}

#[test]
fn exact_resolvers_remain_invocation_owned_across_sibling_pending_and_drop() {
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let origin_a = runmat_execution::FunctionBindingOrigin {
        program: runmat_execution::Digest::sha256("resolver A"),
        function: ProgramFunctionId(1),
    };
    let origin_b = runmat_execution::FunctionBindingOrigin {
        program: runmat_execution::Digest::sha256("resolver B"),
        function: ProgramFunctionId(1),
    };
    let owner_a = Arc::new(runmat_value::FunctionDefinitionOwner::default());
    let owner_b = Arc::new(runmat_value::FunctionDefinitionOwner::default());
    let resolver_a = exact_resolver(origin_a, 41, Arc::clone(&owner_a));
    let resolver_b = exact_resolver(origin_b, 73, Arc::clone(&owner_b));

    let ambient = runtime.enter();
    let ambient_resolver =
        crate::user_functions::install_execution_source_resolver(Some(Rc::clone(&resolver_a)))
            .expect("call-port scope");
    drop(ambient);

    let pending_payload = bound_callable(origin_a, &"a".repeat(DECODE_WORK_QUANTUM.get() * 2));
    let pending_source = CancellationSource::new();
    let mut pending_budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        pending_source.token(),
        || Box::pin(std::future::pending::<Result<(), ExecutionServiceError>>()),
    );
    let mut pending = Box::pin(runtime.clone().scope(decode_inline_values_cooperatively(
        std::slice::from_ref(&pending_payload),
        ValueLimits::default(),
        Some(resolver_a),
        Some(&mut pending_budget),
    )));
    let waker = noop_waker();
    assert!(pending
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());

    let selected_b = bound_callable(origin_b, "selected_b");
    let decoded_b =
        futures::executor::block_on(runtime.clone().scope(decode_inline_values_cooperatively(
            std::slice::from_ref(&selected_b),
            ValueLimits::default(),
            Some(resolver_b),
            None,
        )))
        .expect("sibling exact resolver");
    let [Value::BoundFunctionHandle(handle)] = decoded_b.values() else {
        panic!("bound callable result");
    };
    assert_eq!(handle.function, 73);
    assert!(handle
        .authority
        .local_lease(73)
        .is_some_and(|lease| lease.matches(&owner_b)));
    assert!(handle.authority.local_lease(41).is_none());

    drop(pending);
    drop(ambient_resolver);
}

fn bound_callable(origin: runmat_execution::FunctionBindingOrigin, name: &str) -> ValuePayload {
    let form = CallableForm::Handle;
    let role = FunctionSourceRole::Primary;
    ValuePayload::Inline(Box::new(InlineValue::Callable(CallableValue {
        owner: CallableOwner::Bound(origin),
        qualified_name: name.into(),
        callable_digest: CallableValue::identity_digest(
            CallableOwner::Bound(origin),
            name,
            form,
            role,
            None,
        ),
        form,
        source_role: role,
        source_file: None,
        captures: Vec::new(),
    })))
}

fn exact_resolver(
    origin: runmat_execution::FunctionBindingOrigin,
    function: usize,
    owner: Arc<runmat_value::FunctionDefinitionOwner>,
) -> Rc<crate::user_functions::ExecutionSourceResolver> {
    let info = crate::user_functions::SourceFunctionInfo {
        source_id: SourceId(function),
        name: format!("function_{function}"),
        function,
        role: FunctionSourceRole::Primary,
        binding_origin: Some(origin),
        source_file: None,
        definition_owners: Some(Arc::new(vec![(function, owner)])),
    };
    let by_function = info.clone();
    let by_source = info.clone();
    let by_owners = info.definition_owners.clone();
    let by_origin = info;
    Rc::new(crate::user_functions::ExecutionSourceFacts::new(
        Rc::new(move |requested| (requested == function).then(|| by_function.clone())),
        Rc::new(move |source| {
            (source == SourceId(function))
                .then(|| by_source.clone())
                .into_iter()
                .collect()
        }),
        Rc::new(move |requested| (requested == function).then(|| by_owners.clone()).flatten()),
        Rc::new(move |requested| (requested == origin).then(|| by_origin.clone())),
    ))
}

fn payload_size(payload: &ValuePayload) -> usize {
    let ValuePayload::Inline(value) = payload else {
        return 0;
    };
    let InlineValue::Dense(value) = value.as_ref() else {
        return 0;
    };
    value.little_endian_data.len()
}
