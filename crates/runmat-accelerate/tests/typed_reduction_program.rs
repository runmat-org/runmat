//! Production joins and admission ordering, not a GPU numerical execution test.
#[path = "support/typed_fusion_plan.rs"]
mod fixture;
#[path = "support/reduction_program.rs"]
mod support;
#[path = "support/direct_reduction_runtime.rs"]
mod invocation;
use futures::executor::block_on;
use runmat_accelerate::fusion::FusionOp;
use runmat_accelerate::fusion_exec::resolve_reduction_program;
use runmat_accelerate::graph::PrimitiveOp;
use runmat_accelerate_api::{ProviderPrecision, ThreadProviderGuard};
use runmat_runtime::context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts};
use runmat_types::NumericClass;
use runmat_value::{Tensor, Value};
use std::{
    cell::Cell,
    error::Error,
    future::Future,
    rc::Rc,
    sync::atomic::Ordering,
    task::{Context, Poll},
};
use support::*;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

fn tensor(single: bool, shape: Vec<usize>) -> Value {
    let count = shape.iter().product();
    Value::Tensor(if single {
        Tensor::from_f32(vec![2.0; count], shape).unwrap()
    } else {
        Tensor::new(vec![2.0; count], shape).unwrap()
    })
}
fn add(left: u32, right: u32, output: u32) -> FusionOp {
    FusionOp::Primitive {
        op: PrimitiveOp::Add,
        inputs: vec![left, right],
        output: Some(output),
    }
}

struct ManagedControls(runmat_runtime::context::host_effect::ManagedHostEffectLedger);
impl runmat_runtime::context::RuntimeParallelService for ManagedControls {
    fn supports(&self, _: runmat_runtime::context::ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> runmat_runtime::context::RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        runmat_runtime::context::RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..Default::default()
        }
    }
    fn managed_host_effect_ledger(
        &self,
    ) -> Option<runmat_runtime::context::host_effect::ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn actual_host_controls_fail_before_any_producer_probe_for_all_four_owners() {
    let owner = new_owner();
    let _provider = ThreadProviderGuard::set(Some(owner));
    for managed in [false, true] {
        for name in ["sum", "mean", "min", "max"] {
            owner.reset();
            let left = owner.input();
            let right = owner.input();
            let mut controls = Vec::new();
            if matches!(name, "min" | "max") {
                controls.push(tensor(false, vec![0, 0]));
            }
            controls.push(Value::Num(0.5));
            let mut ids = vec![0, 1];
            ids.extend((0..controls.len()).map(|index| 2 + index as u32));
            let mut terminal = vec![10];
            terminal.extend_from_slice(&ids[2..]);
            let plan = reduction(
                fixture::plan(
                    vec![
                        builtin("power", vec![0, 1], 10),
                        builtin(name, terminal, 11),
                    ],
                    11,
                ),
                ids,
                10,
            );
            let mut values = vec![
                Value::GpuTensor(left.handle.clone()),
                Value::GpuTensor(right.handle.clone()),
            ];
            values.extend(controls.iter().cloned());
            let mut request = request(&plan, values);
            let ledger = managed.then(|| {
                runmat_runtime::context::host_effect::ManagedHostEffectLedger::new(1 << 20)
            });
            if let Some(ledger) = &ledger {
                let runtime = invocation::runtime();
                let ports = runtime
                    .service_ports()
                    .clone()
                    .with_parallel(Rc::new(ManagedControls(ledger.clone())));
                request.runtime = Some(runtime.with_service_ports(ports));
            }
            let failure =
                match block_on(resolve_reduction_program(&request, ProviderPrecision::F64)) {
                    Err(error) => error,
                    _ => panic!("{name}: invalid axis must retain the owner's parser error"),
                };
            let actual = failure
                .source()
                .unwrap()
                .downcast_ref::<runmat_runtime::RuntimeError>()
                .expect("actual parser error source");
            let mut host_args = vec![tensor(false, vec![2, 2])];
            host_args.extend(controls);
            let expected = block_on(runmat_runtime::call_builtin_async_with_outputs(
                name, &host_args, 1,
            ))
            .unwrap_err();
            assert_eq!(actual.identifier(), expected.identifier());
            assert_eq!(actual.to_string(), expected.to_string());
            assert_eq!(owner.calls.load(Ordering::SeqCst), 0);
            if let Some(ledger) = &ledger {
                let snapshot = ledger.snapshot();
                assert!(
                    snapshot.peak_reserved_bytes > 0,
                    "the original managed owner funds parser admission"
                );
                assert_eq!(
                    snapshot.live_reserved_bytes, 0,
                    "failed parser releases its original control claims"
                );
            }
            assert_eq!(left.probe.retirements(), 0);
            assert_eq!(right.probe.retirements(), 0);
            drop(request);
            drop(left.handle);
            drop(right.handle);
            assert_eq!(left.probe.retirements(), 1);
            assert_eq!(right.probe.retirements(), 1);
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn rejected_axis_does_not_bypass_an_exhausted_managed_control_owner() {
    let owner = new_owner();
    let _provider = ThreadProviderGuard::set(Some(owner));
    let left = owner.input();
    let right = owner.input();
    let plan = reduction(
        fixture::plan(
            vec![
                builtin("power", vec![0, 1], 10),
                builtin("sum", vec![10, 2], 11),
            ],
            11,
        ),
        vec![0, 1, 2],
        10,
    );
    let ledger = runmat_runtime::context::host_effect::ManagedHostEffectLedger::new(0);
    let runtime = invocation::runtime();
    let ports = runtime
        .service_ports()
        .clone()
        .with_parallel(Rc::new(ManagedControls(ledger.clone())));
    let mut request = request(
        &plan,
        vec![
            Value::GpuTensor(left.handle.clone()),
            Value::GpuTensor(right.handle.clone()),
            Value::Num(0.5),
        ],
    );
    request.runtime = Some(runtime.with_service_ports(ports));
    let failure = match block_on(resolve_reduction_program(&request, ProviderPrecision::F64)) {
        Err(failure) => failure,
        _ => panic!("an unfunded parser cannot run or create a recipe"),
    };
    assert!(!failure.is_after_effect());
    assert_eq!(owner.calls.load(Ordering::SeqCst), 0);
    assert_eq!(ledger.snapshot().peak_reserved_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(left.probe.retirements(), 0);
    assert_eq!(right.probe.retirements(), 0);
}

struct DenyBindings {
    calls: Cell<usize>,
}
impl RuntimeBuiltinService for DenyBindings {
    fn bindings_by_name(&self, _: &str) -> Vec<runmat_runtime::builtin::RuntimeBuiltinBinding> {
        self.calls.set(self.calls.get() + 1);
        Vec::new()
    }
}
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn admitted_bindings_and_refined_domain_survive_suspension_but_do_not_authorize_next_request() {
    let owner = new_owner();
    let _provider = ThreadProviderGuard::set(Some(owner));
    owner.reset();
    owner.ready.store(false, Ordering::SeqCst);
    let left = owner.input();
    let right = owner.input();
    let plan = reduction(
        fixture::plan(
            vec![
                builtin("power", vec![0, 1], 10),
                fixture::cast("sin", 10, 12),
                builtin("sum", vec![12, 2], 11),
            ],
            11,
        ),
        vec![0, 1, 2],
        12,
    );
    let mut request = request(
        &plan,
        vec![
            Value::GpuTensor(left.handle.clone()),
            Value::GpuTensor(right.handle.clone()),
            Value::Num(2.0),
        ],
    );
    request.runtime = Some(RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    )));
    let mut future = Box::pin(resolve_reduction_program(&request, ProviderPrecision::F64));
    let mut task = Context::from_waker(futures::task::noop_waker_ref());
    assert!(future.as_mut().poll(&mut task).is_pending());
    assert_eq!(owner.calls.load(Ordering::SeqCst), 1);
    let denied = Rc::new(DenyBindings {
        calls: Cell::new(0),
    });
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_builtin(denied.clone()));
    let _context = context.enter();
    owner.ready.store(true, Ordering::SeqCst);
    let receipt = match future.as_mut().poll(&mut task) {
        Poll::Ready(Ok(Some(receipt))) => receipt,
        _ => panic!("captured owner declarations must finish after the real-domain probe"),
    };
    assert_eq!(receipt.contract().output_class, NumericClass::Double);
    assert_eq!(receipt.contract().axes, vec![1]);
    assert!(receipt.producer_body().contains("sin("));
    assert!(receipt.producer_preamble().contains("rm_power"));
    assert_eq!(receipt.data_inputs().len(), 2);
    assert_eq!(
        denied.calls.get(),
        0,
        "resuming does not select a new implementation"
    );
    let mut next_request = support::request(&plan, request.inputs.clone());
    next_request.runtime = Some(context.clone());
    assert!(block_on(resolve_reduction_program(
        &next_request,
        ProviderPrecision::F64
    ))
    .unwrap()
    .is_none());
    assert_eq!(owner.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        denied.calls.get(),
        1,
        "fresh request checks its current selected service"
    );
    drop(next_request);
    drop(receipt);
    drop(future);
    drop(request);
    drop(left.handle);
    drop(right.handle);
    assert_eq!(left.probe.retirements(), 1);
    assert_eq!(right.probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn probe_operational_errors_keep_the_original_source() {
    let owner = new_owner();
    let _provider = ThreadProviderGuard::set(Some(owner));
    owner.reset();
    owner.fail.store(true, Ordering::SeqCst);
    let left = owner.input();
    let right = owner.input();
    let plan = reduction(
        fixture::plan(
            vec![
                builtin("power", vec![0, 1], 10),
                builtin("sum", vec![10], 11),
            ],
            11,
        ),
        vec![0, 1],
        10,
    );
    let request = request(
        &plan,
        vec![
            Value::GpuTensor(left.handle),
            Value::GpuTensor(right.handle),
        ],
    );
    let error = match block_on(resolve_reduction_program(&request, ProviderPrecision::F64)) {
        Err(error) => error,
        _ => panic!("probe failed"),
    };
    let mut cause = error
        .source()
        .expect("producer probe error retains a source");
    while !cause.is::<ProbeFailure>() {
        cause = cause
            .source()
            .expect("original ProbeFailure remains in the source chain");
    }
    assert_eq!(owner.calls.load(Ordering::SeqCst), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn cancelled_admission_does_not_retire_borrowed_inputs_or_publish_a_receipt() {
    let owner = new_owner();
    let _provider = ThreadProviderGuard::set(Some(owner));
    owner.reset();
    owner.ready.store(false, Ordering::SeqCst);
    let left = owner.input();
    let right = owner.input();
    let plan = reduction(
        fixture::plan(
            vec![
                builtin("power", vec![0, 1], 10),
                builtin("sum", vec![10], 11),
            ],
            11,
        ),
        vec![0, 1],
        10,
    );
    let request = request(
        &plan,
        vec![
            Value::GpuTensor(left.handle),
            Value::GpuTensor(right.handle),
        ],
    );
    let mut future = Box::pin(resolve_reduction_program(&request, ProviderPrecision::F64));
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(futures::task::noop_waker_ref()))
        .is_pending());
    assert_eq!(owner.calls.load(Ordering::SeqCst), 1);
    drop(future);
    assert_eq!(left.probe.retirements(), 0);
    assert_eq!(right.probe.retirements(), 0);
    drop(request);
    assert_eq!(left.probe.retirements(), 1);
    assert_eq!(right.probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn original_slots_constants_and_host_controls_have_disjoint_recipe_roles() {
    let plan = reduction(
        fixture::plan(vec![add(0, 4, 10), builtin("sum", vec![10, 9], 11)], 11),
        vec![9, 7, 4, 0],
        10,
    );
    let request = request(
        &plan,
        vec![
            Value::Num(1.0),
            tensor(false, vec![1, 1]),
            tensor(false, vec![2, 3]),
            tensor(false, vec![2, 3]),
        ],
    );
    let receipt = block_on(resolve_reduction_program(&request, ProviderPrecision::F64))
        .unwrap()
        .unwrap();
    let mapping: Vec<_> = receipt
        .data_inputs()
        .iter()
        .map(|input| (input.value_id, input.request_slot))
        .collect();
    assert_eq!(mapping, vec![(4, Some(2)), (0, Some(3))]);
    assert!(std::ptr::eq(
        receipt.data_inputs()[0].value,
        &request.inputs[2]
    ));
    assert_eq!(receipt.contract().input_shape, vec![2, 3]);
    assert_eq!(receipt.contract().output_shape, vec![1, 3]);
    assert!(receipt.producer_body().contains("input1.data[i1]"));
    assert!(receipt.producer_body().contains("input0.data[i0]"));
    assert!(!receipt.producer_body().contains("input2"));
    let identity = receipt.identity();
    drop(receipt);
    let mut axis_two = request;
    axis_two.inputs[0] = Value::Num(2.0);
    let changed = block_on(resolve_reduction_program(&axis_two, ProviderPrecision::F64))
        .unwrap()
        .unwrap();
    assert_ne!(identity, changed.identity());
    assert_eq!(changed.contract().output_shape, vec![2, 1]);
    assert_eq!(plan.inputs, vec![9, 7, 4, 0]);
    assert_eq!(plan.operations.len(), 2);
    let mut constant_plan = plan.clone();
    constant_plan.inputs = vec![9, 0];
    constant_plan
        .const_values
        .insert(4, tensor(false, vec![2, 3]));
    let constant_request = support::request(
        &constant_plan,
        vec![Value::Num(1.0), tensor(false, vec![2, 3])],
    );
    let constant = block_on(resolve_reduction_program(
        &constant_request,
        ProviderPrecision::F64,
    ))
    .unwrap()
    .unwrap();
    assert_eq!(
        constant
            .data_inputs()
            .iter()
            .map(|input| (input.value_id, input.request_slot))
            .collect::<Vec<_>>(),
        vec![(0, Some(1)), (4, None)]
    );
    assert!(std::ptr::eq(
        constant.data_inputs()[1].value,
        &constant_plan.const_values[&4]
    ));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn strict_accumulator_and_mixed_class_proofs_are_not_weakened() {
    for single in [true, false] {
        let plan = reduction(
            fixture::plan(vec![add(0, 1, 10), builtin("sum", vec![10], 11)], 11),
            vec![0, 1],
            10,
        );
        let request = request(
            &plan,
            vec![tensor(single, vec![2, 2]), tensor(single, vec![2, 2])],
        );
        let receipt = block_on(resolve_reduction_program(&request, ProviderPrecision::F64))
            .unwrap()
            .unwrap();
        assert_eq!(receipt.contract().accumulator_class, NumericClass::Double);
        assert!(receipt.producer_body().contains(if single {
            "typed0: f32"
        } else {
            "typed0: f64"
        }));
        assert!(
            block_on(resolve_reduction_program(&request, ProviderPrecision::F32))
                .unwrap()
                .is_none()
        );
    }
    let plan = reduction(
        fixture::plan(vec![add(0, 1, 10), builtin("sum", vec![10], 11)], 11),
        vec![0, 1],
        10,
    );
    let request = request(
        &plan,
        vec![tensor(true, vec![2, 2]), tensor(false, vec![1, 1])],
    );
    let receipt = block_on(resolve_reduction_program(&request, ProviderPrecision::F64))
        .unwrap()
        .expect("Single array plus Double scalar retains canonical Single arithmetic");
    assert_eq!(receipt.contract().input_class, NumericClass::Single);
    assert_eq!(receipt.contract().accumulator_class, NumericClass::Double);
    assert_eq!(receipt.contract().output_class, NumericClass::Single);
    assert!(
        block_on(resolve_reduction_program(&request, ProviderPrecision::F32))
            .unwrap()
            .is_none(),
        "F32 owner cannot weaken default Double sum accumulation"
    );
    let min = reduction(
        fixture::plan(vec![add(0, 1, 10), builtin("min", vec![10], 11)], 11),
        vec![0, 1],
        10,
    );
    let request = support::request(
        &min,
        vec![tensor(true, vec![2, 2]), tensor(true, vec![2, 2])],
    );
    let receipt = block_on(resolve_reduction_program(&request, ProviderPrecision::F32))
        .unwrap()
        .unwrap();
    assert_eq!(receipt.contract().accumulator_class, NumericClass::Single);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn conflicting_missing_computed_and_extra_dependencies_fail_closed() {
    let base = reduction(
        fixture::plan(vec![add(0, 1, 10), builtin("sum", vec![10, 2], 11)], 11),
        vec![0, 1, 2],
        10,
    );
    // Cached values are not a second binding authority for an actual request
    // slot. Deliberately change both cached class and payload: neither may
    // replace the live Double operand or change its selected recipe identity.
    let values = vec![
        tensor(false, vec![2, 2]),
        tensor(false, vec![2, 2]),
        Value::Num(1.0),
    ];
    let original = request(&base, values.clone());
    let original_receipt = block_on(resolve_reduction_program(&original, ProviderPrecision::F64))
        .unwrap()
        .unwrap();
    let mut cached = base.clone();
    cached.const_values.insert(
        0,
        Value::Tensor(Tensor::from_f32(vec![99.0; 4], vec![2, 2]).unwrap()),
    );
    let cached_request = request(&cached, values);
    let cached_receipt = block_on(resolve_reduction_program(
        &cached_request,
        ProviderPrecision::F64,
    ))
    .unwrap()
    .expect("stale cached values cannot replace actual request inputs");
    assert!(std::ptr::eq(
        cached_receipt.data_inputs()[0].value,
        &cached_request.inputs[0],
    ));
    assert_eq!(cached_receipt.contract().input_class, NumericClass::Double);
    assert_eq!(cached_receipt.identity(), original_receipt.identity());

    for case in 1..6 {
        let mut plan = base.clone();
        match case {
            1 => {
                plan.inputs[1] = 0;
            }
            2 => {
                plan.operations[0] = add(0, 99, 10);
            }
            3 => {
                plan.operations.insert(1, add(0, 1, 20));
            }
            4 => {
                plan.operations.insert(1, add(0, 1, 20));
                *plan.operations.last_mut().unwrap() = builtin("sum", vec![10, 20], 11);
            }
            5 => {
                plan.reduction_data = Some(0);
            }
            _ => unreachable!(),
        }
        let request = request(
            &plan,
            vec![
                tensor(false, vec![2, 2]),
                tensor(false, vec![2, 2]),
                Value::Num(1.0),
            ],
        );
        assert!(
            block_on(resolve_reduction_program(&request, ProviderPrecision::F64))
                .unwrap()
                .is_none(),
            "case {case}"
        );
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn input_only_root_is_borrowed_and_buffer_contents_do_not_change_recipe_identity() {
    let plan = reduction(
        fixture::plan(vec![builtin("sum", vec![0], 11)], 11),
        vec![0],
        0,
    );
    let first = request(&plan, vec![tensor(false, vec![2, 2])]);
    let first_receipt = block_on(resolve_reduction_program(&first, ProviderPrecision::F64))
        .unwrap()
        .unwrap();
    assert!(first_receipt.producer_body().is_empty());
    assert_eq!(first_receipt.producer_expression(), "input0.data[i0]");
    assert!(std::ptr::eq(
        first_receipt.data_inputs()[0].value,
        &first.inputs[0]
    ));
    let second = request(
        &plan,
        vec![Value::Tensor(
            Tensor::new(vec![9.0; 4], vec![2, 2]).unwrap(),
        )],
    );
    let second_receipt = block_on(resolve_reduction_program(&second, ProviderPrecision::F64))
        .unwrap()
        .unwrap();
    assert_eq!(first_receipt.identity(), second_receipt.identity());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn captured_bindings_preserve_real_catalog_spelling_resolution() {
    fixture::builder::initialize_registry();
    use runmat_runtime::builtins::common::spec::{selected_fusion_binding, SelectedFusionIdentity};
    let canonical = runmat_builtins::builtin_catalog_entry_by_name("sin").unwrap();
    let mut spellings = vec!["sin", "SiN"];
    // Exercise actual aliases if this catalog declares any. Never invent an
    // alias registry merely to make a positive fixture, or grant an alias that
    // the selected implementation registry does not admit.
    spellings.extend(
        runmat_builtins::builtin_catalog_aliases()
            .iter()
            .filter(|alias| alias.canonical == canonical.identity)
            .map(|alias| alias.alias.name),
    );
    let mut identity = None;
    for name in spellings {
        let entry = runmat_builtins::builtin_catalog_entry_by_name(name).unwrap();
        assert_eq!(entry.identity, canonical.identity);
        let selected = selected_fusion_binding(name, None, None);
        if name == "sin" || name == "SiN" {
            assert!(selected.is_some(), "compiled selected binding for {name}");
        }
        let plan = reduction(
            fixture::plan(
                vec![builtin(name, vec![0], 10), builtin("sum", vec![10], 11)],
                11,
            ),
            vec![0],
            10,
        );
        let request = request(&plan, vec![tensor(false, vec![2, 2])]);
        let receipt =
            block_on(resolve_reduction_program(&request, ProviderPrecision::F64)).unwrap();
        let Some(selected) = selected else {
            assert!(
                receipt.is_none(),
                "catalog aliases cannot bypass selected dispatch"
            );
            continue;
        };
        let SelectedFusionIdentity::Canonical(binding) = selected.identity else {
            panic!("canonical sin binding");
        };
        assert_eq!(binding.builtin, canonical.identity);
        let receipt = receipt.expect("selected spelling uses the captured canonical emitter");
        assert!(receipt.producer_body().contains("sin("));
        if let Some(expected) = identity {
            assert_eq!(receipt.identity(), expected);
        } else {
            identity = Some(receipt.identity());
        }
    }
}
