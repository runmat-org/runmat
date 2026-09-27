//! Production bytecode dispatch ownership, separate from fusion admission.
use super::*;
use std::collections::HashSet;

pub(super) fn compile_borrowed_x(source: &str) -> runmat_vm::Bytecode {
    let variables = HashMap::from([("x".to_string(), 0)]);
    let ast = runmat_parser::parse(source).unwrap();
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::new(&variables)).unwrap();
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
    runmat_vm::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id).unwrap()
}

struct RejectFusion(&'static TestProvider);
impl Drop for RejectFusion {
    fn drop(&mut self) {
        self.0.reject_fusion.store(false, Ordering::Relaxed);
    }
}

struct InspectRootsOnFree(&'static TestProvider);
impl Drop for InspectRootsOnFree {
    fn drop(&mut self) {
        self.0.inspect_roots_on_free.store(false, Ordering::Relaxed);
    }
}

struct FailFreeBuffer(&'static TestProvider);
impl Drop for FailFreeBuffer {
    fn drop(&mut self) {
        self.0.fail_free_buffer.store(0, Ordering::Relaxed);
    }
}

#[test]
fn failed_retirement_preserves_residency_marker_and_live_allocation() {
    gc_test_context(|| {
        let _lock = accel_test_lock();
        ensure_provider_registered();
        let provider = PROVIDER.get().unwrap();
        let owned = provider.push(vec![11.0], vec![1, 1]);
        runmat_accelerate::fusion_residency::mark(&owned);
        provider
            .fail_free_buffer
            .store(owned.buffer_id, Ordering::Relaxed);
        let failure = FailFreeBuffer(provider);
        let mut vars = vec![Value::GpuTensor(owned.clone())];
        let mut context = Default::default();
        runmat_vm::accel::fusion::write_elementwise_materialized_stores(
            vec![(
                runmat_accelerate::fusion::FusionStoreMaterialization {
                    value_id: 0,
                    binding: runmat_accelerate::graph::VarBinding {
                        kind: runmat_accelerate::VarKind::Global,
                        index: 0,
                    },
                },
                Value::Num(0.0),
            )],
            &mut vars,
            &mut context,
        )
        .finish(&[]);
        assert!(runmat_accelerate::fusion_residency::is_resident(&owned));
        assert_eq!(provider.pull(&owned).unwrap(), (vec![11.0], vec![1, 1]));
        drop(failure);
        provider.free(&owned).unwrap();
        runmat_accelerate::fusion_residency::clear(&owned);
    });
}

fn resident_outputs(values: &[Value]) -> Vec<GpuTensorHandle> {
    let mut seen = HashSet::new();
    let mut handles = Vec::new();
    for value in values {
        runmat_runtime::dispatcher::try_visit_gpu_handles(value, &mut |handle| {
            if seen.insert(runmat_accelerate_api::handle_identity(handle)) {
                handles.push(handle.clone());
            }
        })
        .unwrap();
    }
    handles
}

#[test]
fn borrowed_initial_workspace_survives_error_and_retires_only_after_replacement() {
    gc_test_context(|| {
        let _lock = accel_test_lock();
        ensure_provider_registered();
        let provider = PROVIDER.get().unwrap();
        for (source, fails, preserves) in [
            ("x = 3; error('ownership witness');", true, true),
            ("alias = x; x = 3;", false, true),
            ("x = 3;", false, false),
        ] {
            let bytecode = compile_borrowed_x(source);
            let slot = *bytecode
                .var_names
                .iter()
                .find(|(_, name)| name.as_str() == "x")
                .unwrap()
                .0;
            let input = provider.push(vec![7.0], vec![1, 1]);
            let mut initial = vec![Value::Num(0.0); bytecode.var_count];
            initial[slot] = Value::GpuTensor(input.clone());
            let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ));
            let result = block_on(runmat_vm::interpret_with_vars_in_context(
                &bytecode,
                &mut initial,
                None,
                runtime,
            ));
            assert_eq!(result.is_err(), fails, "source={source}; result={result:?}");
            if preserves {
                assert_eq!(provider.pull(&input).unwrap(), (vec![7.0], vec![1, 1]));
                assert_eq!(provider.frees.lock().unwrap().get(&input.buffer_id), None);
                let handles = resident_outputs(&initial);
                assert!(handles
                    .iter()
                    .any(|handle| handle.buffer_id == input.buffer_id));
                provider.free(&input).unwrap();
            } else {
                assert!(!provider
                    .buffers
                    .lock()
                    .unwrap()
                    .contains_key(&input.buffer_id));
                assert_eq!(
                    provider.frees.lock().unwrap().get(&input.buffer_id),
                    Some(&1)
                );
            }
        }
    });
}

#[test]
fn completed_builtin_calls_release_consumed_intermediates_and_adopt_aliases() {
    gc_test_context(|| {
        let _lock = accel_test_lock();
        ensure_provider_registered();
        let provider = PROVIDER.get().unwrap();
        let baseline = provider
            .buffers
            .lock()
            .unwrap()
            .keys()
            .copied()
            .collect::<HashSet<_>>();
        provider.reject_fusion.store(true, Ordering::Relaxed);
        let _reject = RejectFusion(provider);
        let dispatches = provider.fused_dispatches.load(Ordering::Relaxed);
        let source = "x = [0.25 0.5]; y = single(sin(x)); alias = y;";
        let bytecode = compile_semantic(source);
        let input = provider.push(vec![0.25, 0.5], vec![1, 2]);
        let first_temporary = provider.next_id.load(Ordering::Relaxed);
        let resume =
            observation::resume_after_input_store(&bytecode, Value::GpuTensor(input.clone()));
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let runmat_vm::InterpreterOutcome::Completed(completed) = block_on(
            runmat_vm::interpret_resume_in_context(&bytecode, resume, None, runtime),
        )
        .unwrap();
        assert_eq!(
            provider.fused_dispatches.load(Ordering::Relaxed),
            dispatches,
            "this witness must exercise ordinary call dispatch"
        );
        let slot = |name: &str| {
            *bytecode
                .var_names
                .iter()
                .find(|(_, value)| value.as_str() == name)
                .unwrap()
                .0
        };
        let Value::GpuTensor(y) = &completed.values[slot("y")] else {
            panic!("resident Single output")
        };
        let Value::GpuTensor(alias) = &completed.values[slot("alias")] else {
            panic!("resident output alias")
        };
        assert_eq!(
            runmat_accelerate_api::handle_identity(y),
            runmat_accelerate_api::handle_identity(alias)
        );
        let result = block_on(provider.download_numeric(y)).unwrap();
        assert_eq!(result.shape, vec![1, 2]);
        let HostNumericDataOwned::F32(values) = result.data else {
            panic!("physical Single output")
        };
        assert_eq!(values, vec![0.25f64.sin() as f32, 0.5f64.sin() as f32]);
        assert_eq!(
            provider.pull(&input).unwrap(),
            (vec![0.25, 0.5], vec![1, 2])
        );
        let outputs = resident_outputs(&completed.values);
        let mut expected = baseline.clone();
        expected.extend(outputs.iter().map(|handle| handle.buffer_id));
        let live = provider
            .buffers
            .lock()
            .unwrap()
            .keys()
            .copied()
            .collect::<HashSet<_>>();
        assert_eq!(
            live, expected,
            "only adopted outputs and prior buffers may remain"
        );
        let end_temporary = provider.next_id.load(Ordering::Relaxed);
        let retired = (first_temporary..end_temporary)
            .filter(|id| !expected.contains(id))
            .collect::<Vec<_>>();
        assert!(
            !retired.is_empty(),
            "witness must allocate and consume an intermediate"
        );
        for id in retired {
            assert_eq!(
                provider.frees.lock().unwrap().get(&id).copied(),
                Some(1),
                "temporary {id} must be released exactly once"
            );
        }
        for output in outputs {
            provider.free(&output).unwrap();
        }
        assert_eq!(
            provider
                .buffers
                .lock()
                .unwrap()
                .keys()
                .copied()
                .collect::<HashSet<_>>(),
            baseline
        );
    });
}

#[test]
fn overwritten_fusion_stores_preserve_older_nested_alias_and_retire_once() {
    gc_test_context(|| {
        let _lock = accel_test_lock();
        ensure_provider_registered();
        let provider = PROVIDER.get().unwrap();
        let borrowed = provider.push(vec![7.0], vec![1, 1]);
        let caller = runmat_gc::gc_register_value_root(Box::new(runmat_gc::GlobalRoot::new(
            vec![Value::Cell(
                runmat_value::CellArray::new_with_shape(
                    vec![Value::GpuTensor(borrowed.clone())],
                    vec![1, 1],
                )
                .unwrap(),
            )],
            "older_caller_nested_alias".into(),
        )))
        .unwrap();
        let store = |index| runmat_accelerate::fusion::FusionStoreMaterialization {
            value_id: index as u32,
            binding: runmat_accelerate::graph::VarBinding {
                kind: runmat_accelerate::VarKind::Global,
                index,
            },
        };
        let mut context = Default::default();
        let mut vars = vec![
            Value::GpuTensor(borrowed.clone()),
            Value::GpuTensor(borrowed.clone()),
        ];
        runmat_vm::accel::fusion::write_elementwise_materialized_stores(
            vec![(store(0), Value::Num(0.0)), (store(1), Value::Num(0.0))],
            &mut vars,
            &mut context,
        )
        .finish(&[]);
        assert_eq!(provider.pull(&borrowed).unwrap(), (vec![7.0], vec![1, 1]));
        assert_eq!(
            provider.frees.lock().unwrap().get(&borrowed.buffer_id),
            None
        );
        runmat_gc::gc_unregister_root(caller).unwrap();
        provider.free(&borrowed).unwrap();

        let owned = provider.push(vec![9.0], vec![1, 1]);
        provider
            .inspect_roots_on_free
            .store(true, Ordering::Relaxed);
        let _inspect = InspectRootsOnFree(provider);
        vars = vec![
            Value::GpuTensor(owned.clone()),
            Value::GpuTensor(owned.clone()),
        ];
        runmat_vm::accel::fusion::write_elementwise_materialized_stores(
            vec![(store(0), Value::Num(0.0)), (store(1), Value::Num(0.0))],
            &mut vars,
            &mut context,
        )
        .finish(&[]);
        assert!(!provider
            .buffers
            .lock()
            .unwrap()
            .contains_key(&owned.buffer_id));
        assert_eq!(
            provider
                .frees
                .lock()
                .unwrap()
                .get(&owned.buffer_id)
                .copied(),
            Some(1)
        );
    });
}
