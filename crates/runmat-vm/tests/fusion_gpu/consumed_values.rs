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

struct InspectRootsOnFree(&'static TestProvider);

pub(super) struct CanonicalInput {
    pub owner: runmat_runtime::workspace::BaseWorkspaceOwner,
    pub projection: Rc<runmat_runtime::workspace::BaseWorkspaceProjectionService>,
    pub runtime: runmat_runtime::context::RuntimeContext,
    pub vars: runmat_vm::InterpreterVars,
    pub admission: runmat_vm::PendingWorkspaceAdmission,
    pub source: runmat_runtime::workspace::WorkspaceValueReadLease,
}

pub(super) fn canonical_input(
    bytecode: &runmat_vm::Bytecode,
    runtime: runmat_runtime::context::RuntimeContext,
    input: &GpuTensorHandle,
) -> CanonicalInput {
    use runmat_runtime::context::RuntimeWorkspaceService;
    use runmat_runtime::workspace::{
        BaseWorkspaceOwner, BaseWorkspaceProjectionService, BaseWorkspaceService,
        LogicalWorkspaceGeneration,
    };
    let owner = BaseWorkspaceOwner::new();
    let base = Rc::new(BaseWorkspaceService::new(owner.clone()));
    base.assign("x", Value::GpuTensor(input.clone())).unwrap();
    let projection = Rc::new(BaseWorkspaceProjectionService::new_for_context(
        owner.begin_projection().unwrap(),
        &runtime,
    ));
    let ports = runtime
        .service_ports()
        .clone()
        .with_workspace(projection.clone())
        .with_base_workspace(base);
    let runtime = runtime.with_service_ports(ports);
    let generation = LogicalWorkspaceGeneration::capture(
        &runtime,
        &owner,
        &runtime.session_variable_authority(),
    )
    .unwrap();
    let source = generation
        .read_lease("x")
        .expect("canonical source read lease");
    let vars = runmat_vm::InterpreterVars::from_workspace_generation(
        &runtime,
        &generation,
        bytecode
            .var_names
            .iter()
            .map(|(slot, name)| (name.as_str(), *slot)),
        bytecode.var_count,
    )
    .unwrap();
    drop(generation);
    let admission = runmat_vm::workspace_admission(
        bytecode
            .var_names
            .iter()
            .map(|(slot, name)| (name.clone(), *slot))
            .collect(),
        HashSet::from(["x".to_string()]),
    );
    CanonicalInput {
        owner,
        projection,
        runtime,
        vars,
        admission,
        source,
    }
}
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
fn failed_retirement_retains_payload_and_failure_record_without_reviving_view() {
    gc_test_context(|| {
        let _lock = accel_test_lock();
        ensure_provider_registered();
        let provider = registered_provider().unwrap();
        let owned = provider.push(vec![11.0], vec![1, 1]);
        runmat_accelerate::fusion_residency::mark(&owned);
        provider
            .fail_free_buffer
            .store(owned.buffer_id(), Ordering::Relaxed);
        let failure = FailFreeBuffer(provider);
        let mut vars =
            runmat_vm::InterpreterVars::from_owned(vec![Value::GpuTensor(owned.clone())]);
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
        .expect("materialized stores must be adopted");
        assert!(runmat_accelerate::fusion_residency::is_resident(&owned));
        assert_eq!(provider.pull(&owned).unwrap(), (vec![11.0], vec![1, 1]));
        drop(vars);
        drop(context);
        let id = owned.buffer_id();
        let weak = owned.downgrade();
        let error = owned
            .release()
            .expect_err("final release must surface injected failure");
        assert!(error.to_string().contains("injected release failure"));
        assert!(provider.buffers.lock().unwrap().contains_key(&id));
        assert_eq!(provider.frees.lock().unwrap().get(&id), None);
        assert!(
            weak.acquire().is_err(),
            "failed retirement cannot revive a view"
        );
        drop(failure);
    });
}

pub(super) fn resident_outputs_frame(values: &runmat_vm::InterpreterVars) -> Vec<GpuTensorHandle> {
    let mut seen = HashSet::new();
    let mut handles = Vec::new();
    for slot in 0..values.len() {
        let value = values.get(slot).expect("slot in completed frame extent");
        record_resident_outputs(&value, &mut seen, &mut handles);
    }
    handles
}

fn record_resident_outputs(
    value: &Value,
    seen: &mut HashSet<runmat_accelerate_api::GpuHandleIdentity>,
    handles: &mut Vec<GpuTensorHandle>,
) {
    runmat_runtime::dispatcher::try_visit_gpu_handles(value, &mut |handle| {
        if seen.insert(runmat_accelerate_api::handle_identity(handle)) {
            handles.push(handle.clone());
        }
    })
    .unwrap();
}

#[test]
fn borrowed_initial_workspace_survives_error_and_retires_only_after_replacement() {
    gc_test_context(|| {
        let _lock = accel_test_lock();
        ensure_provider_registered();
        let provider = registered_provider().unwrap();
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
            let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ));
            let CanonicalInput {
                owner,
                projection,
                runtime,
                vars,
                admission,
                source: source_lease,
            } = canonical_input(&bytecode, runtime, &input);
            let mut publication =
                block_on(runtime.scope(async { projection.terminal_publication(&runtime) }))
                    .unwrap();
            let result = block_on(runmat_vm::interpret_workspace(
                &bytecode,
                vars,
                None,
                runtime.clone(),
                Some(admission),
                None,
            ));
            assert_eq!(
                result.outcome.is_err(),
                fails,
                "source={source}; result={:?}",
                result.outcome
            );
            assert_eq!(
                block_on(runtime.scope(async { projection.lookup("x") })).unwrap(),
                Some(Value::Num(3.0)),
                "staged prefix survives terminalization"
            );
            block_on(runtime.scope(async { publication.publish_pending() })).unwrap();
            assert_eq!(
                owner.lookup("x").unwrap(),
                Some(Value::Num(3.0)),
                "canonical terminal prefix is published even on error"
            );
            if fails {
                let closed = result
                    .workspace
                    .as_ref()
                    .expect("exact failed frame completion");
                assert_eq!(
                    closed.state.vars.get(slot).as_deref(),
                    Some(&Value::Num(3.0))
                );
            }
            if preserves {
                assert_eq!(provider.pull(&input).unwrap(), (vec![7.0], vec![1, 1]));
                assert_eq!(provider.frees.lock().unwrap().get(&input.buffer_id()), None);
                let retained = source_lease.value();
                assert!(
                    matches!(retained, Value::GpuTensor(handle) if handle.buffer_id() == input.buffer_id())
                );
                if !fails {
                    assert!(
                        matches!(owner.lookup("alias").unwrap(), Some(Value::GpuTensor(handle)) if handle.buffer_id() == input.buffer_id())
                    );
                }
            }
            let id = input.buffer_id();
            drop(result);
            drop(publication);
            drop(source_lease);
            drop(projection);
            drop(runtime);
            drop(owner);
            input.release().expect("last caller view retires input");
            assert!(!provider.buffers.lock().unwrap().contains_key(&id));
            assert_eq!(provider.frees.lock().unwrap().get(&id), Some(&1));
        }
    });
}

#[test]
fn completed_builtin_calls_release_consumed_intermediates_and_adopt_aliases() {
    gc_test_context(|| {
        let _lock = accel_test_lock();
        ensure_provider_registered();
        let provider = registered_provider().unwrap();
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
        let single_calls = provider.single_inputs.lock().unwrap().len();
        // Authored y/alias outputs must survive, but the exact compiler proof
        // may move a uniquely consumed private sine temporary into single.
        // Both real iterations must retire that unobservable intermediate.
        let source = "x = [0.25 0.5]; for k = 1:2; y = single(sin(x)); end; alias = y;";
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
        {
            let y = completed
                .values
                .get(slot("y"))
                .expect("resident Single output");
            let Value::GpuTensor(y) = &*y else {
                panic!("resident Single output")
            };
            let alias = completed
                .values
                .get(slot("alias"))
                .expect("resident output alias");
            let Value::GpuTensor(alias) = &*alias else {
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
        }
        assert_eq!(
            provider.pull(&input).unwrap(),
            (vec![0.25, 0.5], vec![1, 2])
        );
        let outputs = resident_outputs_frame(&completed.values);
        let mut expected = baseline.clone();
        expected.extend(outputs.iter().map(|handle| handle.buffer_id()));
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
        let single_inputs = provider.single_inputs.lock().unwrap()[single_calls..].to_vec();
        assert_eq!(
            single_inputs.len(),
            2,
            "both real call iterations must execute the resident cast hook"
        );
        assert_ne!(single_inputs[0], single_inputs[1]);
        assert!(
            retired.contains(&single_inputs[0]),
            "the first sine result consumed by single must be retired"
        );
        assert!(
            retired.contains(&single_inputs[1]),
            "the final compiler-private sine input must retire after its consuming cast"
        );
        for id in retired {
            assert_eq!(
                provider.frees.lock().unwrap().get(&id).copied(),
                Some(1),
                "temporary {id} must be released exactly once"
            );
        }
        drop(completed);
        drop(outputs);
        drop(input);
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
        let provider = registered_provider().unwrap();
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
        let mut vars = runmat_vm::InterpreterVars::from_owned(vec![
            Value::GpuTensor(borrowed.clone()),
            Value::GpuTensor(borrowed.clone()),
        ]);
        runmat_vm::accel::fusion::write_elementwise_materialized_stores(
            vec![(store(0), Value::Num(0.0)), (store(1), Value::Num(0.0))],
            &mut vars,
            &mut context,
        )
        .expect("overwritten stores must preserve nested caller aliases");
        assert_eq!(provider.pull(&borrowed).unwrap(), (vec![7.0], vec![1, 1]));
        assert_eq!(
            provider.frees.lock().unwrap().get(&borrowed.buffer_id()),
            None
        );
        runmat_gc::gc_unregister_root(caller).unwrap();
        borrowed
            .release()
            .expect("final caller view retires allocation");

        let owned = provider.push(vec![9.0], vec![1, 1]);
        provider
            .inspect_roots_on_free
            .store(true, Ordering::Relaxed);
        let _inspect = InspectRootsOnFree(provider);
        vars = runmat_vm::InterpreterVars::from_owned(vec![
            Value::GpuTensor(owned.clone()),
            Value::GpuTensor(owned.clone()),
        ]);
        runmat_vm::accel::fusion::write_elementwise_materialized_stores(
            vec![(store(0), Value::Num(0.0)), (store(1), Value::Num(0.0))],
            &mut vars,
            &mut context,
        )
        .expect("overwritten stores must preserve owned allocation leases");
        let owned_id = owned.buffer_id();
        assert!(provider.buffers.lock().unwrap().contains_key(&owned_id));
        owned.release().expect("last owned view retires allocation");
        assert!(!provider.buffers.lock().unwrap().contains_key(&owned_id));
        assert_eq!(
            provider.frees.lock().unwrap().get(&owned_id).copied(),
            Some(1)
        );
    });
}
