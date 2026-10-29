//! Compiled VM witnesses for terminal provider and ordinary post-fusion store errors.
use super::*;
use runmat_runtime::context::RuntimeWorkspaceService;
use runmat_runtime::runtime_error::build_runtime_error;
use std::cell::RefCell;
use std::collections::HashSet;

const TERMINAL_SCRIPT: &str = r#"
    x = 1:4096;
    try
        mu = sin(x) .* x;
        sigma = mu + 2;
        out = sigma .* 3;
    catch caught
        observed_identifier = caught.identifier;
        observed_message = caught.message;
    end
"#;

fn slot(bytecode: &runmat_vm::Bytecode, name: &str) -> usize {
    bytecode
        .var_names
        .iter()
        .find_map(|(index, candidate)| (candidate == name).then_some(*index))
        .expect("compiled variable")
}

fn compiled_terminal_fixture() -> runmat_vm::Bytecode {
    let bytecode = compile_semantic(TERMINAL_SCRIPT);
    let groups = bytecode.runtime_fusion_groups();
    assert!(
        groups.iter().any(|group| {
            matches!(group.kind, FusionKind::ElementwiseChain)
                && bytecode.instructions[group.span.start..=group.span.end]
                    .iter()
                    .all(|instruction| {
                        !matches!(instruction, Instr::StoreVar(_) | Instr::StoreLocal(_))
                    })
        }),
        "compiled source must offer an operator-only elementwise candidate"
    );
    bytecode
}

fn top_level_admission(bytecode: &runmat_vm::Bytecode) -> runmat_vm::PendingWorkspaceAdmission {
    let names = bytecode
        .var_names
        .iter()
        .map(|(slot, name)| (name.clone(), *slot))
        .collect();
    runmat_vm::workspace_admission(names, HashSet::new())
}

#[derive(Default)]
struct RejectSecondStore {
    values: RefCell<HashMap<String, Value>>,
    attempts: RefCell<Vec<String>>,
    dispatches_at_rejection: RefCell<Option<u64>>,
}

impl RuntimeWorkspaceService for RejectSecondStore {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        for value in self.values.borrow().values() {
            visitor(value)?;
        }
        Some(())
    }

    fn lookup(&self, name: &str) -> Option<Value> {
        self.values.borrow().get(name).cloned()
    }

    fn snapshot(&self) -> Vec<(String, Value)> {
        self.values
            .borrow()
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }

    fn global_names(&self) -> Vec<String> {
        Vec::new()
    }

    fn assign(&self, name: &str, value: Value) -> Result<(), RuntimeError> {
        self.attempts.borrow_mut().push(name.to_string());
        if name == "sigma" {
            *self.dispatches_at_rejection.borrow_mut() = Some(
                registered_provider()
                    .expect("selected fixture provider")
                    .fused_dispatches
                    .load(Ordering::Relaxed),
            );
            return Err(build_runtime_error("second workspace publication rejected")
                .with_identifier("RunMat:FusionFixture:SecondStore")
                .build());
        }
        self.values.borrow_mut().insert(name.to_string(), value);
        Ok(())
    }

    fn clear(&self) -> Result<(), RuntimeError> {
        self.values.borrow_mut().clear();
        Ok(())
    }

    fn remove(&self, name: &str) -> Result<(), RuntimeError> {
        self.values.borrow_mut().remove(name);
        Ok(())
    }
}

struct ResetSingleFailure(&'static TestProvider);

impl Drop for ResetSingleFailure {
    fn drop(&mut self) {
        self.0
            .fail_next_fused_single
            .store(false, Ordering::Relaxed);
    }
}

#[test]
fn compiled_fused_compute_then_ordinary_second_store_failure_is_caught_without_replay() {
    gc_test_context(|| {
        let _guard = accel_test_lock();
        ensure_provider_registered();
        runmat_accelerate::placement::reset();
        let provider = registered_provider().expect("test provider");
        let bytecode = compiled_terminal_fixture();

        let workspace = Rc::new(RejectSecondStore::default());
        let placement = Rc::new(runmat_accelerate::placement::PlacementSession::default());
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default()
                .with_workspace(workspace.clone())
                .with_placement(placement),
        );
        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let before = provider.fused_dispatches.load(Ordering::Relaxed);
        let runmat_vm::InterpreterOutcome::Completed(completion) =
            block_on(runmat_vm::interpret_workspace(
                &bytecode,
                runmat_vm::InterpreterVars::from_owned(vars),
                None,
                runtime,
                Some(top_level_admission(&bytecode)),
                None,
            ))
            .outcome
            .expect("language catch handles terminal publication failure");
        assert_eq!(
            completion
                .values
                .get(slot(&bytecode, "observed_identifier"))
                .as_deref(),
            Some(&Value::CharArray(runmat_value::CharArray::new_row(
                "RunMat:FusionFixture:SecondStore"
            )))
        );
        assert_eq!(
            completion
                .values
                .get(slot(&bytecode, "observed_message"))
                .as_deref(),
            Some(&Value::CharArray(runmat_value::CharArray::new_row(
                "second workspace publication rejected"
            )))
        );
        assert!(matches!(
            completion.values.get(slot(&bytecode, "mu")).as_deref(),
            Some(Value::GpuTensor(_))
        ));
        assert_eq!(
            completion.values.get(slot(&bytecode, "sigma")).as_deref(),
            Some(&Value::Num(0.0))
        );
        assert_eq!(
            completion.values.get(slot(&bytecode, "out")).as_deref(),
            Some(&Value::Num(0.0))
        );
        let after = provider.fused_dispatches.load(Ordering::Relaxed);
        assert!(
            after > before,
            "compiled operator work must reach the provider"
        );
        assert_eq!(
            *workspace.dispatches_at_rejection.borrow(),
            Some(after),
            "the caught ordinary store error must not replay provider work"
        );
        let publication_attempts = workspace
            .attempts
            .borrow()
            .iter()
            .filter(|name| matches!(name.as_str(), "mu" | "sigma" | "out"))
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            publication_attempts,
            vec!["mu".to_string(), "sigma".to_string()],
            "ordinary stores publish in source order exactly once before catch"
        );
    });
}

#[test]
fn compiled_fusion_hard_provider_failure_is_caught_without_cpu_replay() {
    gc_test_context(|| {
        let _guard = accel_test_lock();
        ensure_provider_registered();
        runmat_accelerate::placement::reset();
        let provider = registered_provider().expect("test provider").as_ref();
        provider
            .fail_next_fused_single
            .store(true, Ordering::Relaxed);
        let _reset = ResetSingleFailure(provider);
        let bytecode = compiled_terminal_fixture();
        let workspace = Rc::new(RejectSecondStore::default());
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default()
                .with_workspace(workspace.clone())
                .with_placement(Rc::new(
                    runmat_accelerate::placement::PlacementSession::default(),
                )),
        );
        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let before = provider.fused_dispatches.load(Ordering::Relaxed);
        let runmat_vm::InterpreterOutcome::Completed(completion) =
            block_on(runmat_vm::interpret_workspace(
                &bytecode,
                runmat_vm::InterpreterVars::from_owned(vars),
                None,
                runtime,
                Some(top_level_admission(&bytecode)),
                None,
            ))
            .outcome
            .expect("language catch handles exact hard provider error");
        assert_eq!(
            completion
                .values
                .get(slot(&bytecode, "observed_identifier"))
                .as_deref(),
            Some(&Value::CharArray(runmat_value::CharArray::new_row(
                "RunMat:FusionFixture:ProviderFailure"
            )))
        );
        assert_eq!(
            completion
                .values
                .get(slot(&bytecode, "observed_message"))
                .as_deref(),
            Some(&Value::CharArray(runmat_value::CharArray::new_row(
                "selected provider rejected single-output execution"
            )))
        );
        assert_eq!(
            completion.values.get(slot(&bytecode, "mu")).as_deref(),
            Some(&Value::Num(0.0))
        );
        assert_eq!(
            completion.values.get(slot(&bytecode, "sigma")).as_deref(),
            Some(&Value::Num(0.0))
        );
        assert_eq!(
            completion.values.get(slot(&bytecode, "out")).as_deref(),
            Some(&Value::Num(0.0))
        );
        assert_eq!(
            provider.fused_dispatches.load(Ordering::Relaxed) - before,
            1
        );
        assert!(workspace
            .attempts
            .borrow()
            .iter()
            .all(|name| name != "mu" && name != "sigma" && name != "out"));
    });
}
