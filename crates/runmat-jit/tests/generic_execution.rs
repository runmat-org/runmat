#![cfg(not(target_arch = "wasm32"))]

use std::{
    collections::BTreeMap,
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use runmat_execution::{
    Digest, EstimateSource, ExecutableComponentDescriptor, ExecutableComponentKind,
    ExecutableComponentPayload, ExecutableComponentRevisions, ExecutableEntrypointKind,
    ExecutableIdentity, ExecutableUnitManifest, ExecutionCandidateKind, ProgramEnvironment,
    ProgramRevision, EXECUTABLE_UNIT_SCHEMA_VERSION,
};
use runmat_hir::{
    FunctionAbi, FunctionId, FunctionKind, FunctionModifiers, FunctionName, Span, WorkspaceEffect,
};
use runmat_jit::{
    deopt::{DeoptimizationPolicy, FaultInjection, ResumeTarget},
    entry::{EntryKey, EntryRegistry},
    invalidation::{DependencyKey, DependencyTracker},
    specialization::GuardEnvironment,
    GenericExecutor, JitError,
};
use runmat_mir::{
    AsyncBehaviorFact, BasicBlock, BasicBlockId, MirAggregateKind, MirAssembly, MirBody, MirCall,
    MirCallArg, MirCallee, MirConstant, MirFunctionMetadata, MirIndexComponent, MirIndexPlan,
    MirIndexing, MirLocal, MirLocalId, MirLocalKind, MirOperand, MirOutputTarget,
    MirOutputTargetList, MirPlace, MirPlaceMutation, MirRvalue, MirStmt, MirStmtKind,
    MirTerminator, MirTerminatorKind,
};
use runmat_native_codegen::{lower_executable, NativeLoweringInput, NativeTarget};
use runmat_runtime::{context::RuntimeContext, execution::RuntimeExecutionService};
use runmat_types::{
    AliasFact, BindingId, BuiltinId, CallableFallbackPolicy, CallableIdentity,
    CapabilityRequirement, CapabilitySet, DeoptimizationPointId, InteropManifest, NumericClass,
    NumericDomain, NumericFact, ParallelManifest, ProgramFunctionId, ProgramPointId,
    ProgramSourceId, RegionContract, RegionGuardCondition, RegionGuardContract, RegionGuardId,
    RegionId, RegionProvenance, RegionValueFact, RegionValueId, RequestedOutputCount,
    ResidencyFact, ShapeFact, ValueFact, ValueKindFact, INTEROP_MANIFEST_SCHEMA_VERSION,
    PARALLEL_MANIFEST_SCHEMA_VERSION, REGION_CONTRACT_SCHEMA_VERSION,
};
use runmat_value::Value;

#[test]
fn stable_entry_cell_retires_stale_target_and_publishes_replacement() {
    let mut dependencies = DependencyTracker::default();
    let program = DependencyKey::Program("project".into());
    let provider = DependencyKey::Provider(7);
    dependencies.observe(program.clone(), "revision-a").unwrap();
    dependencies
        .observe(provider.clone(), "provider-a")
        .unwrap();
    let first_snapshot = dependencies.snapshot([&program, &provider]);

    let key = EntryKey("project/main".into());
    let mut registry = EntryRegistry::default();
    let stable_cell = registry.cell(key.clone());
    let first_executor = Rc::new(GenericExecutor::compile(fixture()).unwrap());
    let first = registry
        .publish(
            key.clone(),
            Rc::clone(&first_executor),
            ProgramFunctionId(0),
            first_snapshot,
        )
        .unwrap();
    assert_eq!(first.publication, 1);
    assert!(Rc::ptr_eq(&first.executor, &first_executor));

    dependencies.observe(provider, "provider-b").unwrap();
    let current = dependencies.snapshot_all();
    assert_eq!(registry.invalidate(&current), 1);
    assert!(!stable_cell.is_published());
    assert_eq!(registry.retained_cell_count(), 1);

    let second_executor = Rc::new(GenericExecutor::compile(fixture()).unwrap());
    let second = registry
        .publish(
            key,
            Rc::clone(&second_executor),
            ProgramFunctionId(0),
            current,
        )
        .unwrap();
    assert_eq!(second.publication, 2);
    assert!(stable_cell.is_published());
    assert!(Rc::ptr_eq(&second.executor, &second_executor));
    assert!(!Rc::ptr_eq(&first.executor, &second.executor));
}

#[test]
fn forced_generic_entry_executes_literal_assignment_and_transactional_return() {
    let executor = GenericExecutor::compile(fixture()).unwrap();
    let execution = executor
        .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
        .unwrap();
    assert_eq!(execution.outputs, vec![Value::Num(41.0)]);
}

#[test]
fn generic_executor_describes_only_its_real_cpu_candidate() {
    let region = fixture_region(ValueFact::scalar(ValueKindFact::String));
    let region_id = region.id;
    let executor = GenericExecutor::compile(fixture_with_regions(vec![region])).unwrap();
    let candidate = executor
        .cpu_candidate(region_id, 4, Some(42))
        .expect("retained region candidate");

    assert_eq!(candidate.kind, ExecutionCandidateKind::GenericNativeCpu);
    assert_eq!(candidate.region, Some(region_id));
    assert_eq!(candidate.cost.components.execution_ns, 42);
    assert_eq!(candidate.cost.source, EstimateSource::Observation);
    assert_eq!(executor.region_contracts().len(), 1);
    assert!(candidate.validate().is_ok());
    assert!(executor
        .cpu_candidate(
            RegionId {
                function: ProgramFunctionId(9),
                ordinal: 0,
            },
            1,
            None,
        )
        .is_none());
}

#[test]
fn failed_representation_guard_materializes_and_resumes_exact_native_state() {
    let region = fixture_region(ValueFact::scalar(ValueKindFact::String));
    let executor = GenericExecutor::compile(fixture_with_regions(vec![region])).unwrap();
    let mut invocation = executor
        .begin_with_deoptimization(
            ProgramFunctionId(0),
            Vec::new(),
            Vec::new(),
            1,
            runtime_context(),
            DeoptimizationPolicy::default(),
        )
        .unwrap();
    let runmat_jit::execute::GenericInvocationStep::Deoptimized {
        reason,
        target,
        frame,
        ..
    } = invocation.advance().unwrap()
    else {
        panic!("mismatched representation guard must deoptimize")
    };
    assert_eq!(
        reason,
        runmat_runtime::native::NativeDeoptReason::REPRESENTATION
    );
    assert_eq!(
        target,
        runmat_runtime::native::NativeResumeKind::GENERIC_NATIVE
    );
    assert_eq!(frame.site.point.position, 1);
    assert_eq!(frame.locals[0].value, Some(Value::Num(41.0)));

    invocation.resume_deoptimization().unwrap();
    let runmat_jit::execute::GenericInvocationStep::Completed(execution) =
        invocation.advance().unwrap()
    else {
        panic!("retired guard must resume at the exact generic-native site")
    };
    assert_eq!(execution.outputs, vec![Value::Num(41.0)]);
}

#[test]
fn injected_failure_at_every_guard_materializes_and_resumes_exact_state() {
    let function = ProgramFunctionId(0);
    let value = RegionValueId { function, local: 0 };
    let numeric = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    });
    let expected = ValueFact::scalar(numeric.clone());
    let mut region = fixture_region(expected.clone());
    region.capabilities = CapabilitySet([CapabilityRequirement::NativeCode].into_iter().collect());
    let conditions = vec![
        RegionGuardCondition::ValueFact {
            value,
            expected: expected.clone(),
        },
        RegionGuardCondition::Shape {
            value,
            expected: ShapeFact::Scalar,
        },
        RegionGuardCondition::Class {
            value,
            expected: numeric,
        },
        RegionGuardCondition::Residency {
            value,
            expected: ResidencyFact::Host,
        },
        RegionGuardCondition::Alias {
            value,
            expected: AliasFact::Unique,
        },
        RegionGuardCondition::RuntimeState {
            identity: "catalog".into(),
            revision: "7".into(),
        },
        RegionGuardCondition::Capability {
            requirement: CapabilityRequirement::NativeCode,
        },
    ];
    region.guards = conditions
        .into_iter()
        .enumerate()
        .map(|(ordinal, condition)| RegionGuardContract {
            id: RegionGuardId {
                region: region.id,
                ordinal: u32::try_from(ordinal).unwrap(),
            },
            condition,
            deopt: DeoptimizationPointId {
                function,
                ordinal: u32::try_from(ordinal).unwrap(),
            },
        })
        .collect();
    let guards = region
        .guards
        .iter()
        .map(|guard| guard.id)
        .collect::<Vec<_>>();
    let assembly = fixture_with_regions(vec![region]);

    for guard in guards {
        let policy = DeoptimizationPolicy {
            guards: GuardEnvironment::default()
                .with_runtime_revision("catalog", "7")
                .with_capability(CapabilityRequirement::NativeCode),
            ..DeoptimizationPolicy::default()
        }
        .inject(FaultInjection::Guard(guard));
        let executor = GenericExecutor::compile(assembly.clone()).unwrap();
        let mut invocation = executor
            .begin_with_deoptimization(
                function,
                Vec::new(),
                Vec::new(),
                1,
                runtime_context(),
                policy,
            )
            .unwrap();
        let runmat_jit::execute::GenericInvocationStep::Deoptimized { frame, .. } =
            invocation.advance().unwrap()
        else {
            panic!("selected guard must deoptimize")
        };
        assert_eq!(frame.locals[0].value, Some(Value::Num(41.0)));
        invocation.resume_deoptimization().unwrap();
        let runmat_jit::execute::GenericInvocationStep::Completed(execution) =
            invocation.advance().unwrap()
        else {
            panic!("retired guard must resume at the exact native site")
        };
        assert_eq!(execution.outputs, vec![Value::Num(41.0)]);
    }
}

#[test]
fn interpreter_target_is_selected_only_for_verified_empty_stack_resume_points() {
    let point = ProgramPointId {
        function: ProgramFunctionId(0),
        block: 0,
        position: 1,
    };
    let region = fixture_region(ValueFact::scalar(ValueKindFact::String));
    let executor = GenericExecutor::compile_with_resume_points(
        fixture_with_regions(vec![region]),
        None,
        BTreeMap::from([(point, 7)]),
    )
    .unwrap();
    let policy = DeoptimizationPolicy {
        target: ResumeTarget::Interpreter,
        ..DeoptimizationPolicy::default()
    };
    let mut invocation = executor
        .begin_with_deoptimization(
            ProgramFunctionId(0),
            Vec::new(),
            Vec::new(),
            1,
            runtime_context(),
            policy,
        )
        .unwrap();
    let runmat_jit::execute::GenericInvocationStep::Deoptimized { target, frame, .. } =
        invocation.advance().unwrap()
    else {
        panic!("mismatched representation guard must deoptimize")
    };
    assert_eq!(
        target,
        runmat_runtime::native::NativeResumeKind::INTERPRETER
    );
    assert_eq!(frame.site.bytecode_pc, Some(7));
    assert!(frame.operands.is_empty());
    assert_eq!(invocation.resume_state().bytecode_pc, 7);
}

#[test]
fn injected_failure_at_every_safepoint_does_not_replay_completed_calls() {
    let assembly = two_semantic_calls_fixture();
    let safepoints = assembly.functions[0].blocks[0]
        .instructions
        .iter()
        .filter_map(|instruction| instruction.safepoint)
        .collect::<Vec<_>>();
    assert_eq!(safepoints.len(), 2);
    for (index, safepoint) in safepoints.into_iter().enumerate() {
        let runtime = runtime_context();
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let activation = runtime.enter();
        let invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
            Arc::new(move |_function, arguments, _requested_outputs| {
                observed.fetch_add(1, Ordering::SeqCst);
                let result = arguments[0].clone();
                Box::pin(async move { Ok(result) })
            }),
        ));
        drop(activation);

        let executor = GenericExecutor::compile(assembly.clone()).unwrap();
        let policy = DeoptimizationPolicy::default().inject(FaultInjection::Safepoint(safepoint));
        let mut invocation = executor
            .begin_with_deoptimization(
                ProgramFunctionId(0),
                Vec::new(),
                Vec::new(),
                1,
                runtime,
                policy,
            )
            .unwrap();
        let runmat_jit::execute::GenericInvocationStep::Deoptimized { frame, .. } =
            invocation.advance().unwrap()
        else {
            panic!("selected safepoint must deoptimize")
        };
        assert_eq!(calls.load(Ordering::SeqCst), index);
        if index == 1 {
            assert_eq!(frame.locals[0].value, Some(Value::Num(11.0)));
        }

        invocation.resume_deoptimization().unwrap();
        let runmat_jit::execute::GenericInvocationStep::Completed(execution) =
            invocation.advance().unwrap()
        else {
            panic!("safepoint deoptimization must resume")
        };
        assert_eq!(execution.outputs, vec![Value::Num(22.0)]);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        drop(invoker);
    }
}

#[test]
fn gc_collection_during_deopt_resume_preserves_native_handle_values() {
    let rooted = runmat_gc::gc_allocate_rooted(Value::String("native-live".to_string()))
        .expect("allocate rooted native test value");
    let handle = rooted.handle();
    let handle_value = Value::HandleObject(runmat_value::HandleRef {
        class_name: "NativeGcValue".to_string(),
        target: handle,
        valid: true,
    });
    let runtime = runtime_context();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let activation = runtime.enter();
    // Runtime's scoped invoker API uses `Arc`, while GC handles are deliberately
    // thread-affine and this callback runs only in the active RuntimeContext.
    #[allow(clippy::arc_with_non_send_sync)]
    let invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(move |_function, _arguments, _requested_outputs| {
            let call = observed.fetch_add(1, Ordering::SeqCst);
            let value = handle_value.clone();
            Box::pin(async move {
                if call == 1 {
                    runmat_gc::gc_collect_major().map_err(|error| {
                        runmat_runtime::RuntimeError::new(format!(
                            "native GC stress collection failed: {error}"
                        ))
                    })?;
                    runmat_gc::gc_clone_value(&handle).map_err(|error| {
                        runmat_runtime::RuntimeError::new(format!(
                            "native arena did not retain its GC handle: {error}"
                        ))
                    })?;
                }
                Ok(value)
            })
        }),
    ));
    drop(activation);

    let assembly = two_semantic_calls_returning_first_fixture();
    let second_safepoint = assembly.functions[0].blocks[0]
        .instructions
        .iter()
        .filter_map(|instruction| instruction.safepoint)
        .nth(1)
        .expect("second semantic call safepoint");
    let executor = GenericExecutor::compile(assembly).unwrap();
    let policy =
        DeoptimizationPolicy::default().inject(FaultInjection::Safepoint(second_safepoint));
    let mut invocation = executor
        .begin_with_deoptimization(
            ProgramFunctionId(0),
            Vec::new(),
            Vec::new(),
            1,
            runtime,
            policy,
        )
        .unwrap();
    let runmat_jit::execute::GenericInvocationStep::Deoptimized { frame, .. } =
        invocation.advance().unwrap()
    else {
        panic!("second safepoint must deoptimize")
    };
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(matches!(
        frame.locals[0].value,
        Some(Value::HandleObject(ref value)) if value.target == handle
    ));
    rooted
        .unroot()
        .expect("transfer handle ownership to native arena");

    invocation.resume_deoptimization().unwrap();
    let runmat_jit::execute::GenericInvocationStep::Completed(execution) =
        invocation.advance().unwrap()
    else {
        panic!("GC-stressed deoptimization must resume")
    };
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(matches!(
        execution.outputs.as_slice(),
        [Value::HandleObject(value)] if value.target == handle
    ));
    drop(invoker);
}

#[test]
fn cancellation_exits_without_committing_speculative_outputs() {
    let executor = GenericExecutor::compile(fixture()).unwrap();
    let runtime = runtime_context();
    runtime
        .cancellation()
        .store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(matches!(
        executor.invoke(ProgramFunctionId(0), Vec::new(), 1, runtime),
        Err(JitError::Cancelled)
    ));
}

#[test]
fn suspending_semantic_call_exits_explicitly_without_nested_executor_or_replay() {
    let runtime = runtime_context();
    let activation = runtime.enter();
    let invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(|function, _arguments, requested_outputs| {
            assert_eq!(function, 9);
            assert_eq!(requested_outputs, 1);
            Box::pin(std::future::pending::<
                Result<Value, runmat_runtime::RuntimeError>,
            >())
        }),
    ));
    drop(activation);
    let executor = GenericExecutor::compile(semantic_call_fixture()).unwrap();
    let error = executor
        .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime)
        .expect_err("R13 must reject a real suspension boundary");
    let JitError::UnsupportedSite(message) = error else {
        panic!("suspension must remain a typed unsupported-site exit");
    };
    assert!(message.contains("requires the R14 continuation cohort"));
    drop(invoker);
}

#[test]
fn generated_branch_uses_shared_truth_semantics_and_exact_edge_values() {
    let executor = GenericExecutor::compile(branch_fixture()).unwrap();
    let execution = executor
        .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
        .unwrap();
    assert_eq!(execution.outputs, vec![Value::Num(7.0)]);
}

#[test]
fn generic_operator_and_range_sites_use_canonical_runtime_builtins() {
    let operator = GenericExecutor::compile(binary_fixture()).unwrap();
    assert_eq!(
        operator
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(42.0)]
    );

    let range = GenericExecutor::compile(range_fixture()).unwrap();
    let output = range
        .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
        .unwrap()
        .outputs
        .pop()
        .unwrap();
    let Value::Tensor(range) = output else {
        panic!("expected colon to produce a tensor")
    };
    assert_eq!(range.materialize_f64(), vec![1.0, 2.0, 3.0]);

    let call = GenericExecutor::compile(call_fixture()).unwrap();
    assert_eq!(
        call.invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(9.0)]
    );

    let aggregate = GenericExecutor::compile(aggregate_fixture()).unwrap();
    let Value::Tensor(matrix) = aggregate
        .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
        .unwrap()
        .outputs
        .pop()
        .unwrap()
    else {
        panic!("expected a tensor aggregate")
    };
    assert_eq!(matrix.shape, vec![1, 2]);
    assert_eq!(matrix.materialize_f64(), vec![1.0, 2.0]);
}

#[test]
fn generic_super_dispatch_uses_runtime_class_and_semantic_call_services() {
    let runtime = runtime_context();
    let activation = runtime.enter();
    runmat_runtime::class_registry::register_class(runmat_runtime::class_registry::RuntimeClass {
        name: "NativeParent".into(),
        parent: None,
        properties: std::collections::HashMap::new(),
        methods: [(
            "increment".into(),
            runmat_runtime::class_registry::RuntimeMethod {
                name: "increment".into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: runmat_types::MemberAccess::Public,
                function_name: "native_parent_increment".into(),
                implicit_class_argument: None,
            },
        )]
        .into_iter()
        .collect(),
    });
    runmat_runtime::class_registry::register_class(runmat_runtime::class_registry::RuntimeClass {
        name: "NativeChild".into(),
        parent: Some("NativeParent".into()),
        properties: std::collections::HashMap::new(),
        methods: std::collections::HashMap::new(),
    });
    let resolver = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
        Arc::new(|name| (name == "native_parent_increment").then_some(9)),
    ));
    let invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(|function, arguments, requested_outputs| {
            assert_eq!(function, 9);
            assert_eq!(requested_outputs, 1);
            let value = arguments[0].clone();
            Box::pin(async move {
                runmat_runtime::call_builtin_async("plus", &[value, Value::Num(1.0)]).await
            })
        }),
    ));
    drop(activation);

    let executor = GenericExecutor::compile(super_method_fixture()).unwrap();
    assert_eq!(
        executor
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime)
            .unwrap()
            .outputs,
        vec![Value::Num(6.0)]
    );

    let runtime = runtime_context();
    let activation = runtime.enter();
    runmat_runtime::class_registry::register_class(runmat_runtime::class_registry::RuntimeClass {
        name: "NativeParent".into(),
        parent: None,
        properties: std::collections::HashMap::new(),
        methods: std::collections::HashMap::new(),
    });
    runmat_runtime::class_registry::register_class(runmat_runtime::class_registry::RuntimeClass {
        name: "NativeChild".into(),
        parent: Some("NativeParent".into()),
        properties: std::collections::HashMap::new(),
        methods: std::collections::HashMap::new(),
    });
    drop(activation);
    let constructor = GenericExecutor::compile(super_constructor_fixture()).unwrap();
    let Value::Object(value) = constructor
        .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime)
        .unwrap()
        .outputs
        .pop()
        .unwrap()
    else {
        panic!("super constructor must return the active child receiver");
    };
    assert_eq!(value.class_name, "NativeChild");
    drop(invoker);
    drop(resolver);
}

#[test]
fn generic_short_circuit_and_switch_preserve_compiled_control_flow() {
    let short = GenericExecutor::compile(short_circuit_fixture()).unwrap();
    assert_eq!(
        short
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Bool(true)]
    );

    let switch = GenericExecutor::compile(switch_fixture()).unwrap();
    assert_eq!(
        switch
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(20.0)]
    );
}

#[test]
fn generic_for_iterates_captured_columns_through_compiled_backedges() {
    let executor = GenericExecutor::compile(for_fixture()).unwrap();
    assert_eq!(
        executor
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(6.0)]
    );
}

#[test]
fn generic_indexing_and_member_reads_use_shared_runtime_semantics() {
    let indexing = GenericExecutor::compile(index_fixture()).unwrap();
    assert_eq!(
        indexing
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(20.0)]
    );

    let member = GenericExecutor::compile(member_fixture()).unwrap();
    assert_eq!(
        member
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(42.0)]
    );
}

#[test]
fn generic_index_and_member_mutations_publish_updated_root_values() {
    let indexing = GenericExecutor::compile(index_assignment_fixture()).unwrap();
    let Value::Tensor(updated) = indexing
        .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
        .unwrap()
        .outputs
        .pop()
        .unwrap()
    else {
        panic!("expected updated tensor")
    };
    assert_eq!(updated.materialize_f64(), vec![10.0, 99.0, 30.0]);

    let member = GenericExecutor::compile(member_assignment_fixture()).unwrap();
    let Value::Struct(updated) = member
        .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
        .unwrap()
        .outputs
        .pop()
        .unwrap()
    else {
        panic!("expected updated structure")
    };
    assert_eq!(updated.fields.get("answer"), Some(&Value::Num(42.0)));

    let multi = GenericExecutor::compile(multi_assignment_fixture()).unwrap();
    assert_eq!(
        multi
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(33.0)]
    );
}

#[test]
fn generic_global_and_persistent_declarations_use_semantic_session_names() {
    let runtime = runtime_context();
    futures::executor::block_on(runtime.scope(async {
        runmat_runtime::workspace::session::store_global_named("shared", Value::Num(41.0));
        runmat_runtime::workspace::session::store_persistent_named(
            "persistent_counter",
            "calls",
            Value::Num(9.0),
        );
    }));

    let global = GenericExecutor::compile(workspace_binding_fixture(
        "global_counter",
        "shared",
        WorkspaceEffect::MutatesGlobal,
    ))
    .unwrap();
    assert_eq!(
        global
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime.clone())
            .unwrap()
            .outputs,
        vec![Value::Num(42.0)]
    );
    futures::executor::block_on(runtime.scope(async {
        assert_eq!(
            runmat_runtime::workspace::session::global_value("shared"),
            Some(Value::Num(42.0))
        );
    }));

    let persistent = GenericExecutor::compile(workspace_binding_fixture(
        "persistent_counter",
        "calls",
        WorkspaceEffect::MutatesPersistent,
    ))
    .unwrap();
    assert_eq!(
        persistent
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime.clone())
            .unwrap()
            .outputs,
        vec![Value::Num(10.0)]
    );
    futures::executor::block_on(runtime.scope(async {
        assert_eq!(
            runmat_runtime::workspace::session::persistent_named_value(
                "persistent_counter",
                "calls"
            ),
            Some(Value::Num(10.0))
        );
    }));
}

#[test]
fn generic_workspace_first_static_property_prefers_named_local_binding() {
    let executor = GenericExecutor::compile(workspace_first_static_fixture()).unwrap();
    assert_eq!(
        executor
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(55.0)]
    );
}

#[test]
fn generic_context_dependent_end_selectors_execute_once_against_the_base_shape() {
    let scalar = GenericExecutor::compile(end_scalar_fixture()).unwrap();
    assert_eq!(
        scalar
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(30.0)]
    );

    let range = GenericExecutor::compile(end_range_fixture()).unwrap();
    let Value::Tensor(selected) = range
        .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
        .unwrap()
        .outputs
        .pop()
        .unwrap()
    else {
        panic!("expected end-relative range selection to return a tensor")
    };
    assert_eq!(selected.materialize_f64(), vec![10.0, 20.0, 30.0]);

    let runtime = runtime_context();
    let calls = Arc::new(AtomicUsize::new(0));
    let invoker_calls = Arc::clone(&calls);
    let activation = runtime.enter();
    let invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(move |function, arguments, requested_outputs| {
            assert_eq!(function, 9);
            assert_eq!(requested_outputs, 1);
            invoker_calls.fetch_add(1, Ordering::SeqCst);
            let value = arguments[0].clone();
            Box::pin(async move { Ok(value) })
        }),
    ));
    drop(activation);
    let called = GenericExecutor::compile(end_call_fixture()).unwrap();
    assert_eq!(
        called
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime)
            .unwrap()
            .outputs,
        vec![Value::Num(40.0)]
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    drop(invoker);
}

fn runtime_context() -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
}

fn fixture() -> runmat_native_codegen::NativeAssembly {
    fixture_with_regions(Vec::new())
}

fn fixture_with_regions(regions: Vec<RegionContract>) -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 2 };
    let function = FunctionId(0);
    let mir = MirAssembly {
        bodies: [(
            function,
            MirBody {
                function,
                abi: FunctionAbi {
                    fixed_inputs: Vec::new(),
                    varargin: None,
                    fixed_outputs: Vec::new(),
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                locals: vec![MirLocal {
                    id: MirLocalId(0),
                    binding: None,
                    kind: MirLocalKind::Temporary,
                    span,
                }],
                blocks: vec![BasicBlock {
                    id: BasicBlockId(0),
                    statements: vec![MirStmt {
                        kind: MirStmtKind::Assign {
                            place: MirPlace::Local(MirLocalId(0)),
                            value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number(
                                "41".into(),
                            ))),
                        },
                        span,
                    }],
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(0))]),
                        span,
                    },
                }],
            },
        )]
        .into_iter()
        .collect(),
        functions: [(
            function,
            MirFunctionMetadata {
                source: ProgramSourceId(0),
                name: FunctionName("main".into()),
                parent: None,
                enclosing_class: None,
                kind: FunctionKind::SyntheticEntrypoint,
                argument_validations: Vec::new(),
                captures: Vec::new(),
                modifiers: FunctionModifiers::default(),
                span,
            },
        )]
        .into_iter()
        .collect(),
        classes: Vec::new(),
        entrypoints: vec![function],
    };
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let mut manifest = manifest(analysis.revision.schema_version);
    manifest.capabilities.0.extend(
        regions
            .iter()
            .flat_map(|region| region.capabilities.0.iter().copied()),
    );
    manifest.regions = regions;
    lower_executable(NativeLoweringInput {
        mir: &mir,
        analysis: &analysis,
        manifest: &manifest,
        binding_names: None,
        target: NativeTarget::current(),
    })
    .unwrap()
}

fn fixture_region(expected: ValueFact) -> RegionContract {
    let function = ProgramFunctionId(0);
    let region = RegionId {
        function,
        ordinal: 0,
    };
    let value = RegionValueId { function, local: 0 };
    RegionContract {
        schema_version: REGION_CONTRACT_SCHEMA_VERSION,
        id: region,
        source: ProgramSourceId(0),
        span: runmat_types::ProgramSpan { start: 0, end: 2 },
        entry: ProgramPointId {
            function,
            block: 0,
            position: 1,
        },
        exits: Vec::new(),
        live_in: vec![value],
        live_out: Vec::new(),
        value_facts: vec![RegionValueFact {
            value,
            fact: expected.clone(),
        }],
        effects: Default::default(),
        capabilities: Default::default(),
        guards: vec![RegionGuardContract {
            id: RegionGuardId { region, ordinal: 0 },
            condition: RegionGuardCondition::ValueFact { value, expected },
            deopt: DeoptimizationPointId {
                function,
                ordinal: 0,
            },
        }],
        provenance: RegionProvenance::Profiled {
            profile_digest: "guard-test".into(),
        },
    }
}

fn branch_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    let function = FunctionId(0);
    let local = |id| MirLocal {
        id: MirLocalId(id),
        binding: None,
        kind: MirLocalKind::Temporary,
        span,
    };
    let assign = |local, number: &str| MirStmt {
        kind: MirStmtKind::Assign {
            place: MirPlace::Local(MirLocalId(local)),
            value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number(number.into()))),
        },
        span,
    };
    let mir = MirAssembly {
        bodies: [(
            function,
            MirBody {
                function,
                abi: FunctionAbi {
                    fixed_inputs: Vec::new(),
                    varargin: None,
                    fixed_outputs: Vec::new(),
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                locals: vec![local(0), local(1)],
                blocks: vec![
                    BasicBlock {
                        id: BasicBlockId(0),
                        statements: vec![assign(0, "1")],
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Branch {
                                cond: MirOperand::Local(MirLocalId(0)),
                                then_block: BasicBlockId(1),
                                else_block: BasicBlockId(2),
                            },
                            span,
                        },
                    },
                    BasicBlock {
                        id: BasicBlockId(1),
                        statements: vec![assign(1, "7")],
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(1))]),
                            span,
                        },
                    },
                    BasicBlock {
                        id: BasicBlockId(2),
                        statements: vec![assign(1, "9")],
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(1))]),
                            span,
                        },
                    },
                ],
            },
        )]
        .into_iter()
        .collect(),
        functions: [(
            function,
            MirFunctionMetadata {
                source: ProgramSourceId(0),
                name: FunctionName("branch".into()),
                parent: None,
                enclosing_class: None,
                kind: FunctionKind::SyntheticEntrypoint,
                argument_validations: Vec::new(),
                captures: Vec::new(),
                modifiers: FunctionModifiers::default(),
                span,
            },
        )]
        .into_iter()
        .collect(),
        classes: Vec::new(),
        entrypoints: vec![function],
    };
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(analysis.revision.schema_version);
    lower_executable(NativeLoweringInput {
        mir: &mir,
        analysis: &analysis,
        manifest: &manifest,
        binding_names: None,
        target: NativeTarget::current(),
    })
    .unwrap()
}

fn binary_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    let statement = |local, value| MirStmt {
        kind: MirStmtKind::Assign {
            place: MirPlace::Local(MirLocalId(local)),
            value,
        },
        span,
    };
    lower_body(
        "binary",
        3,
        vec![
            statement(
                0,
                MirRvalue::Use(MirOperand::Constant(MirConstant::Number("40".into()))),
            ),
            statement(
                1,
                MirRvalue::Use(MirOperand::Constant(MirConstant::Number("2".into()))),
            ),
            statement(
                2,
                MirRvalue::Binary(
                    MirOperand::Local(MirLocalId(0)),
                    runmat_types::OperatorKind::Add,
                    MirOperand::Local(MirLocalId(1)),
                ),
            ),
        ],
        MirOperand::Local(MirLocalId(2)),
        span,
    )
}

fn range_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    lower_body(
        "range",
        1,
        vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::Range {
                    start: MirOperand::Constant(MirConstant::Number("1".into())),
                    step: None,
                    end: MirOperand::Constant(MirConstant::Number("3".into())),
                },
            },
            span,
        }],
        MirOperand::Local(MirLocalId(0)),
        span,
    )
}

fn call_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    lower_body(
        "call",
        1,
        vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::Call(MirCall {
                    callee: MirCallee::Static(CallableIdentity::Builtin(BuiltinId("sqrt".into()))),
                    args: vec![MirCallArg::Single(MirOperand::Constant(
                        MirConstant::Number("81".into()),
                    ))],
                    arg_spans: vec![span],
                    syntax: runmat_hir::CallSyntax::Plain,
                    requested_outputs: RequestedOutputCount::One,
                    fallback_policy: CallableFallbackPolicy::None,
                    workspace_first_name: None,
                    bare_identifier: false,
                    async_behavior: AsyncBehaviorFact::NeverSuspends,
                    effects: runmat_builtins::BuiltinEffects::none(),
                    workspace_effect: None,
                    environment_effect: None,
                    purity: runmat_builtins::BuiltinPurity::Pure,
                    semantic_kind: runmat_builtins::BuiltinSemanticKind::General,
                }),
            },
            span,
        }],
        MirOperand::Local(MirLocalId(0)),
        span,
    )
}

fn semantic_call_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    lower_body(
        "semantic_call",
        1,
        vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::Call(MirCall {
                    callee: MirCallee::Static(CallableIdentity::BoundFunction(
                        runmat_types::FunctionId(9),
                    )),
                    args: vec![MirCallArg::Single(MirOperand::Constant(
                        MirConstant::Number("1".into()),
                    ))],
                    arg_spans: vec![span],
                    syntax: runmat_hir::CallSyntax::Plain,
                    requested_outputs: RequestedOutputCount::One,
                    fallback_policy: CallableFallbackPolicy::None,
                    workspace_first_name: None,
                    bare_identifier: false,
                    async_behavior: AsyncBehaviorFact::NeverSuspends,
                    effects: runmat_builtins::BuiltinEffects::none(),
                    workspace_effect: None,
                    environment_effect: None,
                    purity: runmat_builtins::BuiltinPurity::Pure,
                    semantic_kind: runmat_builtins::BuiltinSemanticKind::General,
                }),
            },
            span,
        }],
        MirOperand::Local(MirLocalId(0)),
        span,
    )
}

fn two_semantic_calls_fixture() -> runmat_native_codegen::NativeAssembly {
    two_semantic_calls_fixture_returning(1, "two_semantic_calls")
}

fn two_semantic_calls_returning_first_fixture() -> runmat_native_codegen::NativeAssembly {
    two_semantic_calls_fixture_returning(0, "two_semantic_calls_returning_first")
}

fn two_semantic_calls_fixture_returning(
    return_local: usize,
    name: &str,
) -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 4 };
    let call = |argument: &str| {
        MirRvalue::Call(MirCall {
            callee: MirCallee::Static(CallableIdentity::BoundFunction(runmat_types::FunctionId(9))),
            args: vec![MirCallArg::Single(MirOperand::Constant(
                MirConstant::Number(argument.into()),
            ))],
            arg_spans: vec![span],
            syntax: runmat_hir::CallSyntax::Plain,
            requested_outputs: RequestedOutputCount::One,
            fallback_policy: CallableFallbackPolicy::None,
            workspace_first_name: None,
            bare_identifier: false,
            async_behavior: AsyncBehaviorFact::NeverSuspends,
            effects: runmat_builtins::BuiltinEffects::none(),
            workspace_effect: None,
            environment_effect: None,
            purity: runmat_builtins::BuiltinPurity::Pure,
            semantic_kind: runmat_builtins::BuiltinSemanticKind::General,
        })
    };
    lower_body(
        name,
        2,
        vec![
            statement(0, call("11"), span),
            statement(1, call("22"), span),
        ],
        MirOperand::Local(MirLocalId(return_local)),
        span,
    )
}

fn super_method_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    lower_body(
        "super_method",
        1,
        vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::Call(MirCall {
                    callee: MirCallee::SuperMethod {
                        current_class: "NativeChild".into(),
                        super_class: "NativeParent".into(),
                        method: "increment".into(),
                    },
                    args: vec![MirCallArg::Single(MirOperand::Constant(
                        MirConstant::Number("5".into()),
                    ))],
                    arg_spans: vec![span],
                    syntax: runmat_hir::CallSyntax::Plain,
                    requested_outputs: RequestedOutputCount::One,
                    fallback_policy: CallableFallbackPolicy::None,
                    workspace_first_name: None,
                    bare_identifier: false,
                    async_behavior: AsyncBehaviorFact::NeverSuspends,
                    effects: runmat_builtins::BuiltinEffects::none(),
                    workspace_effect: None,
                    environment_effect: None,
                    purity: runmat_builtins::BuiltinPurity::Pure,
                    semantic_kind: runmat_builtins::BuiltinSemanticKind::General,
                }),
            },
            span,
        }],
        MirOperand::Local(MirLocalId(0)),
        span,
    )
}

fn super_constructor_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    lower_body(
        "super_constructor",
        1,
        vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::Call(MirCall {
                    callee: MirCallee::SuperConstructor {
                        current_class: "NativeChild".into(),
                        super_class: "NativeParent".into(),
                    },
                    args: Vec::new(),
                    arg_spans: Vec::new(),
                    syntax: runmat_hir::CallSyntax::Plain,
                    requested_outputs: RequestedOutputCount::One,
                    fallback_policy: CallableFallbackPolicy::None,
                    workspace_first_name: None,
                    bare_identifier: false,
                    async_behavior: AsyncBehaviorFact::NeverSuspends,
                    effects: runmat_builtins::BuiltinEffects::none(),
                    workspace_effect: None,
                    environment_effect: None,
                    purity: runmat_builtins::BuiltinPurity::Pure,
                    semantic_kind: runmat_builtins::BuiltinSemanticKind::General,
                }),
            },
            span,
        }],
        MirOperand::Local(MirLocalId(0)),
        span,
    )
}

fn aggregate_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    lower_body(
        "aggregate",
        1,
        vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::Aggregate {
                    kind: MirAggregateKind::Tensor,
                    rows: 1,
                    cols: 2,
                    elements: vec![
                        MirOperand::Constant(MirConstant::Number("1".into())),
                        MirOperand::Constant(MirConstant::Number("2".into())),
                    ],
                },
            },
            span,
        }],
        MirOperand::Local(MirLocalId(0)),
        span,
    )
}

fn short_circuit_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    lower_body(
        "short",
        1,
        vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::ShortCircuit {
                    left: MirOperand::Constant(MirConstant::Bool(true)),
                    op: runmat_mir::MirShortCircuitOp::Or,
                    right_temps: Vec::new(),
                    right: MirOperand::Constant(MirConstant::Bool(false)),
                },
            },
            span,
        }],
        MirOperand::Local(MirLocalId(0)),
        span,
    )
}

fn switch_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 4 };
    let function = FunctionId(0);
    let return_block = |id, number: &str| BasicBlock {
        id: BasicBlockId(id),
        statements: vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number(number.into()))),
            },
            span,
        }],
        terminator: MirTerminator {
            kind: MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(0))]),
            span,
        },
    };
    let mir = MirAssembly {
        bodies: [(
            function,
            MirBody {
                function,
                abi: FunctionAbi {
                    fixed_inputs: Vec::new(),
                    varargin: None,
                    fixed_outputs: Vec::new(),
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                locals: vec![MirLocal {
                    id: MirLocalId(0),
                    binding: None,
                    kind: MirLocalKind::Temporary,
                    span,
                }],
                blocks: vec![
                    BasicBlock {
                        id: BasicBlockId(0),
                        statements: Vec::new(),
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Switch {
                                discr: MirOperand::Constant(MirConstant::Number("2".into())),
                                cases: vec![
                                    (
                                        MirOperand::Constant(MirConstant::Number("1".into())),
                                        BasicBlockId(1),
                                    ),
                                    (
                                        MirOperand::Constant(MirConstant::Number("2".into())),
                                        BasicBlockId(2),
                                    ),
                                ],
                                otherwise: BasicBlockId(3),
                            },
                            span,
                        },
                    },
                    return_block(1, "10"),
                    return_block(2, "20"),
                    return_block(3, "30"),
                ],
            },
        )]
        .into_iter()
        .collect(),
        functions: [(
            function,
            MirFunctionMetadata {
                source: ProgramSourceId(0),
                name: FunctionName("switch".into()),
                parent: None,
                enclosing_class: None,
                kind: FunctionKind::SyntheticEntrypoint,
                argument_validations: Vec::new(),
                captures: Vec::new(),
                modifiers: FunctionModifiers::default(),
                span,
            },
        )]
        .into_iter()
        .collect(),
        classes: Vec::new(),
        entrypoints: vec![function],
    };
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(analysis.revision.schema_version);
    lower_executable(NativeLoweringInput {
        mir: &mir,
        analysis: &analysis,
        manifest: &manifest,
        binding_names: None,
        target: NativeTarget::current(),
    })
    .unwrap()
}

fn for_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 4 };
    let function = FunctionId(0);
    let local = |id| MirLocal {
        id: MirLocalId(id),
        binding: None,
        kind: MirLocalKind::Temporary,
        span,
    };
    let mir = MirAssembly {
        bodies: [(
            function,
            MirBody {
                function,
                abi: FunctionAbi {
                    fixed_inputs: Vec::new(),
                    varargin: None,
                    fixed_outputs: Vec::new(),
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                locals: vec![local(0), local(1)],
                blocks: vec![
                    BasicBlock {
                        id: BasicBlockId(0),
                        statements: vec![MirStmt {
                            kind: MirStmtKind::Assign {
                                place: MirPlace::Local(MirLocalId(1)),
                                value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number(
                                    "0".into(),
                                ))),
                            },
                            span,
                        }],
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Goto(BasicBlockId(1)),
                            span,
                        },
                    },
                    BasicBlock {
                        id: BasicBlockId(1),
                        statements: Vec::new(),
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::For {
                                binding: MirLocalId(0),
                                iterable: MirRvalue::Range {
                                    start: MirOperand::Constant(MirConstant::Number("1".into())),
                                    step: None,
                                    end: MirOperand::Constant(MirConstant::Number("3".into())),
                                },
                                body_block: BasicBlockId(2),
                                exit_block: BasicBlockId(3),
                            },
                            span,
                        },
                    },
                    BasicBlock {
                        id: BasicBlockId(2),
                        statements: vec![MirStmt {
                            kind: MirStmtKind::Assign {
                                place: MirPlace::Local(MirLocalId(1)),
                                value: MirRvalue::Binary(
                                    MirOperand::Local(MirLocalId(1)),
                                    runmat_types::OperatorKind::Add,
                                    MirOperand::Local(MirLocalId(0)),
                                ),
                            },
                            span,
                        }],
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Goto(BasicBlockId(1)),
                            span,
                        },
                    },
                    BasicBlock {
                        id: BasicBlockId(3),
                        statements: Vec::new(),
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(1))]),
                            span,
                        },
                    },
                ],
            },
        )]
        .into_iter()
        .collect(),
        functions: [(
            function,
            MirFunctionMetadata {
                source: ProgramSourceId(0),
                name: FunctionName("for_loop".into()),
                parent: None,
                enclosing_class: None,
                kind: FunctionKind::SyntheticEntrypoint,
                argument_validations: Vec::new(),
                captures: Vec::new(),
                modifiers: FunctionModifiers::default(),
                span,
            },
        )]
        .into_iter()
        .collect(),
        classes: Vec::new(),
        entrypoints: vec![function],
    };
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(analysis.revision.schema_version);
    lower_executable(NativeLoweringInput {
        mir: &mir,
        analysis: &analysis,
        manifest: &manifest,
        binding_names: None,
        target: NativeTarget::current(),
    })
    .unwrap()
}

fn index_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    lower_body(
        "index",
        2,
        vec![
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(0)),
                    value: MirRvalue::Aggregate {
                        kind: MirAggregateKind::Tensor,
                        rows: 1,
                        cols: 3,
                        elements: vec![
                            MirOperand::Constant(MirConstant::Number("10".into())),
                            MirOperand::Constant(MirConstant::Number("20".into())),
                            MirOperand::Constant(MirConstant::Number("30".into())),
                        ],
                    },
                },
                span,
            },
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(1)),
                    value: MirRvalue::Index {
                        base: MirOperand::Local(MirLocalId(0)),
                        indexing: MirIndexing {
                            kind: runmat_types::IndexKind::Paren,
                            plan: MirIndexPlan::Scalar,
                            components: vec![MirIndexComponent::Expr(MirOperand::Constant(
                                MirConstant::Number("2".into()),
                            ))],
                            result_context: runmat_types::IndexResultContext::ReadSingle,
                            cell_expand_all: false,
                        },
                    },
                },
                span,
            },
        ],
        MirOperand::Local(MirLocalId(1)),
        span,
    )
}

fn member_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    lower_body(
        "member",
        2,
        vec![
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(0)),
                    value: MirRvalue::StructLiteral {
                        fields: vec![(
                            runmat_types::MemberName("answer".into()),
                            MirOperand::Constant(MirConstant::Number("42".into())),
                        )],
                    },
                },
                span,
            },
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(1)),
                    value: MirRvalue::Member {
                        base: MirOperand::Local(MirLocalId(0)),
                        member: runmat_types::MemberName("answer".into()),
                    },
                },
                span,
            },
        ],
        MirOperand::Local(MirLocalId(1)),
        span,
    )
}

fn index_assignment_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 4 };
    let indexing = MirIndexing {
        kind: runmat_types::IndexKind::Paren,
        plan: MirIndexPlan::Scalar,
        components: vec![MirIndexComponent::Expr(MirOperand::Constant(
            MirConstant::Number("2".into()),
        ))],
        result_context: runmat_types::IndexResultContext::AssignmentTarget,
        cell_expand_all: false,
    };
    let place = MirPlace::Index(Box::new(MirPlace::Local(MirLocalId(0))), indexing);
    lower_body(
        "index_assignment",
        1,
        vec![
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(0)),
                    value: MirRvalue::Aggregate {
                        kind: MirAggregateKind::Tensor,
                        rows: 1,
                        cols: 3,
                        elements: vec![
                            MirOperand::Constant(MirConstant::Number("10".into())),
                            MirOperand::Constant(MirConstant::Number("20".into())),
                            MirOperand::Constant(MirConstant::Number("30".into())),
                        ],
                    },
                },
                span,
            },
            MirStmt {
                kind: MirStmtKind::PlaceMutation(MirPlaceMutation {
                    place: place.clone(),
                    kind: runmat_types::PlaceMutationKind::IndexedAssign,
                    creation_policy: runmat_types::AssignmentCreationPolicy::CreateArrayByIndex,
                    shape_policy: runmat_types::AssignmentShapePolicy::MatlabCompatible,
                }),
                span,
            },
            MirStmt {
                kind: MirStmtKind::Assign {
                    place,
                    value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number("99".into()))),
                },
                span,
            },
        ],
        MirOperand::Local(MirLocalId(0)),
        span,
    )
}

fn member_assignment_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 4 };
    let place = MirPlace::Member(
        Box::new(MirPlace::Local(MirLocalId(0))),
        runmat_types::MemberName("answer".into()),
    );
    lower_body(
        "member_assignment",
        1,
        vec![
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(0)),
                    value: MirRvalue::StructLiteral {
                        fields: vec![(
                            runmat_types::MemberName("answer".into()),
                            MirOperand::Constant(MirConstant::Number("1".into())),
                        )],
                    },
                },
                span,
            },
            MirStmt {
                kind: MirStmtKind::PlaceMutation(MirPlaceMutation {
                    place: place.clone(),
                    kind: runmat_types::PlaceMutationKind::MemberAssign,
                    creation_policy: runmat_types::AssignmentCreationPolicy::CreateStructFieldPath,
                    shape_policy: runmat_types::AssignmentShapePolicy::MatlabCompatible,
                }),
                span,
            },
            MirStmt {
                kind: MirStmtKind::Assign {
                    place,
                    value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number("42".into()))),
                },
                span,
            },
        ],
        MirOperand::Local(MirLocalId(0)),
        span,
    )
}

fn multi_assignment_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 4 };
    lower_body(
        "multi_assignment",
        4,
        vec![
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(0)),
                    value: MirRvalue::Aggregate {
                        kind: MirAggregateKind::Cell,
                        rows: 1,
                        cols: 2,
                        elements: vec![
                            MirOperand::Constant(MirConstant::Number("11".into())),
                            MirOperand::Constant(MirConstant::Number("22".into())),
                        ],
                    },
                },
                span,
            },
            MirStmt {
                kind: MirStmtKind::MultiAssign {
                    targets: MirOutputTargetList {
                        targets: vec![
                            MirOutputTarget::Place(MirPlace::Local(MirLocalId(1))),
                            MirOutputTarget::Place(MirPlace::Local(MirLocalId(2))),
                        ],
                        requested_outputs: RequestedOutputCount::Exactly(2),
                    },
                    value: MirRvalue::Index {
                        base: MirOperand::Local(MirLocalId(0)),
                        indexing: MirIndexing {
                            kind: runmat_types::IndexKind::Brace,
                            plan: MirIndexPlan::Cell,
                            components: vec![MirIndexComponent::Colon],
                            result_context: runmat_types::IndexResultContext::ReadCommaList,
                            cell_expand_all: true,
                        },
                    },
                },
                span,
            },
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(3)),
                    value: MirRvalue::Binary(
                        MirOperand::Local(MirLocalId(1)),
                        runmat_types::OperatorKind::Add,
                        MirOperand::Local(MirLocalId(2)),
                    ),
                },
                span,
            },
        ],
        MirOperand::Local(MirLocalId(3)),
        span,
    )
}

fn lower_body(
    name: &str,
    local_count: usize,
    statements: Vec<MirStmt>,
    returned: MirOperand,
    span: Span,
) -> runmat_native_codegen::NativeAssembly {
    let function = FunctionId(0);
    let mir = MirAssembly {
        bodies: [(
            function,
            MirBody {
                function,
                abi: FunctionAbi {
                    fixed_inputs: Vec::new(),
                    varargin: None,
                    fixed_outputs: Vec::new(),
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                locals: (0..local_count)
                    .map(|id| MirLocal {
                        id: MirLocalId(id),
                        binding: None,
                        kind: MirLocalKind::Temporary,
                        span,
                    })
                    .collect(),
                blocks: vec![BasicBlock {
                    id: BasicBlockId(0),
                    statements,
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::Return(vec![returned]),
                        span,
                    },
                }],
            },
        )]
        .into_iter()
        .collect(),
        functions: [(
            function,
            MirFunctionMetadata {
                source: ProgramSourceId(0),
                name: FunctionName(name.into()),
                parent: None,
                enclosing_class: None,
                kind: FunctionKind::SyntheticEntrypoint,
                argument_validations: Vec::new(),
                captures: Vec::new(),
                modifiers: FunctionModifiers::default(),
                span,
            },
        )]
        .into_iter()
        .collect(),
        classes: Vec::new(),
        entrypoints: vec![function],
    };
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(analysis.revision.schema_version);
    lower_executable(NativeLoweringInput {
        mir: &mir,
        analysis: &analysis,
        manifest: &manifest,
        binding_names: None,
        target: NativeTarget::current(),
    })
    .unwrap()
}

fn workspace_binding_fixture(
    name: &str,
    binding_name: &str,
    effect: WorkspaceEffect,
) -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    let function = FunctionId(0);
    let binding = BindingId(0);
    let mir = MirAssembly {
        bodies: [(
            function,
            MirBody {
                function,
                abi: FunctionAbi {
                    fixed_inputs: Vec::new(),
                    varargin: None,
                    fixed_outputs: Vec::new(),
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                locals: vec![MirLocal {
                    id: MirLocalId(0),
                    binding: Some(binding),
                    kind: MirLocalKind::Binding,
                    span,
                }],
                blocks: vec![BasicBlock {
                    id: BasicBlockId(0),
                    statements: vec![
                        MirStmt {
                            kind: MirStmtKind::WorkspaceEffect {
                                effect,
                                bindings: vec![MirLocalId(0)],
                            },
                            span,
                        },
                        MirStmt {
                            kind: MirStmtKind::Assign {
                                place: MirPlace::Local(MirLocalId(0)),
                                value: MirRvalue::Binary(
                                    MirOperand::Local(MirLocalId(0)),
                                    runmat_types::OperatorKind::Add,
                                    MirOperand::Constant(MirConstant::Number("1".into())),
                                ),
                            },
                            span,
                        },
                    ],
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(0))]),
                        span,
                    },
                }],
            },
        )]
        .into_iter()
        .collect(),
        functions: [(
            function,
            MirFunctionMetadata {
                source: ProgramSourceId(0),
                name: FunctionName(name.into()),
                parent: None,
                enclosing_class: None,
                kind: FunctionKind::SyntheticEntrypoint,
                argument_validations: Vec::new(),
                captures: Vec::new(),
                modifiers: FunctionModifiers::default(),
                span,
            },
        )]
        .into_iter()
        .collect(),
        classes: Vec::new(),
        entrypoints: vec![function],
    };
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(analysis.revision.schema_version);
    let binding_names = BTreeMap::from([(binding, binding_name.to_string())]);
    lower_executable(NativeLoweringInput {
        mir: &mir,
        analysis: &analysis,
        manifest: &manifest,
        binding_names: Some(&binding_names),
        target: NativeTarget::current(),
    })
    .unwrap()
}

fn workspace_first_static_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    let function = FunctionId(0);
    let binding = BindingId(0);
    let local = |id, binding, kind| MirLocal {
        id: MirLocalId(id),
        binding,
        kind,
        span,
    };
    let mir = MirAssembly {
        bodies: [(
            function,
            MirBody {
                function,
                abi: FunctionAbi {
                    fixed_inputs: Vec::new(),
                    varargin: None,
                    fixed_outputs: Vec::new(),
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                locals: vec![
                    local(0, Some(binding), MirLocalKind::Binding),
                    local(1, None, MirLocalKind::Temporary),
                ],
                blocks: vec![BasicBlock {
                    id: BasicBlockId(0),
                    statements: vec![
                        MirStmt {
                            kind: MirStmtKind::Assign {
                                place: MirPlace::Local(MirLocalId(0)),
                                value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number(
                                    "55".into(),
                                ))),
                            },
                            span,
                        },
                        MirStmt {
                            kind: MirStmtKind::Assign {
                                place: MirPlace::Local(MirLocalId(1)),
                                value: MirRvalue::WorkspaceFirstStaticProperty {
                                    workspace_name: runmat_hir::SymbolName("Candidate".into()),
                                    class_name: "Candidate".into(),
                                    property: runmat_hir::MemberName("Missing".into()),
                                },
                            },
                            span,
                        },
                    ],
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(1))]),
                        span,
                    },
                }],
            },
        )]
        .into_iter()
        .collect(),
        functions: [(
            function,
            MirFunctionMetadata {
                source: ProgramSourceId(0),
                name: FunctionName("workspace_first".into()),
                parent: None,
                enclosing_class: None,
                kind: FunctionKind::SyntheticEntrypoint,
                argument_validations: Vec::new(),
                captures: Vec::new(),
                modifiers: FunctionModifiers::default(),
                span,
            },
        )]
        .into_iter()
        .collect(),
        classes: Vec::new(),
        entrypoints: vec![function],
    };
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(analysis.revision.schema_version);
    let binding_names = BTreeMap::from([(binding, "Candidate".to_string())]);
    lower_executable(NativeLoweringInput {
        mir: &mir,
        analysis: &analysis,
        manifest: &manifest,
        binding_names: Some(&binding_names),
        target: NativeTarget::current(),
    })
    .unwrap()
}

fn end_scalar_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 4 };
    let mut statements = tensor_assignment(span);
    statements.extend([
        statement(1, MirRvalue::End, span),
        statement(
            2,
            MirRvalue::Binary(
                MirOperand::Local(MirLocalId(1)),
                runmat_types::OperatorKind::Subtract,
                MirOperand::Constant(MirConstant::Number("1".into())),
            ),
            span,
        ),
        statement(
            3,
            MirRvalue::Index {
                base: MirOperand::Local(MirLocalId(0)),
                indexing: MirIndexing {
                    kind: runmat_types::IndexKind::Paren,
                    plan: MirIndexPlan::SliceExpr,
                    components: vec![MirIndexComponent::Expr(MirOperand::Local(MirLocalId(2)))],
                    result_context: runmat_types::IndexResultContext::ReadSingle,
                    cell_expand_all: false,
                },
            },
            span,
        ),
    ]);
    lower_body(
        "end_scalar",
        4,
        statements,
        MirOperand::Local(MirLocalId(3)),
        span,
    )
}

fn end_range_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 5 };
    let mut statements = tensor_assignment(span);
    statements.extend([
        statement(1, MirRvalue::End, span),
        statement(
            2,
            MirRvalue::Binary(
                MirOperand::Local(MirLocalId(1)),
                runmat_types::OperatorKind::Subtract,
                MirOperand::Constant(MirConstant::Number("1".into())),
            ),
            span,
        ),
        statement(
            3,
            MirRvalue::Range {
                start: MirOperand::Constant(MirConstant::Number("1".into())),
                step: None,
                end: MirOperand::Local(MirLocalId(2)),
            },
            span,
        ),
        statement(
            4,
            MirRvalue::Index {
                base: MirOperand::Local(MirLocalId(0)),
                indexing: MirIndexing {
                    kind: runmat_types::IndexKind::Paren,
                    plan: MirIndexPlan::SliceExpr,
                    components: vec![MirIndexComponent::Expr(MirOperand::Local(MirLocalId(3)))],
                    result_context: runmat_types::IndexResultContext::ReadSingle,
                    cell_expand_all: false,
                },
            },
            span,
        ),
    ]);
    lower_body(
        "end_range",
        5,
        statements,
        MirOperand::Local(MirLocalId(4)),
        span,
    )
}

fn end_call_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 4 };
    let mut statements = tensor_assignment(span);
    statements.extend([
        statement(1, MirRvalue::End, span),
        statement(
            2,
            MirRvalue::Call(MirCall {
                callee: MirCallee::Static(CallableIdentity::BoundFunction(
                    runmat_types::FunctionId(9),
                )),
                args: vec![MirCallArg::Single(MirOperand::Local(MirLocalId(1)))],
                arg_spans: vec![span],
                syntax: runmat_hir::CallSyntax::Plain,
                requested_outputs: RequestedOutputCount::One,
                fallback_policy: CallableFallbackPolicy::None,
                workspace_first_name: None,
                bare_identifier: false,
                async_behavior: AsyncBehaviorFact::NeverSuspends,
                effects: runmat_builtins::BuiltinEffects::none(),
                workspace_effect: None,
                environment_effect: None,
                purity: runmat_builtins::BuiltinPurity::Pure,
                semantic_kind: runmat_builtins::BuiltinSemanticKind::General,
            }),
            span,
        ),
        statement(
            3,
            MirRvalue::Index {
                base: MirOperand::Local(MirLocalId(0)),
                indexing: MirIndexing {
                    kind: runmat_types::IndexKind::Paren,
                    plan: MirIndexPlan::SliceExpr,
                    components: vec![MirIndexComponent::Expr(MirOperand::Local(MirLocalId(2)))],
                    result_context: runmat_types::IndexResultContext::ReadSingle,
                    cell_expand_all: false,
                },
            },
            span,
        ),
    ]);
    lower_body(
        "end_call",
        4,
        statements,
        MirOperand::Local(MirLocalId(3)),
        span,
    )
}

fn tensor_assignment(span: Span) -> Vec<MirStmt> {
    vec![statement(
        0,
        MirRvalue::Aggregate {
            kind: MirAggregateKind::Tensor,
            rows: 1,
            cols: 4,
            elements: ["10", "20", "30", "40"]
                .into_iter()
                .map(|value| MirOperand::Constant(MirConstant::Number(value.into())))
                .collect(),
        },
        span,
    )]
}

fn statement(local: usize, value: MirRvalue, span: Span) -> MirStmt {
    MirStmt {
        kind: MirStmtKind::Assign {
            place: MirPlace::Local(MirLocalId(local)),
            value,
        },
        span,
    }
}

fn manifest(analysis_schema: u16) -> ExecutableUnitManifest {
    let program = ProgramRevision::new(
        Digest::sha256(b"r13-graph"),
        Digest::sha256(b"r13-sources"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"r13-runtime"),
            Digest::sha256(b"r13-catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap();
    let revisions = ExecutableComponentRevisions {
        catalog_schema: 1,
        catalog_fingerprint: *program.catalog_fingerprint(),
        contract_schema: 1,
        contract_fingerprint: Digest::sha256(b"r13-contracts"),
        analysis_schema,
        mir_schema: runmat_mir::MIR_SCHEMA_VERSION,
        bytecode_schema: 1,
        vm_layout_schema: 1,
        function_registry_schema: 1,
        source_map_schema: 1,
        region_schema: REGION_CONTRACT_SCHEMA_VERSION,
        interop_schema: INTEROP_MANIFEST_SCHEMA_VERSION,
        parallel_schema: PARALLEL_MANIFEST_SCHEMA_VERSION,
    };
    let payloads = ExecutableComponentKind::REQUIRED
        .into_iter()
        .map(|kind| {
            ExecutableComponentPayload::new(kind, format!("{kind:?}-r13").into_bytes()).unwrap()
        })
        .collect::<Vec<_>>();
    let components = payloads
        .iter()
        .map(|payload| {
            let schema = match payload.kind {
                ExecutableComponentKind::Mir => revisions.mir_schema,
                ExecutableComponentKind::Analysis => revisions.analysis_schema,
                ExecutableComponentKind::Bytecode => revisions.bytecode_schema,
                ExecutableComponentKind::VmLayout => revisions.vm_layout_schema,
                ExecutableComponentKind::FunctionRegistry => revisions.function_registry_schema,
                ExecutableComponentKind::SourceMap => revisions.source_map_schema,
            };
            ExecutableComponentDescriptor::from_payload(payload.kind, schema, &payload.bytes)
                .unwrap()
        })
        .collect();
    ExecutableUnitManifest {
        schema_version: EXECUTABLE_UNIT_SCHEMA_VERSION,
        identity: ExecutableIdentity {
            program,
            root_package: "r13-fixture@1.0.0".into(),
            entrypoint: "main".into(),
            entrypoint_function: ProgramFunctionId(0),
            entrypoint_kind: ExecutableEntrypointKind::Function,
            source_digest: Digest::sha256(b"r13-main.m"),
        },
        revisions,
        components,
        capabilities: CapabilitySet::default(),
        regions: Vec::new(),
        interop: InteropManifest::default(),
        parallel: ParallelManifest::default(),
        optional_sections: Vec::new(),
    }
}
