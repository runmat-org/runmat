//! Actual admitted adapter CPU routes, not a scalar allocation exemption.
use super::*;
use runmat_runtime::context::host_effect::ManagedHostEffectLedger;
use runmat_runtime::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService,
};
struct Managed(ManagedHostEffectLedger);
impl RuntimeParallelService for Managed {
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
#[derive(Clone, Copy)]
enum Route {
    InstalledNoProvider,
    PrivateDisabled,
    ScalarProviderFallback,
}
struct CpuAdapter {
    route: Route,
    provider: &'static dyn AccelProvider,
}
impl RuntimeAccelerationService for CpuAdapter {
    fn prepare_admitted(
        &self,
        context: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        let provider = self.provider;
        let route = self.route;
        Box::pin(async move {
            let mut auto = NativeAutoOffload::new(
                provider,
                ThresholdConfig {
                    cpu_elem_per_elem: 1.0,
                    ..ThresholdConfig::default()
                },
            );
            if matches!(route, Route::PrivateDisabled) {
                auto.enabled = false;
            }
            let prepared = super::super::admitted::prepare_async(&auto, &context, request)
                .await
                .map_err(|error| *error)?;
            match route {
                Route::PrivateDisabled => assert!(prepared.observation.is_none()),
                Route::ScalarProviderFallback => assert!(prepared
                    .observation
                    .as_ref()
                    .is_some_and(BuiltinCallObservation::shared_fallback_selected)),
                Route::InstalledNoProvider => {
                    unreachable!("actual installed adapter is used directly")
                }
            }
            let (operands, guard) = prepared.into_execution();
            PreparedBuiltinOperands::new(operands, Some(Box::new(guard)))
        })
    }
}
fn runtime_for(
    route: Route,
    ledger: Option<ManagedHostEffectLedger>,
    session: Rc<crate::placement::PlacementSession>,
) -> RuntimeContext {
    let mut ports = RuntimeServicePorts::default().with_placement(session);
    if let Some(ledger) = ledger {
        ports = ports.with_parallel(Rc::new(Managed(ledger)));
    }
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(ports);
    if matches!(route, Route::InstalledNoProvider) {
        super::super::install_preparation_service(runtime)
    } else {
        let ports = runtime
            .service_ports()
            .clone()
            .with_acceleration(Rc::new(CpuAdapter {
                route,
                provider: super::admitted_owners::owner(),
            }));
        runtime.with_service_ports(ports)
    }
}
fn scalar(value: &Value) -> f64 {
    match value {
        Value::Num(v) => *v,
        Value::Tensor(v) if v.len() == 1 => v.materialize_f64()[0],
        _ => panic!("scalar result"),
    }
}

#[test]
fn scoped_policy_matrix_keeps_cpu_work_and_explicit_provider_intent_separate() {
    use runmat_runtime::context::{RuntimeAccelerationPolicy, RuntimeProviderPreference};
    let provider = super::admitted_owners::owner();
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    for (enabled, automatic_offload, fallback) in [
        (false, false, false),
        (true, false, false),
        (true, false, true),
    ] {
        let context = runtime_for(
            Route::InstalledNoProvider,
            None,
            Rc::new(crate::placement::PlacementSession::default()),
        )
        .with_acceleration_policy(RuntimeAccelerationPolicy {
            enabled,
            provider: RuntimeProviderPreference::Auto,
            automatic_offload,
            allow_host_fallback: fallback,
            calibrate: false,
        });
        let result = futures::executor::block_on(context.scope(async {
            assert!(!auto_enabled());
            assert_eq!(
                runmat_accelerate_api::provider().is_some(),
                enabled,
                "auto-offload=false does not deny an explicit admitted provider"
            );
            runmat_runtime::dispatcher::call_builtin_async_in_context(
                &context,
                "plus",
                &[Value::Num(2.0), Value::Num(5.0)],
            )
            .await
        }))
        .unwrap();
        assert_eq!(
            scalar(result.value()),
            7.0,
            "ordinary CPU work is not fallback"
        );
    }
}
#[test]
fn cpu_admitted_routes_keep_one_finite_output_owner_and_no_fictitious_placement_sample() {
    for route in [
        Route::InstalledNoProvider,
        Route::PrivateDisabled,
        Route::ScalarProviderFallback,
    ] {
        // No global option/provider initialization or device permission changes.
        let provider = if matches!(route, Route::ScalarProviderFallback) {
            Some(super::admitted_owners::owner() as &'static dyn AccelProvider)
        } else {
            None
        };
        let _provider = runmat_accelerate_api::ThreadProviderGuard::set(provider);
        for managed in [false, true] {
            let ledger = ManagedHostEffectLedger::new(64 * 1024 * 1024);
            let session = Rc::new(crate::placement::PlacementSession::default());
            let context = runtime_for(route, managed.then(|| ledger.clone()), session.clone());
            let result = futures::executor::block_on(context.scope(async {
                let first = runmat_runtime::dispatcher::call_builtin_async_in_context(
                    &context,
                    "plus",
                    &[Value::Num(2.0), Value::Num(5.0)],
                )
                .await?;
                runmat_runtime::dispatcher::call_builtin_async_in_context(
                    &context,
                    "times",
                    &[first.value().clone(), Value::Num(6.0)],
                )
                .await
            }))
            .unwrap();
            assert_eq!(scalar(result.value()), 42.0);
            if managed {
                assert!(
                    ledger.snapshot().peak_reserved_bytes > 0,
                    "the canonical owner was admitted, not bypassed"
                );
            }
            assert!(session.ordinary_feedback_for_test().is_empty());
            assert!(session.profile_snapshot().feedback.is_empty());
            drop(result);
            assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
            // Non-scalar output keeps its real claim until final physical drop.
            if !matches!(route, Route::ScalarProviderFallback) {
                let values = [
                    Value::Tensor(Tensor::new(vec![2.0, 5.0], vec![1, 2]).unwrap()),
                    Value::Num(6.0),
                ];
                let output = futures::executor::block_on(context.scope(
                    runmat_runtime::dispatcher::call_builtin_async_in_context(
                        &context, "times", &values,
                    ),
                ))
                .unwrap();
                let Value::Tensor(value) = output.value() else {
                    panic!("real array result")
                };
                assert_eq!(value.materialize_f64(), vec![12.0, 30.0]);
                if managed {
                    assert!(ledger.snapshot().live_claimed_bytes > 0);
                }
                drop(output);
                assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
            }
        }
    }
}
#[test]
fn local_function_under_the_same_finite_grant_reaches_the_original_assertion() {
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(None);
    runmat_vm::reset_thread_state_for_tests();
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            runmat_vm::reset_thread_state_for_tests();
        }
    }
    let _reset = Reset;
    let text:std::sync::Arc<str>=std::sync::Arc::from(
        "answer=sharedAnswer(); assert(answer==42); assert(false);\nfunction y=sharedAnswer()\ny=(2+5)*6;\nend\n");
    let ast = runmat_parser::parse(&text).expect("parse genuine function witness");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower source");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower assembly");
    let bytecode = runmat_vm::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
        .expect("compile source");
    let ordinal = bytecode
        .source_id
        .and_then(runmat_vm::BytecodeSourceRef::program)
        .expect("program source");
    let ledger = ManagedHostEffectLedger::new(64 * 1024 * 1024);
    let session = Rc::new(crate::placement::PlacementSession::default());
    let context = runtime_for(
        Route::InstalledNoProvider,
        Some(ledger.clone()),
        session.clone(),
    )
    .with_program_sources([runmat_runtime::source_context::ProgramSourceRecord {
        ordinal,
        name: std::sync::Arc::from("<finite-shared-answer>"),
        fullpath_name: None,
        text,
    }])
    .expect("bind original source");
    let error = futures::executor::block_on(context.scope(runmat_vm::interpret(&bytecode)))
        .expect_err("the intentionally false second assertion must fail");
    assert_eq!(
        error.identifier(),
        runmat_builtins::ASSERT_ERROR_ASSERTION_FAILED.identifier
    );
    assert!(!error.message.contains("no prepared output owner"));
    assert!(ledger.snapshot().peak_reserved_bytes > 0);
    assert!(session.ordinary_feedback_for_test().is_empty());
    assert!(session.profile_snapshot().feedback.is_empty());
}
