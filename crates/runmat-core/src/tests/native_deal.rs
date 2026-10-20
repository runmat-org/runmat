//! Deal's native witnesses distinguish VM lowering from an executed JIT entry.

use super::*;
use runmat_value::{IntegerStorage, Tensor, Value};

#[cfg(feature = "jit")]
fn deterministic_native_session() -> RunMatSession {
    let mut session = RunMatSession::with_options(true, false).expect("session init");
    session.set_native_tiering_config_for_testing(runmat_jit::tiering::TieringConfig {
        generic_hot_threshold: 1,
        specialized_hot_threshold: 100,
        deterministic: true,
        ..runmat_jit::tiering::TieringConfig::default()
    });
    session
}

fn wide_tensor() -> Value {
    Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_992, 9_007_199_254_740_993]),
            vec![1, 2],
        )
        .expect("exact wide integer storage"),
    )
}

#[test]
fn deal_replication_executes_in_submitted_parfor_iterations() {
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    let service = std::rc::Rc::new(InProcessMultiLabExecutionService::new(2));
    session.install_execution_services(service.clone());
    let source = "pool = parpool(2); input = [0x0020000000000001u64, 0x0020000000000002u64, 0x0020000000000003u64, 0x0020000000000004u64]; first = zeros(1, 4, 'uint64'); second = zeros(1, 4, 'uint64'); parfor (index = 1:4, 2); [left, right] = deal(input(index)); first(index) = left; second(index) = right; end;";
    let unit = block_on(session.compile_executable_unit(
        ExecutableSource::new("deal-parfor-test@1", "deal_parfor.m", source),
        None,
    ))
    .expect("compile parallel deal contract");
    let region = unit
        .bytecode()
        .parfor_regions
        .first()
        .expect("compiler-bound parallel region");
    assert!(
        region.contract.effects.0.iter().all(|effect| matches!(
            effect,
            runmat_types::EffectKind::MayThrow | runmat_types::EffectKind::Randomness
        )),
        "parallel deal effects: {:?}",
        region.contract.effects
    );
    assert!(
        region
            .input_variables()
            .all(|variable| variable.contract.transferability
                != runmat_types::ParallelTransferability::NonTransferable),
        "parallel deal inputs: {:?}",
        region.variables
    );
    let outcome =
        execute_text_request(&mut session, source).expect("execute parallel deal replication");
    assert!(outcome.diagnostics.is_empty(), "{:?}", outcome.diagnostics);
    assert_eq!(service.submitted_revisions().len(), 4);
    let variables = session.get_variables().expect("parallel outputs");
    let expected = IntegerStorage::U64((1..=4).map(|index| (1_u64 << 53) + index).collect());
    for name in ["first", "second"] {
        let Some(Value::Tensor(output)) = variables.get(name) else {
            panic!("expected exact parallel tensor for {name}");
        };
        assert_eq!(output.integer_storage(), Some(&expected));
        assert_eq!(output.shape(), [1, 4]);
    }
}

#[test]
fn deal_retains_distributed_identity_and_pool_retirement() {
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    session.install_execution_services(std::rc::Rc::new(InProcessMultiLabExecutionService::new(2)));
    let outcome = execute_text_request(
        &mut session,
        "pool = parpool(2); source = uint64([1, 0x0020000000000001u64]); original = distributed(source); [first, second] = deal(original); gatheredFirst = gather(first); gatheredSecond = gather(second);",
    )
    .expect("execute distributed deal replication");
    assert!(outcome.diagnostics.is_empty(), "{:?}", outcome.diagnostics);
    {
        let variables = session.get_variables().expect("distributed outputs");
        let Some(Value::Distributed(original)) = variables.get("original") else {
            panic!("expected distributed input");
        };
        for name in ["first", "second"] {
            let Some(Value::Distributed(alias)) = variables.get(name) else {
                panic!("expected retained distributed output for {name}");
            };
            assert_eq!(alias, original);
        }
        for name in ["gatheredFirst", "gatheredSecond"] {
            assert_eq!(variables.get(name), variables.get("source"));
        }
    }
    execute_text_request(&mut session, "delete(pool);").expect("retire distributed pool");
    for name in ["first", "second"] {
        let outcome = execute_text_request(&mut session, &format!("stale = gather({name});"))
            .expect("retired alias returns a diagnostic");
        assert!(outcome
            .diagnostics
            .iter()
            .any(|diagnostic| { diagnostic.code == "RunMat:parallel:StaleDistributedLease" }));
    }
}

fn assert_wide_aliases(left: &Value, right: &Value) {
    let (Value::Tensor(left), Value::Tensor(right)) = (left, right) else {
        panic!("deal must retain tensor class in both outputs");
    };
    let expected = IntegerStorage::U64(vec![9_007_199_254_740_992, 9_007_199_254_740_993]);
    assert_eq!(left.integer_storage(), Some(&expected));
    assert_eq!(right.integer_storage(), Some(&expected));
    assert!(
        left.host_buffer()
            .shares_allocation_with(right.host_buffer()),
        "deal replication must retain the same host allocation until mutation"
    );
}

#[test]
fn forced_generic_native_deal_preserves_requested_discarded_and_zero_outputs() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).expect("session init");
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(
                "native-deal-outputs@1",
                "nativeDealReplicate.m",
                "function [left, right] = nativeDealReplicate(x)\n[left, right] = deal(x);\nend\nfunction selected = nativeDealDiscard(x)\n[~, selected] = deal(uint8(3), x);\nend\nfunction nativeDealZero(x)\ndeal(x);\nend\n",
            ),
            None,
        ))
        .expect("compile exact native deal forms");
        let input = wide_tensor();
        let control = InvocationControl::default().force_generic_native();
        let invoke = |session: &mut RunMatSession, name: &str, requested_outputs| {
            block_on(session.invoke_executable(
                &unit,
                ProcedureInvocation {
                    target: ProcedureTarget::Function(name.into()),
                    arguments: crate::ProcedureArguments::Source(vec![input.clone()]),
                    requested_outputs,
                },
                &control,
            ))
            .unwrap_or_else(|error| panic!("{name}: {error:?}"))
        };

        let replicated = invoke(&mut session, "nativeDealReplicate", 2);
        assert_eq!(replicated.as_slice().len(), 2);
        assert_wide_aliases(&replicated.as_slice()[0], &replicated.as_slice()[1]);
        let Value::Tensor(original) = &input else {
            panic!("exact tensor input");
        };
        let Value::Tensor(first) = &replicated.as_slice()[0] else {
            panic!("exact first output");
        };
        assert!(
            original
                .host_buffer()
                .shares_allocation_with(first.host_buffer()),
            "native deal must not copy the input payload"
        );

        let selected = invoke(&mut session, "nativeDealDiscard", 1);
        assert_single_output(&selected, input.clone());
        let Value::Tensor(chosen) = &selected.as_slice()[0] else {
            panic!("discarded-output selection must preserve the tensor");
        };
        assert!(
            original
                .host_buffer()
                .shares_allocation_with(chosen.host_buffer()),
            "discarding the first deal output must not copy the selected payload"
        );
        let discarded = invoke(&mut session, "nativeDealZero", 0);
        assert_output_values(&discarded, &[]);
    });
}

#[test]
fn forced_generic_native_anonymous_body_forwards_requested_outputs() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).expect("session init");
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(
                "native-anonymous-forward@1",
                "nativeAnonForward.m",
                "function varargout = nativeAnonForward(x)\nbias = 2; operation = @(v) deal(v + bias, v + bias + 1);\n[varargout{1:nargout}] = operation(x);\nend\nfunction varargout = nativeAnonZero(x)\noperation = @(v) deal(v);\n[varargout{1:nargout}] = operation(x);\nend\n",
            ),
            None,
        ))
        .expect("compile anonymous forwarding forms");
        let control = InvocationControl::default().force_generic_native();
        let outputs = block_on(session.invoke_executable(
            &unit,
            ProcedureInvocation {
                target: ProcedureTarget::Function("nativeAnonForward".into()),
                arguments: crate::ProcedureArguments::Source(vec![Value::Num(5.0)]),
                requested_outputs: 2,
            },
            &control,
        ))
        .expect("execute native anonymous two-output forwarding");
        assert_output_values(&outputs, &[Value::Num(7.0), Value::Num(8.0)]);

        let zero = block_on(session.invoke_executable(
            &unit,
            ProcedureInvocation {
                target: ProcedureTarget::Function("nativeAnonZero".into()),
                arguments: crate::ProcedureArguments::Source(vec![Value::Num(5.0)]),
                requested_outputs: 0,
            },
            &control,
        ))
        .expect("execute native anonymous zero-output forwarding");
        assert_output_values(&zero, &[]);
    });
}

#[test]
fn forced_generic_native_deal_count_error_does_not_duplicate_prior_effects() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).expect("session init");
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(
                "native-deal-no-replay@1",
                "nativeDealNoReplay.m",
                "function calls = nativeDealNoReplay()\nglobal dealCalls; dealCalls = 0; caught = false;\ntry\n operation = @deal;\n [a,b,c] = operation(nextDealInput(), uint16(8));\ncatch failure\n caught = strcmp(failure.identifier, 'RunMat:deal:InputOutputCountMismatch');\nend\nassert(caught);\ncalls = dealCalls;\nend\nfunction value = nextDealInput()\nglobal dealCalls; dealCalls = dealCalls + 1;\nvalue = uint64(9007199254740992) + uint64(1);\nend\n",
            ),
            None,
        ))
        .expect("compile effect-counted deal error");
        let output = block_on(session.invoke_executable(
            &unit,
            ProcedureInvocation {
                target: ProcedureTarget::Function("nativeDealNoReplay".into()),
                arguments: crate::ProcedureArguments::Source(Vec::new()),
                requested_outputs: 1,
            },
            &InvocationControl::default().force_generic_native(),
        ))
        .expect("catch exact deal count error");
        assert_single_output(&output, Value::Num(1.0));
    });
}

#[test]
#[cfg(feature = "jit")]
fn interactive_deal_wide_integer_replication_reaches_a_jit_entry() {
    runmat_gc::gc_test_context(|| {
        let mut session = deterministic_native_session();
        execute_text_request(
            &mut session,
            "wide = uint64(9007199254740992) + uint64([0, 1]);",
        )
        .expect("seed exact wide integers");

        let mut entered_jit = false;
        for _ in 0..4 {
            let outcome = execute_text_request(&mut session, "[left, right] = deal(wide);")
                .expect("execute deal replication");
            entered_jit |= outcome.used_jit;
            let variables = session.get_variables().expect("workspace after deal");
            assert_wide_aliases(
                variables.get("left").expect("left output"),
                variables.get("right").expect("right output"),
            );
        }
        assert!(entered_jit, "deal witness must execute a native entry");
    });
}

#[test]
fn forced_generic_native_deal_count_error_keeps_its_identifier() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).expect("session init");
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(
                "native-deal-count@1",
                "nativeDealMismatch.m",
                "function [a, b, c] = nativeDealMismatch(x)\noperation = @deal;\n[a, b, c] = operation(x, uint16(8));\nend\n",
            ),
            None,
        ))
        .expect("compile dynamic count-error call");
        let error = block_on(session.invoke_executable(
            &unit,
            ProcedureInvocation {
                target: ProcedureTarget::Function("nativeDealMismatch".into()),
                arguments: crate::ProcedureArguments::Source(vec![wide_tensor()]),
                requested_outputs: 3,
            },
            &InvocationControl::default().force_generic_native(),
        ))
        .expect_err("generic-native deal count mismatch");
        let RunError::Runtime(error) = error else {
            panic!("expected a typed runtime deal error, got {error:?}");
        };
        assert_eq!(
            error.identifier(),
            Some("RunMat:deal:InputOutputCountMismatch")
        );
    });
}
