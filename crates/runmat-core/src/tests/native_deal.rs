//! Deal's native witnesses distinguish VM lowering from an executed JIT entry.

use super::*;
use runmat_value::{IntegerStorage, Tensor, Value};

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
                arguments: vec![wide_tensor()],
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
