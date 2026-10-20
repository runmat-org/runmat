//! Indexed operations and tensor construction retain one semantic effect per
//! selector when generic native execution resumes a completed operation.

use super::*;
use runmat_value::{Tensor, Value};

const SOURCE: &str = r#"
function [selected, changed, pair, nested, calls] = nativeSemanticSuspension()
    global selector_calls;
    selector_calls = 0;
    values = [10, 20, 30];
    selected = values(nextIndex());
    values(nextIndex()) = 99;
    changed = values(2);
    pair = [selected, changed];
    nested = values([nextIndex(), 3]);
    calls = selector_calls;
end

function index = nextIndex()
    global selector_calls;
    selector_calls = selector_calls + 1;
    index = 2;
end
"#;

#[test]
fn compiled_index_and_aggregate_operations_do_not_replay_nested_selector_effects() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).expect("session");
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(
                "native-semantic-suspension@1",
                "nativeSemanticSuspension.m",
                SOURCE,
            ),
            None,
        ))
        .expect("compile indexed read, write, and aggregate");
        let expected = [
            Value::Num(20.0),
            Value::Num(99.0),
            Value::Tensor(Tensor::new(vec![20.0, 99.0], vec![1, 2]).expect("expected row")),
            Value::Tensor(Tensor::new(vec![99.0, 30.0], vec![1, 2]).expect("nested row")),
            Value::Num(3.0),
        ];
        for control in [
            InvocationControl::default(),
            InvocationControl::default().force_generic_native(),
        ] {
            let outputs = block_on(session.invoke_executable(
                &unit,
                ProcedureInvocation {
                    target: ProcedureTarget::Function("nativeSemanticSuspension".into()),
                    arguments: crate::ProcedureArguments::Source(Vec::new()),
                    requested_outputs: expected.len(),
                },
                &control,
            ))
            .expect("execute selected compiled mode");
            assert_eq!(outputs.as_slice(), expected.as_slice());
        }
    });
}
