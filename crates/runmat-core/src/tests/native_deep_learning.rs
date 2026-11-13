use crate::{execute_text_request_for_testing, ExecutableSource, RunMatSession};
use futures::executor::block_on;
use runmat_jit::tiering::TieringConfig;

const SOURCE: &str = r#"
net = dlnetwork({sequenceInputLayer(2); globalAveragePooling1dLayer(); regressionLayer()});
data = single(reshape(1:24, 2, 3, 4));
x = dlarray(data, 'CBT');
[y, state] = forward(net, x);
assert(strcmp(class(y.Data), 'single'));
assert(strcmp(y.Format, 'CB'));
assert(isequal(y.Data, single([10 12 14; 11 13 15])));
assert(strcmp(class(state), 'table') && isempty(state.Value));
[raw, nextState] = predict(net, data);
assert(strcmp(class(raw), 'single'));
assert(isequal(raw, single([10 11; 12 13; 14 15])));
assert(strcmp(class(nextState), 'table') && isempty(nextState.Value));
input = dlarray(single([1 2 3]), 'CB');
[loss, gradients] = dlfeval(@nestedGradients, input);
assert(loss.Data == single(14));
inner = gradients{1};
first = inner{1};
second = gradients{2};
assert(strcmp(class(first.Data), 'single'));
assert(strcmp(class(second.Data), 'single'));
assert(strcmp(first.Format, 'CB'));
assert(isequal(first.Data, single([2 4 6])));
assert(isequal(second.Data, single([2 4 6])));
function [loss, gradients] = nestedGradients(x)
    loss = sum(x .* x, 'all');
    targets = {{x}, x};
    gradients = dlgradient(loss, targets, uint8(9));
end
"#;

#[test]
fn native_compilation_digest_is_stable_and_tracks_semantic_binding_layout() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let source = || ExecutableSource::new("native-digest@1", "digest.m", "result = 1;");
        let first = block_on(session.compile_executable_unit(source(), None)).unwrap();
        let repeated = block_on(session.compile_executable_unit(source(), None)).unwrap();
        assert_eq!(
            first.native_compilation_digest().unwrap(),
            repeated.native_compilation_digest().unwrap()
        );
        let outcome = execute_text_request_for_testing(&mut session, "ans = 42;").unwrap();
        assert!(outcome.error.is_none());
        let changed = block_on(session.compile_executable_unit(source(), None)).unwrap();
        assert_eq!(
            first.revision(),
            changed.revision(),
            "authored source is unchanged"
        );
        assert_ne!(
            first.entrypoint_workspace_bindings(),
            changed.entrypoint_workspace_bindings()
        );
        assert_ne!(
            first.native_compilation_digest().unwrap(),
            changed.native_compilation_digest().unwrap()
        );
    });
}

#[test]
fn native_compilation_digest_tracks_exact_retained_vm_capture() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let source = |revision, value| {
            ExecutableSource::new(
                revision,
                "capture.m",
                format!("result = helper();\nfunction result = helper()\nresult = {value};\nend"),
            )
        };
        let original = block_on(
            session.compile_executable_unit(source("native-capture-digest@inf", "Inf"), None),
        )
        .unwrap();
        let changed = block_on(
            session.compile_executable_unit(source("native-capture-digest@nan", "NaN"), None),
        )
        .unwrap();
        // Each retained VM payload must come from its own sealed authored
        // program. Editing bytecode after MIR projection is not admissible.
        assert_eq!(original.binding_names(), changed.binding_names());
        assert_ne!(
            runmat_vm::encode_interpreter_program_v2(original.functions()).unwrap(),
            runmat_vm::encode_interpreter_program_v2(changed.functions()).unwrap()
        );
        assert_ne!(
            original.native_compilation_digest().unwrap(),
            changed.native_compilation_digest().unwrap()
        );
    });
}

#[test]
fn hot_native_deep_learning_preserves_precision_axes_and_nested_adjoint_targets() {
    runmat_gc::gc_test_context(|| {
        for jit in [false, true] {
            let mut session = RunMatSession::with_options(jit, false).unwrap();
            if jit {
                session.use_generic_ready_route_for_testing();
                session.set_native_tiering_config_for_testing(TieringConfig {
                    generic_hot_threshold: 1,
                    specialized_hot_threshold: 2,
                    deterministic: true,
                    ..Default::default()
                });
            }
            let mut native_executions = 0;
            for _ in 0..4 {
                let result = execute_text_request_for_testing(&mut session, SOURCE).unwrap();
                assert!(result.error.is_none(), "{:?}", result.error);
                native_executions += usize::from(result.used_jit);
            }
            if jit {
                assert!(
                    native_executions >= 2,
                    "actual hot native execution is required"
                );
            } else {
                assert_eq!(native_executions, 0);
            }
        }
    });
}
