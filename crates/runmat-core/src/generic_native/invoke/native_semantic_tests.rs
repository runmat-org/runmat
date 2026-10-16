use futures::executor::block_on;
use runmat_value::Value;

use crate::{
    ExecutableSource, InvocationControl, ProcedureInvocation, ProcedureTarget, RunMatSession,
};

#[test]
fn current_bound_nested_and_escaped_handles_enter_native_with_their_activation() {
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    session.set_compat_mode(runmat_parser::CompatMode::RunMat);
    let unit = block_on(session.compile_executable_unit(
        ExecutableSource::new(
            "native-bound-lexical@1",
            "nativeBoundLexical.m",
            "function [direct, escaped] = nativeBoundLexical(x)\nshared = x;\nhandle = @bump;\ndirect = feval(handle, 2);\nreturned = makeAdder(7);\nescaped = feval(returned, 3);\nfunction y = bump(delta)\nshared = shared + delta;\ny = shared;\nend\nfunction result = makeAdder(base)\nfunction y = add(value)\ny = base + value;\nend\nresult = @add;\nend\nend\n",
        ),
        None,
    ))
    .expect("compile native bound lexical fixture");
    super::super::deopt::reset_native_lexical_completions();
    super::native_semantic::reset_bound_native_completions();
    let result = block_on(session.invoke_executable(
        &unit,
        ProcedureInvocation {
            target: ProcedureTarget::Function("nativeBoundLexical".into()),
            arguments: vec![Value::Num(4.0)],
            requested_outputs: 2,
        },
        &InvocationControl::default().force_generic_native(),
    ))
    .expect("execute forced native bound lexical fixture");
    assert_eq!(result.as_slice(), &[Value::Num(6.0), Value::Num(10.0)]);
    assert_eq!(
        super::native_semantic::bound_native_completions(),
        2,
        "both bound handles must complete generated native execution, not VM deoptimization"
    );
}

#[test]
fn escaped_sibling_mutations_complete_native_without_cross_activation_aliasing() {
    let source = r#"
function result = nativeMutableEscaped()
    [inc, peek] = makePair(4);
    a = feval(inc, 1);
    b = feval(peek);
    c = feval(inc, 2);
    [inc2, peek2] = makePair(10);
    d = feval(inc2, 3);
    e = feval(peek);
    f = feval(peek2);
    result = a + 10*b + 100*c + 1000*d + 10000*e + 100000*f;
    function [writer, reader] = makePair(seed)
        shared = seed;
        writer = @bump;
        reader = @read;
        function value = bump(delta)
            shared = shared + delta;
            value = shared;
        end
        function value = read()
            value = shared;
        end
    end
end
"#;
    for mode in [
        runmat_parser::CompatMode::Matlab,
        runmat_parser::CompatMode::RunMat,
    ] {
        runmat_gc::gc_test_context(|| {
            let mut session = RunMatSession::with_options(false, false).expect("session init");
            session.set_compat_mode(mode);
            let unit = block_on(session.compile_executable_unit(
                ExecutableSource::new("native-mutable-escaped@1", "nativeMutableEscaped.m", source),
                None,
            ))
            .expect("compile escaped sibling fixture");
            super::native_semantic::reset_bound_native_completions();
            let result = block_on(session.invoke_executable(
                &unit,
                ProcedureInvocation {
                    target: ProcedureTarget::Function("nativeMutableEscaped".into()),
                    arguments: Vec::new(),
                    requested_outputs: 1,
                },
                &InvocationControl::default().force_generic_native(),
            ))
            .expect("execute escaped sibling fixture in generated native lane");
            assert_eq!(result.as_slice(), &[Value::Num(1_383_755.0)]);
            assert_eq!(
                super::native_semantic::bound_native_completions(),
                6,
                "all six escaped/sibling calls must complete generated native execution"
            );
        });
    }
}

#[test]
fn undefined_indexed_root_completes_in_forced_generic_native_with_vm_parity() {
    let source = ExecutableSource::new(
        "native-undefined-index-root@1",
        "nativeUndefinedIndexRoot.m",
        "function output = nativeUndefinedIndexRoot()\nfor k = 1:3\n    g(k) = k^2;\nend\noutput = g;\nend\n",
    );
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    let unit = block_on(session.compile_executable_unit(source, None)).expect("compile unit");
    let invocation = ProcedureInvocation {
        target: ProcedureTarget::Function("nativeUndefinedIndexRoot".into()),
        arguments: Vec::new(),
        requested_outputs: 1,
    };
    let vm = block_on(session.invoke_executable(
        &unit,
        invocation.clone(),
        &InvocationControl::default(),
    ))
    .expect("VM indexed-root execution");
    let native = block_on(session.invoke_executable(
        &unit,
        invocation,
        &InvocationControl::default().force_generic_native(),
    ))
    .expect("forced generic-native indexed-root execution");
    let expected = Value::Tensor(
        runmat_value::Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).expect("expected row vector"),
    );
    assert_eq!(vm.as_slice(), std::slice::from_ref(&expected));
    assert_eq!(native.as_slice(), std::slice::from_ref(&expected));
}

#[test]
fn prepared_multi_output_initializes_undefined_indexed_root_with_vm_parity() {
    let source = ExecutableSource::new(
        "native-prepared-index-root@1",
        "nativePreparedIndexRoot.m",
        "function [output, y] = nativePreparedIndexRoot()\n[g(1), y] = deal(2, 3);\noutput = g;\nend\n",
    );
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    let unit = block_on(session.compile_executable_unit(source, None)).expect("compile unit");
    let invocation = ProcedureInvocation {
        target: ProcedureTarget::Function("nativePreparedIndexRoot".into()),
        arguments: Vec::new(),
        requested_outputs: 2,
    };
    let vm = block_on(session.invoke_executable(
        &unit,
        invocation.clone(),
        &InvocationControl::default(),
    ))
    .expect("VM prepared indexed-root execution");
    let native = block_on(session.invoke_executable(
        &unit,
        invocation,
        &InvocationControl::default().force_generic_native(),
    ))
    .expect("forced generic-native prepared indexed-root execution");
    let expected = [
        Value::Tensor(runmat_value::Tensor::new(vec![2.0], vec![1, 1]).unwrap()),
        Value::Num(3.0),
    ];
    assert_eq!(vm.as_slice(), &expected);
    assert_eq!(native.as_slice(), &expected);
}
