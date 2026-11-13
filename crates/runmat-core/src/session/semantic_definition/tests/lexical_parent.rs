//! Generated native entries use the same typed lexical parent and pass-through captures.

use crate::{
    ExecutableSource, InvocationControl, ProcedureInvocation, ProcedureTarget, RunMatSession,
};
use futures::executor::block_on;
use runmat_value::Value;

const SOURCE: &str = r#"
function result = lexicalParent()
    shared = 1;
    function value = reader()
        value = shared;
    end
    function value = writer()
        function increment()
            shared = shared + 2;
        end
        increment();
        value = reader();
    end
    result = writer() + shared;
end
"#;

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn generated_native_sibling_and_descendant_calls_return_shared_mutation() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).expect("native session");
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new("lexical-parent@1", "lexicalParent.m", SOURCE),
            None,
        ))
        .expect("compile lexical function unit");
        assert!(unit.functions().names.contains_key("lexicalParent"));
        for nested in ["reader", "writer", "increment"] {
            assert!(
                !unit.functions().names.contains_key(nested),
                "nested {nested} must not be published for textual lookup"
            );
        }
        let invocation = ProcedureInvocation {
            target: ProcedureTarget::Function("lexicalParent".into()),
            arguments: crate::ProcedureArguments::Source(Vec::new()),
            requested_outputs: 1,
        };
        let output = block_on(session.invoke_executable(
            &unit,
            invocation,
            &InvocationControl::default().force_generic_native(),
        ))
        .expect("generated native lexical call");
        assert_eq!(output.as_slice(), &[Value::Num(6.0)]);
    });
}
