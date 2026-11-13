//! Runtime contract failures and member values across execution mechanisms.
use super::*;

#[test]
fn protected_contract_failures_reach_the_handler_in_both_executors() {
    for (operation, definitions) in [
        ("ones(2, 2) * ones(3, 3)", ""),
        ("reshape(ones(2, 2), 3, 3)", ""),
        ("sqrt(struct('field', 1))", ""),
        ("uninitializedLocal()", "function value = uninitializedLocal()\nif false\nmissing = 1;\nend\nvalue = missing;\nend\n"),
        (
            "uninitializedOperand([1, 2, 3])",
            "function value = uninitializedOperand(input)\nif false\nmissing = 1;\nend\nvalue = (input + [1, 2, 3]) .* missing;\nend\n",
        ),
        (
            "missingOperand([1, 2, 3])",
            "function value = missingOperand(input, missing)\nvalue = (input + [1, 2, 3]) .* missing;\nend\n",
        ),
        (
            "scalarOnly([1, 2])",
            "function value = scalarOnly(value)\narguments\nvalue (1,1) double\nend\nend\n",
        ),
    ] {
        for helper in [false, true] {
            let mut session = RunMatSession::with_options(false, false).expect("session");
            let expression = if helper {
                "failingOperation()"
            } else {
                operation
            };
            let mut source = format!(
                "function caught = protectedContract()\ncaught = false;\ntry\nvalue = {expression};\ncatch err\ncaught = ~isempty(err.identifier);\nend\nend\n"
            );
            if helper {
                source.push_str(&format!(
                    "function value = failingOperation()\nvalue = {operation};\nend\n"
                ));
            }
            source.push_str(definitions);
            let unit = block_on(session.compile_executable_unit(
                ExecutableSource::new("runtime-contract-transfer", "protectedContract.m", source),
                None,
            ))
            .unwrap_or_else(|error| {
                panic!("protected operation must compile: {operation}: {error}")
            });
            for control in [
                InvocationControl::default(),
                InvocationControl::default().force_generic_native(),
            ] {
                let invocation = ProcedureInvocation {
                    target: ProcedureTarget::Function("protectedContract".into()),
                    arguments: crate::ProcedureArguments::Source(Vec::new()),
                    requested_outputs: 1,
                };
                let result = block_on(session.invoke_executable(&unit, invocation, &control))
                    .unwrap_or_else(|error| {
                        panic!("protected operation must reach catch: {operation}: {error}")
                    });
                assert_single_output(&result, runmat_value::Value::Bool(true));
            }
        }
    }
}

#[test]
fn member_inputs_remain_available_to_following_numeric_expressions() {
    for source in [
        "value = struct('samples', [0.1, 0.2, 0.3]); delta = value.samples - [0.1, 0.2, 0.3]; assert(max(abs(delta)) <= 1e-12);",
        "value = struct('samples', single([1, 2, 3])); assert(max(abs(value.samples - single([1, 2, 3]))) == 0);",
        "values = [0.1, 0.2, 0.3]; delta = values([1, 2, 3]) - [0.1, 0.2, 0.3]; assert(max(abs(delta)) <= 1e-12);",
        "transform = @(value) value; delta = transform([0.1, 0.2, 0.3]) - [0.1, 0.2, 0.3]; assert(max(abs(delta)) <= 1e-12);",
    ] {
        let mut session = RunMatSession::with_options(false, false).expect("session");
        let outcome = execute_text_request(&mut session, source)
            .unwrap_or_else(|error| panic!("member input must survive numeric planning: {source}: {error}"));
        assert!(outcome.diagnostics.is_empty(), "member expression diagnostics: {:#?}", outcome.diagnostics);
    }
    let mut session = RunMatSession::with_options(false, false).expect("session");
    let declaration = execute_text_request(&mut session,
        "classdef RuntimeMemberValue\nproperties\nsamples\nend\nmethods\nfunction obj = RuntimeMemberValue(value)\nobj.samples = value;\nend\nend\nend\n")
        .expect("register the ordinary value class");
    assert!(
        declaration.diagnostics.is_empty(),
        "class diagnostics: {:#?}",
        declaration.diagnostics
    );
    let outcome = execute_text_request(&mut session,
        "value = RuntimeMemberValue([0.1, 0.2, 0.3]); assert(max(abs(value.samples - [0.1, 0.2, 0.3])) <= 1e-12);")
        .expect("evaluate an ordinary object property in a numeric expression");
    assert!(
        outcome.diagnostics.is_empty(),
        "object member diagnostics: {:#?}",
        outcome.diagnostics
    );
}
