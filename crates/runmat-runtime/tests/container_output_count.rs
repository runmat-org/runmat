use futures::executor::block_on;
use runmat_runtime::call_builtin_async_with_outputs;
use runmat_value::Value;

#[test]
fn fixed_container_outputs_reject_before_payload_validation() {
    for (name, inputs) in [
        ("cell", 0),
        ("cell2mat", 1),
        ("cell2struct", 2),
        ("cellstr", 1),
        ("mat2cell", 1),
        ("num2cell", 1),
        ("fieldnames", 1),
        ("getfield", 1),
        ("isfield", 2),
        ("rmfield", 1),
        ("setfield", 1),
        ("struct", 0),
    ] {
        let error = block_on(call_builtin_async_with_outputs(
            name,
            &vec![Value::Num(0.); inputs],
            2,
        ))
        .expect_err("fixed container output count");
        assert_eq!(
            error.identifier(),
            Some(format!("RunMat:{name}:TooManyOutputs").as_str()),
            "{name}: {error}"
        );
    }
}
