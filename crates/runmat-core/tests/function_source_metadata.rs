use runmat_core::{ExecutableSource, InvocationControl, ProcedureInvocation, RunMatSession};
use runmat_value::Value;

const SOURCE: &str = "function handles = metadata_file()\nhandles = localfunctions();\nend\nfunction y = child_local(x)\ny = x + 1;\nend\n";

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn function_file_excludes_primary_from_localfunctions() {
    let mut session = RunMatSession::with_options(false, false).expect("session");
    let unit = session
        .compile_executable_unit(
            ExecutableSource::new("path:function-source-metadata", "metadata_file.m", SOURCE),
            None,
        )
        .await
        .expect("function file compiles");
    let mut invocation = ProcedureInvocation::function("metadata_file", Vec::new());
    invocation.requested_outputs = 1;
    let output = session
        .invoke_executable(&unit, invocation, &InvocationControl::default())
        .await
        .expect("invoke primary function");
    let [Value::Cell(handles)] = output.as_slice() else {
        panic!("expected one local-functions cell result: {output:?}");
    };
    assert_eq!(handles.shape, vec![1, 1]);
    assert!(matches!(&handles.data()[0],
        Value::BoundFunctionHandle(handle) if handle.name == "child_local"));
}
