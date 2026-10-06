use super::*;
use runmat_value::CharArray;

#[test]
fn cold_original_program_reads_and_mutates_only_its_installed_environment() {
    let name = "RUNMAT_PORTABLE_CONTEXT";
    let host = std::env::var_os(name);
    let original = request("ans = getenv('RUNMAT_PORTABLE_CONTEXT');", None, 1);
    let cold = serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
    let mut first = RunMatSession::with_options(false, false).unwrap();
    first
        .install_environment_snapshot(vec![(name.into(), "first".into())])
        .unwrap();
    let mut second = RunMatSession::with_options(false, false).unwrap();
    second
        .install_environment_snapshot(vec![(name.into(), "second".into())])
        .unwrap();
    assert_eq!(
        values(invoke_session(
            &mut first,
            &cold,
            &InvocationControl::default()
        )),
        [Value::CharArray(CharArray::new_row("first"))]
    );
    assert_eq!(
        values(invoke_session(
            &mut second,
            &cold,
            &InvocationControl::default()
        )),
        [Value::CharArray(CharArray::new_row("second"))]
    );
    let mutation = request(
        "setenv('RUNMAT_PORTABLE_CONTEXT','changed'); ans = getenv('RUNMAT_PORTABLE_CONTEXT');",
        None,
        1,
    );
    assert_eq!(
        values(invoke_session(
            &mut first,
            &mutation,
            &InvocationControl::default()
        )),
        [Value::CharArray(CharArray::new_row("changed"))]
    );
    assert_eq!(
        second.environment_snapshot(),
        [(name.into(), "second".into())]
    );
    let missing = request("ans = getenv('HOME');", None, 1);
    assert_eq!(
        values(invoke_session(
            &mut first,
            &missing,
            &InvocationControl::default()
        )),
        [Value::CharArray(CharArray::new_row(""))]
    );
    assert_eq!(std::env::var_os(name), host);
}

#[test]
#[cfg(unix)]
fn native_subprocess_consumes_the_session_environment_without_host_mutation() {
    let name = "RUNMAT_PORTABLE_CONTEXT";
    let host = std::env::var_os(name);
    let mut session =
        RunMatSession::with_frozen_environment(false, false, vec![(name.into(), "frozen".into())])
            .unwrap();
    let original = request("setenv('RUNMAT_PORTABLE_CONTEXT','changed'); [status,text] = system('printf \"$RUNMAT_PORTABLE_CONTEXT\"'); ans = text;", None, 1);
    assert_eq!(
        values(invoke_session(
            &mut session,
            &original,
            &InvocationControl::default()
        )),
        [Value::CharArray(CharArray::new_row("changed"))]
    );
    assert_eq!(std::env::var_os(name), host);
}
