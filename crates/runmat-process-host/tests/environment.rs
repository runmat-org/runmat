#[cfg(unix)]
use std::collections::BTreeMap;

#[cfg(unix)]
use runmat_process_host::environment::EnvironmentPolicy;
#[cfg(unix)]
use runmat_process_host::HostCommand;
#[cfg(unix)]
use tokio::io::AsyncReadExt;

#[cfg(unix)]
#[tokio::test]
async fn clear_policy_exposes_only_explicit_environment() {
    let mut spec = HostCommand::new("/bin/sh");
    spec.arguments = vec![
        "-c".into(),
        "printf '%s:%s' \"$RUNMAT_VISIBLE\" \"${HOME-unset}\"".into(),
    ];
    spec.environment_policy = EnvironmentPolicy::Clear;
    spec.environment = BTreeMap::from([("RUNMAT_VISIBLE".into(), "yes".into())]);
    let mut child = spec.spawn().await.unwrap();
    let mut stdio = child.take_stdio().unwrap();
    let mut output = String::new();
    stdio.stdout.read_to_string(&mut output).await.unwrap();
    let exit = child.wait().await.unwrap();
    assert!(exit.success);
    assert_eq!(output, "yes:unset");
}
