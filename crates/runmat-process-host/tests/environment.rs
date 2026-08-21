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

#[tokio::test]
async fn platform_runtime_policy_preserves_loader_state_without_application_state() {
    #[cfg(target_os = "linux")]
    let loader_name = "LD_LIBRARY_PATH";
    #[cfg(target_os = "macos")]
    let loader_name = "DYLD_LIBRARY_PATH";
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    return;

    let Some(expected) = std::env::var_os(loader_name) else {
        // Developer machines with no dynamic-loader override still exercise
        // the allowlist itself in the crate unit tests. CI and native SDK
        // builds set this variable and cover the propagation boundary here.
        return;
    };
    let mut spec = HostCommand::new("/usr/bin/env");
    spec.environment_policy = EnvironmentPolicy::Allow(
        runmat_process_host::environment::EnvironmentAllowlist::platform_runtime(),
    );
    let mut child = spec.spawn().await.unwrap();
    let mut stdio = child.take_stdio().unwrap();
    let mut output = String::new();
    stdio.stdout.read_to_string(&mut output).await.unwrap();
    let exit = child.wait().await.unwrap();
    assert!(exit.success);

    let expected = format!("{loader_name}={}", expected.to_string_lossy());
    assert!(output.lines().any(|line| line == expected), "{output}");
    assert!(!output.lines().any(|line| line.starts_with("HOME=")));
}
