#![cfg(unix)]

use runmat_process_host::{HostCommand, ResourceLimits};

#[tokio::test]
async fn supported_resource_limits_are_applied_before_exec() {
    let limits = vec![ResourceLimits {
        cpu_seconds: Some(60),
        ..ResourceLimits::default()
    }];
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let limits = {
        let mut limits = limits;
        limits.extend([
            ResourceLimits {
                memory_bytes: Some(8 * 1024 * 1024 * 1024),
                ..ResourceLimits::default()
            },
            ResourceLimits {
                process_count: Some(64),
                ..ResourceLimits::default()
            },
        ]);
        limits
    };
    for limits in limits {
        let mut command = HostCommand::new("/usr/bin/true");
        command.resource_limits = limits;
        let mut child = command.spawn().await.unwrap();
        assert!(child.wait().await.unwrap().success);
    }
}
