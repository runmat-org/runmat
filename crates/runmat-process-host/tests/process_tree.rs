#[cfg(unix)]
use runmat_process_host::{is_process_alive, HostCommand, ProcessGroupPlacement};
#[cfg(unix)]
use runmat_process_host::{ProcessRetirementLease, RetirementBoundChild};
#[cfg(unix)]
use tokio::io::AsyncReadExt;

#[cfg(unix)]
#[tokio::test]
async fn completion_preserves_exit_status_and_retirement_is_idempotent() {
    struct Lease;
    impl ProcessRetirementLease for Lease {
        fn persist_unverified(&mut self, _: Option<u32>, _: &str) -> Result<(), String> {
            Ok(())
        }
    }
    let mut command = HostCommand::new("/bin/sh");
    command.arguments = vec!["-c".into(), "exit 7".into()];
    let mut child = RetirementBoundChild::new(command.spawn().await.unwrap(), Lease)
        .unwrap_or_else(|_| panic!("owned tree"));
    let observed = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if let Some(exit) = child.try_wait().unwrap() {
                break exit;
            }
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    // Repeated observation must not consume the waitable process identity.
    assert_eq!(child.try_wait().unwrap(), Some(observed));
    let exit = child.wait().await.unwrap();
    assert_eq!(observed, exit);
    assert_eq!(exit.code, Some(7));
    assert!(!exit.success);
    child.terminate_tree().await.unwrap();
    assert_eq!(child.wait().await.unwrap(), exit);
}

#[cfg(unix)]
#[tokio::test]
async fn terminate_kills_the_child_process_group() {
    for leader_exited in [false, true] {
        let mut spec = HostCommand::new("/bin/sh");
        let script = if leader_exited {
            "sleep 60 & echo $!; exit 0"
        } else {
            "sleep 60 & echo $!; wait"
        };
        spec.arguments = vec!["-c".into(), script.into()];
        let mut child = spec.spawn().await.unwrap();
        let mut stdio = child.take_stdio().unwrap();
        let mut pid_line = Vec::new();
        loop {
            let mut byte = [0];
            stdio.stdout.read_exact(&mut byte).await.unwrap();
            if byte[0] == b'\n' {
                break;
            }
            pid_line.push(byte[0]);
        }
        let descendant: i32 = String::from_utf8(pid_line).unwrap().parse().unwrap();
        if leader_exited {
            assert!(child.wait_and_retire_tree().await.unwrap().success);
        }
        child.terminate_tree().await.unwrap();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        while is_process_alive(descendant.try_into().unwrap()) {
            assert!(
                tokio::time::Instant::now() < deadline,
                "descendant process {descendant} survived process-tree termination (leader exited: {leader_exited})"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
}

#[cfg(windows)]
#[tokio::test]
async fn failed_taskkill_does_not_advertise_tree_retirement_from_absent_leader() {
    let unknown_process = u32::MAX;
    assert!(!runmat_process_host::is_process_alive(unknown_process));
    assert!(runmat_process_host::terminate_process_tree(unknown_process)
        .await
        .is_err());
}

#[cfg(unix)]
#[tokio::test]
async fn abandoned_retirement_bound_child_reaps_tree_before_releasing_lease() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct Lease(Arc<AtomicUsize>);
    impl ProcessRetirementLease for Lease {
        fn persist_unverified(&mut self, _: Option<u32>, _: &str) -> Result<(), String> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    let mut spec = HostCommand::new("/bin/sh");
    spec.arguments = vec!["-c".into(), "sleep 60 & echo $!; wait".into()];
    let persisted = Arc::new(AtomicUsize::new(0));
    let mut child =
        RetirementBoundChild::new(spec.spawn().await.unwrap(), Lease(persisted.clone()))
            .ok()
            .unwrap();
    let mut stdout = child.take_stdio().unwrap().stdout;
    let mut pid_line = Vec::new();
    loop {
        let mut byte = [0];
        stdout.read_exact(&mut byte).await.unwrap();
        if byte[0] == b'\n' {
            break;
        }
        pid_line.push(byte[0]);
    }
    let descendant: u32 = String::from_utf8(pid_line).unwrap().parse().unwrap();
    drop(child);
    assert_eq!(persisted.load(Ordering::SeqCst), 0);
    assert!(!is_process_alive(descendant));
}

#[cfg(unix)]
#[tokio::test]
async fn joined_helper_stays_in_ancestor_group_without_killing_its_sibling() {
    let mut spec = HostCommand::new("/bin/sleep");
    spec.arguments = vec!["60".into()];
    spec.process_group_placement = ProcessGroupPlacement::JoinAncestorGroup;
    if unsafe { libc::getpid() } != unsafe { libc::getpgrp() } {
        assert!(
            spec.spawn().await.is_err(),
            "uncontained join must fail closed"
        );
        // Exercise the admitted branch in an actual owned process group,
        // rather than returning with only the refusal branch covered.
        let mut owned = HostCommand::new(std::env::current_exe().unwrap());
        owned.arguments = vec![
            "--exact".into(),
            "joined_helper_stays_in_ancestor_group_without_killing_its_sibling".into(),
            "--test-threads=1".into(),
        ];
        let mut child = owned.spawn().await.unwrap();
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), child.wait()).await;
        child.terminate_tree().await.unwrap();
        assert!(outcome.unwrap().unwrap().success);
        return;
    }
    let mut first = spec.clone().spawn().await.unwrap();
    let mut sibling = spec.spawn().await.unwrap();
    let first_id = i32::try_from(first.id().unwrap()).unwrap();
    let sibling_id = i32::try_from(sibling.id().unwrap()).unwrap();
    let parent_group = unsafe { libc::getpgrp() };
    assert_eq!(unsafe { libc::getpgid(first_id) }, parent_group);
    assert_eq!(unsafe { libc::getpgid(sibling_id) }, parent_group);
    assert!(!first.owns_process_tree());
    first.terminate_tree().await.unwrap();
    assert!(sibling.try_wait().unwrap().is_none());
    sibling.terminate_tree().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn invalid_process_identities_reject_without_signalling_a_substitute_group() {
    for process_id in [0, u32::MAX] {
        let error = runmat_process_host::terminate_process_tree(process_id)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(!is_process_alive(process_id));
    }
}
