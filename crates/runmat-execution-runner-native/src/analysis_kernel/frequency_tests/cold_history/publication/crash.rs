//! Abrupt process loss after real preparation, followed by an authentic cold retry.
use super::*;
use std::io::{BufRead, Read, Write};
use std::process::{Command, Stdio};

const ORDER_ENV: &str = "RUNMAT_TEST_FREQUENCY_PUBLICATION_ORDER";
const STOP_ENV: &str = "RUNMAT_TEST_FREQUENCY_PUBLICATION_STOP_PRECOMMIT";
const READY: &str = "ORIGINAL_FREQUENCY_PREPARED_FOR_TERMINATION";

pub(super) fn selected(order: ElementOrder) -> bool {
    std::env::var(ORDER_ENV).map_or(true, |selected| selected == format!("{order:?}"))
}

pub(super) fn wait_for_termination() {
    if std::env::var_os(STOP_ENV).is_some() {
        // Serial libtest writes its test-name prefix without a trailing newline.
        // Keep the readiness protocol on its own line, independent of that prefix.
        println!("\n{READY}");
        std::io::stdout().flush().unwrap();
        // The parent retains the pipe and kills this process. EOF is a fixture failure,
        // never a way to finish or publish the prepared study.
        let mut byte = [0];
        std::io::stdin().read_exact(&mut byte).unwrap();
        panic!("precommit producer must be terminated by its parent");
    }
}

#[test]
fn original_frequency_precommit_process_loss_is_invisible_and_retryable() {
    let directory = tempfile::tempdir().unwrap();
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let root = directory.path().join(format!("attempt-{order:?}"));
        let mut command = producer(&root, order);
        let mut child = command
            .env(STOP_ENV, "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (ready, received) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            let reached = std::io::BufReader::new(stdout)
                .lines()
                .any(|line| line.is_ok_and(|line| line.trim() == READY));
            let _ = ready.send(reached);
        });
        let reached = received.recv_timeout(std::time::Duration::from_secs(60));
        if reached != Ok(true) {
            let _ = child.kill();
            let status = child.wait().unwrap();
            reader.join().unwrap();
            panic!("producer did not reach real preparation: {reached:?}, {status}");
        }
        reader.join().unwrap();
        assert!(child.try_wait().unwrap().is_none());
        child.kill().unwrap();
        assert!(!child.wait().unwrap().success());
        let original = root.join(format!("{order:?}"));
        assert!(!original.join("expected.json").exists());
        let cold = FilesystemAnalysisArtifactStore::new(original.join("runs"));
        assert!(cold.list_run_metadata().unwrap().is_empty());
        // Reuse the abandoned artifact root and unchanged identity, numerical controls,
        // and host limits; do not delete or repair anything before the authentic retry.
        let retry = producer(&root, order).output().unwrap();
        assert!(
            retry.status.success(),
            "retry after process loss failed:\n{}\n{}",
            String::from_utf8_lossy(&retry.stdout),
            String::from_utf8_lossy(&retry.stderr)
        );
        verify_cold(&original);
    }
}

fn producer(root: &std::path::Path, order: ElementOrder) -> Command {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([PRODUCER, "--exact", "--ignored", "--nocapture"])
        .env("RUST_TEST_THREADS", "1")
        .env(ROOT_ENV, root)
        .env(ORDER_ENV, format!("{order:?}"));
    command
}
