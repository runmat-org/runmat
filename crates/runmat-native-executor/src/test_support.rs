//! Process isolation for tests that require a completed global collection.

/// Run the exact test in a fresh process, returning true in the parent.
///
/// The collector defers collection while another thread owns roots. Its test
/// mutex does not exclude executor tests that create roots without that mutex.
/// Reclamation tests therefore need a private heap, not suite-wide serialization.
pub(crate) fn run_isolated_gc_test(qualified: &str) -> bool {
    const CHILD: &str = "RUNMAT_NATIVE_EXECUTOR_GC_TEST";
    let test = qualified
        .strip_prefix(concat!(env!("CARGO_CRATE_NAME"), "::"))
        .expect("test belongs to this crate");
    if std::env::var_os(CHILD).as_deref() == Some(std::ffi::OsStr::new(test)) {
        return false;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([test, "--exact", "--test-threads=1"])
        .env(CHILD, test)
        .output()
        .expect("run isolated GC test");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed; 0 failed"),
        "isolated GC test {test} {}:\n{stdout}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    true
}
