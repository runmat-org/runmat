//! Process isolation for reclamation tests that require a completed collection.

/// Run the exact test in a fresh process, returning true in the parent.
///
/// Pass `concat!(module_path!(), "::test_name")`. The collector defers collection
/// while another thread owns roots. The GC test mutex cannot exclude ordinary
/// executor tests, so reclamation assertions need a private heap rather than
/// suite-wide serialization or a retry of a potentially skipped collection.
pub fn run_isolated_gc_test(qualified: &str) -> bool {
    const CHILD: &str = "RUNMAT_ISOLATED_GC_TEST";
    let (_, test) = qualified
        .split_once("::")
        .expect("fully qualified reclamation test name");
    if std::env::var_os(CHILD).as_deref() == Some(std::ffi::OsStr::new(qualified)) {
        return false;
    }
    let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
        .args([test, "--exact", "--test-threads=1"])
        .env(CHILD, qualified)
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
