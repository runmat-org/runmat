use super::*;
use runmat_filesystem::FsProvider;
use std::sync::Arc;

#[test]
fn diary_sink_survives_repeated_runs_without_redirecting_after_provider_swap() {
    let _lock = runmat_filesystem::provider_override_lock();
    let first = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/first",
    ));
    let first_guard = runmat_filesystem::replace_provider(first.clone());
    runmat_filesystem::set_current_dir("/first").expect("first fixture directory");
    let mut session = RunMatSession::with_options(false, false).expect("session");
    execute_text_request(&mut session, "diary session.log; disp('first');").expect("first run");

    let second = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/second",
    ));
    let second_guard = runmat_filesystem::replace_provider(second.clone());
    runmat_filesystem::set_current_dir("/second").expect("second fixture directory");
    execute_text_request(&mut session, "disp('second'); diary off;").expect("second run");

    let original = block_on(first.read(Path::new("/first/session.log"))).expect("original log");
    let text = String::from_utf8(original).expect("UTF-8 diary");
    assert!(text.contains("first\n"), "diary text was {text:?}");
    assert!(text.contains("second\n"), "diary text was {text:?}");
    assert!(block_on(second.read(Path::new("/second/session.log"))).is_err());
    drop(second_guard);
    drop(first_guard);
}

#[test]
fn two_sessions_keep_distinct_open_sinks_across_interleaved_runs() {
    let _lock = runmat_filesystem::provider_override_lock();
    let first = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/first",
    ));
    let first_guard = runmat_filesystem::replace_provider(first.clone());
    runmat_filesystem::set_current_dir("/first").expect("first fixture directory");
    let mut first_session = RunMatSession::with_options(false, false).expect("first session");
    let mut second_session = RunMatSession::with_options(false, false).expect("second session");
    execute_text_request(&mut first_session, "diary first.log; disp('A1');")
        .expect("first session run");

    let second = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/second",
    ));
    let second_guard = runmat_filesystem::replace_provider(second.clone());
    runmat_filesystem::set_current_dir("/second").expect("second fixture directory");
    execute_text_request(&mut second_session, "diary second.log; disp('B1');")
        .expect("second session run");
    execute_text_request(&mut first_session, "disp('A2'); diary off;")
        .expect("first session resumes");
    execute_text_request(&mut second_session, "disp('B2'); diary off;")
        .expect("second session resumes");

    let a = String::from_utf8(
        block_on(first.read(Path::new("/first/first.log"))).expect("first diary"),
    )
    .expect("first UTF-8");
    let b = String::from_utf8(
        block_on(second.read(Path::new("/second/second.log"))).expect("second diary"),
    )
    .expect("second UTF-8");
    assert!(a.contains("A1\n") && a.contains("A2\n"), "{a:?}");
    assert!(!a.contains("B1") && !a.contains("B2"), "{a:?}");
    assert!(b.contains("B1\n") && b.contains("B2\n"), "{b:?}");
    assert!(!b.contains("A1") && !b.contains("A2"), "{b:?}");
    drop(second_guard);
    drop(first_guard);
}
