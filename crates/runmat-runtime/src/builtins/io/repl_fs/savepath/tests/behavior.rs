use runmat_value::Value;
use tempfile::tempdir;

#[test]
fn explicit_file_preserves_order_and_escapes_source_text() {
    let _lock = super::support::lock();
    let directory = tempdir().expect("directory");
    let target = directory.path().join("pathdef.m");
    let path = format!(
        "alpha{}o'neil",
        crate::builtins::common::path_state::PATH_LIST_SEPARATOR
    );
    let _path = super::support::set_path(&path);
    let value = super::support::run(
        vec![Value::String(target.to_string_lossy().into_owned())],
        1,
        false,
    )
    .expect("savepath");
    assert_eq!(super::support::status(&value), 0.0);
    let contents = std::fs::read_to_string(target).expect("pathdef");
    assert!(contents.contains("%   alpha"));
    assert!(contents.contains("%   o'neil"));
    assert!(contents.contains("o''neil"));
}

#[test]
fn default_target_honors_environment_override() {
    let _lock = super::support::lock();
    let directory = tempdir().expect("directory");
    let target = directory.path().join("default.m");
    let _environment = super::support::EnvironmentGuard::set(&target);
    let _path = super::support::set_path("saved");
    let value = super::support::run(Vec::new(), 1, false).expect("savepath");
    assert_eq!(super::support::status(&value), 0.0);
    assert!(target.exists());
}

#[test]
fn zero_outputs_still_performs_the_write() {
    let _lock = super::support::lock();
    let directory = tempdir().expect("directory");
    let target = directory.path().join("zero.m");
    let value = super::support::run(
        vec![Value::String(target.to_string_lossy().into_owned())],
        0,
        false,
    )
    .expect("savepath");
    assert!(matches!(value, Value::OutputList(ref outputs) if outputs.is_empty()));
    assert!(target.exists());
}

#[test]
fn direct_call_without_output_context_returns_bare_status() {
    let _lock = super::support::lock();
    let directory = tempdir().expect("directory");
    let target = directory.path().join("direct.m");
    let value = super::support::run_without_output_context(
        vec![Value::String(target.to_string_lossy().into_owned())],
        false,
    )
    .expect("savepath");
    assert!(matches!(value, Value::Num(0.0)));
    assert!(target.exists());
}

#[test]
fn runmat_directory_target_and_diagnostic_outputs_are_preserved() {
    let _lock = super::support::lock();
    let directory = tempdir().expect("directory");
    let profile = directory.path().join("profile");
    std::fs::create_dir(&profile).expect("profile");
    let value = super::support::run(
        vec![Value::String(profile.to_string_lossy().into_owned())],
        3,
        true,
    )
    .expect("savepath");
    let Value::OutputList(outputs) = value else {
        panic!("outputs");
    };
    assert!(matches!(outputs[0], Value::Num(0.0)));
    assert!(matches!(outputs[1], Value::CharArray(ref chars) if chars.cols == 0));
    assert!(matches!(outputs[2], Value::CharArray(ref chars) if chars.cols == 0));
    assert!(profile
        .join(super::super::target::default_filename())
        .exists());
}
