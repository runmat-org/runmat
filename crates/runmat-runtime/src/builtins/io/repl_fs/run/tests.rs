use super::*;
use crate::builtins::common::path_state::{current_path_string, set_path_string};
use futures::executor::block_on;
use runmat_value::CharArray;
use std::env;
use std::path::{Path, PathBuf};

struct CwdGuard {
    original: PathBuf,
}

struct PathStateGuard {
    previous: String,
}

impl Drop for PathStateGuard {
    fn drop(&mut self) {
        set_path_string(&self.previous).expect("test search path update");
    }
}

impl Drop for CwdGuard {
    fn drop(&mut self) {
        let _ = env::set_current_dir(&self.original);
    }
}

fn push_cwd(path: &Path) -> CwdGuard {
    let original = env::current_dir().expect("current dir");
    env::set_current_dir(path).expect("set current dir");
    CwdGuard { original }
}

fn push_path_state(path: &Path) -> PathStateGuard {
    let previous = current_path_string();
    let path = path.to_string_lossy().to_string();
    set_path_string(&path).expect("test search path update");
    PathStateGuard { previous }
}

#[test]
fn run_requires_explicit_execution_context() {
    let err = block_on(run_builtin_registered(
        None,
        None,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("run execution requires an explicit context");
    assert_eq!(err.identifier(), Some("RunMat:run:RequiresVm"));
}

#[test]
fn resolves_script_from_current_directory_with_implicit_m_extension() {
    let _lock = runmat_filesystem::provider_override_lock();
    let temp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(temp.path().join("worker.m"), "generated = 41;\n").expect("write script");
    let _cwd = push_cwd(temp.path());

    let source = block_on(resolve_run_source(&Value::from("worker"))).expect("resolve source");
    assert!(source.display_name.ends_with("worker.m"));
    assert_eq!(source.text, "generated = 41;\n");
}

#[test]
fn resolves_bare_m_filename_from_search_path() {
    let _lock = runmat_filesystem::provider_override_lock();
    let temp = tempfile::TempDir::new().expect("tempdir");
    let scripts = temp.path().join("scripts");
    std::fs::create_dir_all(&scripts).expect("create scripts dir");
    std::fs::write(scripts.join("path_worker.m"), "path_value = 17;\n").expect("write script");
    let _cwd = push_cwd(temp.path());
    let _path = push_path_state(&scripts);

    let source =
        block_on(resolve_run_source(&Value::from("path_worker.m"))).expect("resolve source");
    assert!(source.display_name.ends_with("path_worker.m"));
    assert_eq!(source.text, "path_value = 17;\n");
}

#[test]
fn resolves_script_from_char_row_path() {
    let _lock = runmat_filesystem::provider_override_lock();
    let temp = tempfile::TempDir::new().expect("tempdir");
    let path = temp.path().join("direct_script.m");
    std::fs::write(&path, "x = 1;\n").expect("write script");

    let value = Value::CharArray(CharArray::new_row(path.to_string_lossy().as_ref()));
    let source = block_on(resolve_run_source(&value)).expect("resolve source");
    assert_eq!(source.text, "x = 1;\n");
    assert!(source.path.ends_with("direct_script.m"));
}

#[test]
fn missing_script_reports_stable_identifier() {
    let _lock = runmat_filesystem::provider_override_lock();
    let temp = tempfile::TempDir::new().expect("tempdir");
    let _cwd = push_cwd(temp.path());

    let err = block_on(resolve_run_source(&Value::from("missing_script")))
        .expect_err("missing script should fail");
    assert_eq!(err.identifier(), Some("RunMat:run:FileNotFound"));
}

#[test]
fn invalid_script_argument_reports_stable_identifier() {
    let err =
        block_on(resolve_run_source(&Value::Num(1.0))).expect_err("numeric script should fail");
    assert_eq!(err.identifier(), Some("RunMat:run:InvalidScriptArgument"));
}

#[test]
fn rejects_numeric_residency_and_character_pages_before_provider_access() {
    let resident = Value::GpuTensor(
        crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1])
            .handle,
    );
    let pages = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap(),
    );
    for value in [resident, pages] {
        let error = block_on(resolve_run_source(&value)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:run:InvalidScriptArgument"));
    }
}

#[test]
fn trailing_singleton_character_row_resolves_normally() {
    let _lock = runmat_filesystem::provider_override_lock();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("trailing.m");
    std::fs::write(&path, "value = 4;").unwrap();
    let chars = path.to_string_lossy().chars().collect::<Vec<_>>();
    let shape = vec![1, path.to_string_lossy().encode_utf16().count(), 1];
    let source = block_on(resolve_run_source(&Value::CharArray(
        CharArray::from_unicode_scalars_with_shape(chars, shape).unwrap(),
    )))
    .unwrap();
    assert_eq!(source.text, "value = 4;");
}
