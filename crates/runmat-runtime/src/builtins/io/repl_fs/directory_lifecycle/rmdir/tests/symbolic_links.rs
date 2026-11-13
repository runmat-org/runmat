#[cfg(unix)]
mod unix {
    use std::fs;
    use std::os::unix::fs::symlink;

    use runmat_value::Value;
    use tempfile::tempdir;

    use super::super::{evaluate, lock};

    #[test]
    fn default_mode_removes_the_link_and_preserves_its_target() {
        let _lock = lock();
        let temp = tempdir().expect("temporary directory");
        let target = temp.path().join("target");
        let link = temp.path().join("link");
        fs::create_dir(&target).expect("seed target");
        symlink(&target, &link).expect("seed link");

        let outcome = evaluate(vec![Value::from(link.to_string_lossy().to_string())])
            .expect("rmdir evaluates");

        assert!(outcome.status());
        assert!(!link.exists());
        assert!(target.is_dir());
    }

    #[test]
    fn resolution_mode_removes_the_target_and_preserves_the_link_entry() {
        let _lock = lock();
        let temp = tempdir().expect("temporary directory");
        let target = temp.path().join("target");
        let link = temp.path().join("link");
        fs::create_dir(&target).expect("seed target");
        symlink(&target, &link).expect("seed link");

        let outcome = evaluate(vec![
            Value::from(link.to_string_lossy().to_string()),
            Value::from("ResolveSymbolicLinks"),
            Value::Bool(true),
        ])
        .expect("rmdir evaluates");

        assert!(outcome.status());
        assert!(fs::symlink_metadata(&link).is_ok());
        assert!(!target.exists());
    }
}
