//! A Dataset lease follows one Core session across separate REPL evaluations.

use crate::{execute_text_request_for_testing, RunMatSession};
use runmat_filesystem::{self as fs, FsProvider};
use std::sync::Arc;

#[test]
fn dataset_receiver_retains_session_and_original_filesystem_across_evaluations() {
    runmat_gc::gc_test_context(|| {
        let _serial = fs::provider_override_lock();
        let selected_root = std::path::Path::new("/selected-data-workspace");
        let replacement_root = std::path::Path::new("/replacement-data-workspace");
        let original = Arc::new(fs::MemoryFsProvider::with_current_dir(selected_root));
        let _original = fs::replace_provider(original.clone() as Arc<dyn FsProvider>);
        let mut session = RunMatSession::with_options(false, false).expect("Core session");
        let first = execute_text_request_for_testing(
            &mut session,
            "schema = struct(); schema.arrays = struct(); ds = data.create('session.data', schema);",
        )
        .expect("first evaluation");
        assert!(first.error.is_none(), "{:?}", first.error);

        let replacement = Arc::new(fs::MemoryFsProvider::with_current_dir(replacement_root));
        let _replacement = fs::replace_provider(replacement.clone() as Arc<dyn FsProvider>);
        let second = execute_text_request_for_testing(
            &mut session,
            "assert(strcmp(Dataset.path(ds), 'session.data')); Dataset.set_attr(ds, 'discarded', 'kept'); assert(Dataset.set_attr(ds, 'owner', 'original')); assert(strcmp(Dataset.get_attr(ds, 'owner'), 'original')); assert(strcmp(Dataset.get_attr(ds, 'discarded'), 'kept'));",
        )
        .expect("later evaluation through same session");
        assert!(second.error.is_none(), "{:?}", second.error);
        assert!(original
            .read_project_path(selected_root.join("session.data/manifest.json"))
            .is_ok());
        assert!(replacement
            .read_project_path(replacement_root.join("session.data/manifest.json"))
            .is_err());
    });
}
