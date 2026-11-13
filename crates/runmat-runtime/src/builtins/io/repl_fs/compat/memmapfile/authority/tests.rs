use super::*;
use crate::execution::RuntimeExecutionService;
use runmat_value::MemmapFormatPresentationState;
use std::rc::Rc;

fn opened(runtime: &RuntimeContext, suffix: &str) -> MemmapReceiverState {
    let root = std::path::PathBuf::from("/mapped-lifecycle");
    let provider = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(&root));
    let path = root.join(suffix);
    provider.write_project_path(&path, b"data").unwrap();
    let backing = futures::executor::block_on(runmat_filesystem::capture_mapped_backing_on(
        provider, path, false,
    ))
    .unwrap()
    .unwrap();
    MemmapReceiverState {
        authority: MemmapReceiverAuthority::Opened(issue(Some(runtime), backing).unwrap()),
        offset: 0,
        repeat: Some(4),
        writable: false,
        fields: Vec::new(),
        format_presentation: MemmapFormatPresentationState::TypeName,
    }
}

#[test]
fn final_session_clone_releases_backing_even_when_receiver_remains_live() {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let clone = runtime.clone();
    let receiver = opened(&runtime, "first.bin");
    let backing = resolve(Some(&runtime), &receiver).unwrap();
    let weak = Arc::downgrade(&backing);
    drop(backing);
    drop(runtime);
    assert!(
        weak.upgrade().is_some(),
        "remaining context owns the session"
    );
    drop(clone);
    assert!(
        weak.upgrade().is_none(),
        "receiver must not retain ended-session backing"
    );
    assert!(resolve(None, &receiver).is_err());
}

#[test]
fn cleanup_is_registered_once_and_retires_only_its_own_session() {
    let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let other = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let first_identity = first.local_identity();
    let first_receiver = opened(&first, "first.bin");
    let second_receiver = opened(&first, "second.bin");
    let other_receiver = opened(&other, "other.bin");
    assert_eq!(first.state().resource_cleanup.0.borrow().len(), 1);
    let first_backing = Arc::downgrade(&resolve(Some(&first), &first_receiver).unwrap());
    let second_backing = Arc::downgrade(&resolve(Some(&first), &second_receiver).unwrap());
    drop(first);
    assert!(first_backing.upgrade().is_none());
    assert!(second_backing.upgrade().is_none());
    assert!(!registry()
        .lock()
        .unwrap()
        .cleanup_registered
        .contains(&first_identity));
    assert!(resolve(Some(&other), &other_receiver).is_ok());
    assert!(resolve(Some(&other), &first_receiver).is_err());
}
