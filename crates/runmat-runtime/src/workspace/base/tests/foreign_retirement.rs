//! A final foreign lease may call back into workspace services while retiring.
//! Workspace publication must finish its mutation borrows before that callback.

use super::super::{
    BaseWorkspaceError, BaseWorkspaceOwner, BaseWorkspaceProjectionService, WorkspaceDelta,
};
use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
use runmat_value::{ForeignRef, ForeignResourceKey, ForeignResourceRelease, Value};
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

struct ProbeContext {
    owner: BaseWorkspaceOwner,
    service: Option<BaseWorkspaceProjectionService>,
}

thread_local! {
    static CURRENT_WORKSPACE: RefCell<Option<ProbeContext>> = const { RefCell::new(None) };
}

struct ProbeScope;

impl ProbeScope {
    fn enter(owner: &BaseWorkspaceOwner, service: Option<&BaseWorkspaceProjectionService>) -> Self {
        CURRENT_WORKSPACE.with(|current| {
            let mut current = current.borrow_mut();
            assert!(
                current.is_none(),
                "foreign release probe is already installed"
            );
            *current = Some(ProbeContext {
                owner: owner.clone(),
                service: service.cloned(),
            });
        });
        Self
    }
}

impl Drop for ProbeScope {
    fn drop(&mut self) {
        let retired = CURRENT_WORKSPACE.with(|current| current.borrow_mut().take());
        drop(retired);
    }
}

#[derive(Debug, PartialEq)]
enum WorkspaceRead {
    Available {
        canonical: Result<Option<f64>, ProbeReadError>,
        projected: Option<Result<Option<f64>, ProbeReadError>>,
    },
    ProbeUnavailable,
}

#[derive(Debug, PartialEq)]
enum ProbeReadError {
    Closed,
    BorrowConflict,
    Other(String),
}

impl From<BaseWorkspaceError> for ProbeReadError {
    fn from(error: BaseWorkspaceError) -> Self {
        match error {
            BaseWorkspaceError::Closed => Self::Closed,
            BaseWorkspaceError::BorrowConflict => Self::BorrowConflict,
            other => Self::Other(format!("{other:?}")),
        }
    }
}

fn numeric(value: Option<Value>) -> Result<Option<f64>, ProbeReadError> {
    match value {
        None => Ok(None),
        Some(Value::Num(number)) => Ok(Some(number)),
        Some(other) => Err(ProbeReadError::Other(format!(
            "unexpected workspace value: {other:?}"
        ))),
    }
}

#[derive(Debug, Default)]
struct ReentrantRelease {
    count: AtomicUsize,
    reads: Mutex<Vec<WorkspaceRead>>,
}

impl ReentrantRelease {
    fn take_reads(&self) -> Vec<WorkspaceRead> {
        assert_eq!(self.count.load(Ordering::SeqCst), 1);
        std::mem::take(&mut *self.reads.lock().unwrap())
    }
}

impl ForeignResourceRelease for ReentrantRelease {
    fn release(&self, _key: &ForeignResourceKey) {
        let read = CURRENT_WORKSPACE.with(|current| {
            let Ok(current) = current.try_borrow() else {
                return WorkspaceRead::ProbeUnavailable;
            };
            let Some(context) = current.as_ref() else {
                return WorkspaceRead::ProbeUnavailable;
            };
            WorkspaceRead::Available {
                canonical: context
                    .owner
                    .lookup("lease")
                    .map_err(ProbeReadError::from)
                    .and_then(numeric),
                projected: context.service.as_ref().map(|service| {
                    service
                        .lookup("lease")
                        .map_err(|error| ProbeReadError::Other(format!("{error:?}")))
                        .and_then(numeric)
                }),
            }
        });
        self.reads.lock().unwrap().push(read);
        self.count.fetch_add(1, Ordering::SeqCst);
    }
}

fn managed_foreign(release: Arc<ReentrantRelease>, handle: u64) -> Value {
    Value::Foreign(ForeignRef::managed(
        ForeignResourceKey {
            host_identity: "workspace-release-probe".into(),
            handle,
            generation: 1,
        },
        ForeignTypeIdentity {
            family: "native-ffi".into(),
            name: "test.resource".into(),
            version: 1,
        },
        ForeignOwnership::Shared,
        ForeignAffinity::AnyThread,
        ForeignLifetime::Session,
        release,
    ))
}

fn assignment(value: Value) -> WorkspaceDelta {
    let mut delta = WorkspaceDelta::new();
    delta.assign("lease", value).unwrap();
    delta
}

#[test]
fn replacement_releases_foreign_only_after_canonical_borrow_ends() {
    let owner = BaseWorkspaceOwner::new();
    let release = Arc::new(ReentrantRelease::default());
    owner
        .apply(assignment(managed_foreign(release.clone(), 1)))
        .unwrap();
    let _scope = ProbeScope::enter(&owner, None);

    owner.apply(assignment(Value::Num(7.0))).unwrap();
    assert_eq!(
        release.take_reads(),
        vec![WorkspaceRead::Available {
            canonical: Ok(Some(7.0)),
            projected: None,
        }]
    );
}

#[test]
fn clear_releases_foreign_only_after_canonical_and_service_borrows_end() {
    let owner = BaseWorkspaceOwner::new();
    let service = BaseWorkspaceProjectionService::new(owner.begin_projection().unwrap());
    let release = Arc::new(ReentrantRelease::default());
    service
        .assign("lease", managed_foreign(release.clone(), 2))
        .unwrap();
    service.publish().unwrap();
    let _scope = ProbeScope::enter(&owner, Some(&service));

    service.clear().unwrap();
    service.publish().unwrap();
    assert_eq!(
        release.take_reads(),
        vec![WorkspaceRead::Available {
            canonical: Ok(None),
            projected: Some(Ok(None)),
        }]
    );
}

#[test]
fn rejected_write_retires_foreign_after_closed_owner_borrow_ends() {
    let owner = BaseWorkspaceOwner::new();
    owner.close().unwrap();
    let release = Arc::new(ReentrantRelease::default());
    let _scope = ProbeScope::enter(&owner, None);

    owner
        .apply(assignment(managed_foreign(release.clone(), 3)))
        .expect_err("closed owner rejects the write before publication");
    let reads = release.take_reads();
    assert_eq!(reads.len(), 1);
    assert_eq!(
        reads[0],
        WorkspaceRead::Available {
            canonical: Err(ProbeReadError::Closed),
            projected: None,
        }
    );
}
