//! Sealed, clone-stable filesystem owner for Dataset family receiver values.

use super::data_error;
use crate::context::{RuntimeContext, RuntimeContextLocalId};
use crate::BuiltinResult;
use runmat_filesystem::FsProvider;
use runmat_types::{
    ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity, StaticClassIdentity,
};
use runmat_value::{
    ForeignRef, ForeignResourceKey, ForeignResourceRelease, ObjectInstance, Value, WeakForeignRef,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, Weak};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum HandleRole {
    Dataset,
    Array(String),
    Transaction(String),
}

impl HandleRole {
    fn class(&self) -> StaticClassIdentity {
        match self {
            Self::Dataset => runmat_builtins::DATASET_CLASS,
            Self::Array(_) => runmat_builtins::DATA_ARRAY_CLASS,
            Self::Transaction(_) => runmat_builtins::DATA_TRANSACTION_CLASS,
        }
    }
}

pub(crate) struct AuthorizedHandle {
    pub filesystem: Arc<dyn FsProvider>,
    pub root: PathBuf,
    pub public_path: String,
    pub role: HandleRole,
    reference: ForeignRef,
    issuer: IssuerScope,
}

impl AuthorizedHandle {
    pub fn reference(&self) -> &ForeignRef {
        &self.reference
    }
}

struct Record {
    filesystem: Arc<dyn FsProvider>,
    root: PathBuf,
    public_path: String,
    role: HandleRole,
    lease: WeakForeignRef,
    issuer: IssuerScope,
}

/// A session-issued handle never owns its session. Context-free embedding
/// issuance is explicitly process-local and remains live while its managed
/// ForeignRef is retained; it is never silently upgraded to session scope.
#[derive(Clone)]
enum IssuerScope {
    Session {
        identity: RuntimeContextLocalId,
        lease: Weak<()>,
    },
    Process,
}

impl IssuerScope {
    fn from_context(runtime: Option<&RuntimeContext>) -> Self {
        match runtime {
            Some(runtime) => Self::Session {
                identity: runtime.local_identity(),
                lease: runtime.session_lease(),
            },
            None => Self::Process,
        }
    }

    fn lifetime(&self) -> ForeignLifetime {
        match self {
            Self::Session { .. } => ForeignLifetime::Session,
            Self::Process => ForeignLifetime::Persistent,
        }
    }

    fn admits(&self, runtime: Option<&RuntimeContext>) -> bool {
        match self {
            Self::Process => true,
            Self::Session { identity, lease } => runtime.is_some_and(|runtime| {
                *identity == runtime.local_identity()
                    && lease.strong_count() != 0
                    && Weak::ptr_eq(lease, &runtime.session_lease())
            }),
        }
    }
}

#[derive(Default)]
struct State {
    next_handle: u64,
    records: BTreeMap<u64, Record>,
}

const HOST_IDENTITY: &str = "runmat.data.dataset-authority";
const GENERATION: u64 = 1;
static REGISTRY: OnceLock<Arc<Mutex<State>>> = OnceLock::new();

fn registry() -> &'static Arc<Mutex<State>> {
    REGISTRY.get_or_init(|| Arc::new(Mutex::new(State::default())))
}

struct Release {
    state: Weak<Mutex<State>>,
}

impl std::fmt::Debug for Release {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DatasetAuthorityRelease")
    }
}

impl ForeignResourceRelease for Release {
    fn release(&self, key: &ForeignResourceKey) {
        if key.host_identity != HOST_IDENTITY || key.generation != GENERATION {
            return;
        }
        let Some(state) = self.state.upgrade() else {
            return;
        };
        if let Ok(mut state) = state.lock() {
            state.records.remove(&key.handle);
        };
    }
}

pub(crate) fn issue_dataset(
    runtime: Option<&RuntimeContext>,
    filesystem: Arc<dyn FsProvider>,
    root: PathBuf,
    public_path: String,
) -> BuiltinResult<ForeignRef> {
    if !filesystem.path_namespace().is_absolute(&root) {
        return Err(data_error(
            "Dataset authority requires an absolute resolved root",
        ));
    }
    issue(
        filesystem,
        root,
        public_path,
        HandleRole::Dataset,
        IssuerScope::from_context(runtime),
    )
}

pub(crate) fn issue_child(
    parent: &AuthorizedHandle,
    role: HandleRole,
) -> BuiltinResult<ForeignRef> {
    if parent.role != HandleRole::Dataset {
        return Err(data_error(
            "Dataset child must derive from a Dataset authority",
        ));
    }
    if matches!(role, HandleRole::Dataset) {
        return Err(data_error(
            "Dataset child requires an array or transaction role",
        ));
    }
    issue(
        parent.filesystem.clone(),
        parent.root.clone(),
        parent.public_path.clone(),
        role,
        parent.issuer.clone(),
    )
}

pub(crate) fn renew_dataset(parent: &AuthorizedHandle) -> BuiltinResult<ForeignRef> {
    if parent.role != HandleRole::Dataset {
        return Err(data_error("Dataset refresh requires a Dataset authority"));
    }
    issue(
        parent.filesystem.clone(),
        parent.root.clone(),
        parent.public_path.clone(),
        HandleRole::Dataset,
        parent.issuer.clone(),
    )
}

fn issue(
    filesystem: Arc<dyn FsProvider>,
    root: PathBuf,
    public_path: String,
    role: HandleRole,
    issuer: IssuerScope,
) -> BuiltinResult<ForeignRef> {
    let state = registry();
    let mut records = state
        .lock()
        .map_err(|_| data_error("Dataset authority registry is unavailable"))?;
    let handle = records
        .next_handle
        .checked_add(1)
        .ok_or_else(|| data_error("Dataset authority handle space is exhausted"))?;
    records.next_handle = handle;
    let reference = ForeignRef::managed(
        ForeignResourceKey {
            host_identity: HOST_IDENTITY.into(),
            handle,
            generation: GENERATION,
        },
        ForeignTypeIdentity {
            family: "runmat.data".into(),
            name: role.class().display_name().into(),
            version: 1,
        },
        ForeignOwnership::Shared,
        ForeignAffinity::OriginProcess,
        issuer.lifetime(),
        Arc::new(Release {
            state: Arc::downgrade(state),
        }),
    );
    let lease = reference
        .downgrade()
        .expect("newly issued authority has a managed lease");
    records.records.insert(
        handle,
        Record {
            filesystem,
            root,
            public_path,
            role,
            lease,
            issuer,
        },
    );
    Ok(reference)
}

pub(crate) fn resolve_object(
    runtime: Option<&RuntimeContext>,
    value: &Value,
    expected: HandleRole,
) -> BuiltinResult<AuthorizedHandle> {
    let resolved = resolve_any_object(runtime, value)?;
    if resolved.role != expected {
        return Err(data_error(
            "Dataset family receiver has the wrong sealed role",
        ));
    }
    Ok(resolved)
}

pub(crate) fn resolve_array(
    runtime: Option<&RuntimeContext>,
    value: &Value,
) -> BuiltinResult<(AuthorizedHandle, String)> {
    let resolved = resolve_any_object(runtime, value)?;
    let HandleRole::Array(name) = &resolved.role else {
        return Err(data_error("DataArray receiver has the wrong sealed role"));
    };
    let name = name.clone();
    Ok((resolved, name))
}

pub(crate) fn resolve_transaction(
    runtime: Option<&RuntimeContext>,
    value: &Value,
) -> BuiltinResult<(AuthorizedHandle, String)> {
    let resolved = resolve_any_object(runtime, value)?;
    let HandleRole::Transaction(tx_id) = &resolved.role else {
        return Err(data_error(
            "DataTransaction receiver has the wrong sealed role",
        ));
    };
    let tx_id = tx_id.clone();
    Ok((resolved, tx_id))
}

fn resolve_any_object(
    runtime: Option<&RuntimeContext>,
    value: &Value,
) -> BuiltinResult<AuthorizedHandle> {
    let Value::Object(object) = value else {
        return Err(data_error("expected object receiver for Dataset family"));
    };
    let reference = object
        .dataset_authority()
        .ok_or_else(|| data_error("Dataset family receiver has no managed authority"))?;
    let resolved = resolve(runtime, reference)?;
    if !object.is_class(resolved.role.class()) || !presentation_matches(object, &resolved) {
        return Err(data_error(
            "Dataset family receiver authority does not match its presentation",
        ));
    }
    Ok(resolved)
}

fn resolve(
    runtime: Option<&RuntimeContext>,
    reference: &ForeignRef,
) -> BuiltinResult<AuthorizedHandle> {
    if !reference.is_managed()
        || reference.host_identity != HOST_IDENTITY
        || reference.generation != GENERATION
    {
        return Err(data_error("Dataset authority is foreign or detached"));
    }
    let records = registry()
        .lock()
        .map_err(|_| data_error("Dataset authority registry is unavailable"))?;
    let record = records
        .records
        .get(&reference.handle)
        .ok_or_else(|| data_error("Dataset authority is stale"))?;
    if !record.issuer.admits(runtime) {
        return Err(data_error(
            "Dataset authority belongs to another or ended session",
        ));
    }
    let issued = record
        .lease
        .upgrade()
        .ok_or_else(|| data_error("Dataset authority lease has expired"))?;
    if !issued.shares_managed_lease(reference)
        || issued.resource_key() != reference.resource_key()
        || reference.type_identity.family != "runmat.data"
        || reference.type_identity.name != record.role.class().display_name()
        || reference.type_identity.version != 1
        || reference.ownership != ForeignOwnership::Shared
        || reference.affinity != ForeignAffinity::OriginProcess
        || reference.lifetime != record.issuer.lifetime()
    {
        return Err(data_error(
            "Dataset authority does not belong to this registry",
        ));
    }
    Ok(AuthorizedHandle {
        filesystem: record.filesystem.clone(),
        root: record.root.clone(),
        public_path: record.public_path.clone(),
        role: record.role.clone(),
        reference: reference.clone(),
        issuer: record.issuer.clone(),
    })
}

#[cfg(test)]
#[path = "authority/tests.rs"]
mod tests;

fn presentation_matches(object: &ObjectInstance, handle: &AuthorizedHandle) -> bool {
    if object.properties().get("__data_path") != Some(&Value::string(handle.public_path.clone())) {
        return false;
    }
    match &handle.role {
        HandleRole::Dataset => true,
        HandleRole::Array(name) => {
            object.properties().get("__array_name") == Some(&Value::string(name.clone()))
        }
        HandleRole::Transaction(tx_id) => {
            object.properties().get("__tx_id") == Some(&Value::string(tx_id.clone()))
        }
    }
}
