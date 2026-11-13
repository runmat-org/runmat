//! Sealed, session-scoped ownership for a `matfile` receiver. Public object
//! properties are presentation and never authorize filesystem access.

use super::MatFsPath;
use crate::context::{RuntimeContext, RuntimeContextLocalId};
use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
use runmat_value::{
    ForeignRef, ForeignResourceKey, ForeignResourceRelease, ObjectInstance, Value, WeakForeignRef,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use super::super::matfile::{
    matfile_error, matfile_error_with_source, MATFILE_CLASS, PROPERTIES_FIELD, SOURCE_FIELD,
    WRITABLE_FIELD,
};
use runmat_builtins::MATFILE_ERROR_IO;

pub(crate) struct AuthorizedMatFile {
    pub(crate) source: Arc<MatFsPath>,
    pub(crate) writable: bool,
}

struct Record {
    source: Arc<MatFsPath>,
    public_path: String,
    writable: bool,
    lease: WeakForeignRef,
    issuer: IssuerScope,
}

enum IssuerScope {
    Session {
        identity: RuntimeContextLocalId,
        lease: Weak<()>,
    },
    Process,
}

impl IssuerScope {
    fn from_runtime(runtime: Option<&RuntimeContext>) -> Self {
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

const HOST: &str = "runmat.matfile.authority";
const GENERATION: u64 = 1;
static REGISTRY: OnceLock<Arc<Mutex<State>>> = OnceLock::new();

fn registry() -> &'static Arc<Mutex<State>> {
    REGISTRY.get_or_init(|| Arc::new(Mutex::new(State::default())))
}

struct Release(Weak<Mutex<State>>);

impl std::fmt::Debug for Release {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MatFileAuthorityRelease")
    }
}

impl ForeignResourceRelease for Release {
    fn release(&self, key: &ForeignResourceKey) {
        if key.host_identity != HOST || key.generation != GENERATION {
            return;
        }
        if let Some(state) = self.0.upgrade() {
            if let Ok(mut state) = state.lock() {
                state.records.remove(&key.handle);
            }
        }
    }
}

pub(crate) fn issue_matfile(
    runtime: Option<&RuntimeContext>,
    source: Arc<MatFsPath>,
    public_path: String,
    writable: bool,
) -> crate::BuiltinResult<ForeignRef> {
    let issuer = IssuerScope::from_runtime(runtime);
    let registry = registry();
    let mut state = registry
        .lock()
        .map_err(|_| matfile_error("matfile authority registry is unavailable"))?;
    let handle = state
        .next_handle
        .checked_add(1)
        .ok_or_else(|| matfile_error("matfile authority handle space is exhausted"))?;
    state.next_handle = handle;
    let reference = ForeignRef::managed(
        ForeignResourceKey {
            host_identity: HOST.into(),
            handle,
            generation: GENERATION,
        },
        ForeignTypeIdentity {
            family: "runmat.matfile".into(),
            name: MATFILE_CLASS.display_name().into(),
            version: 1,
        },
        ForeignOwnership::Shared,
        ForeignAffinity::OriginProcess,
        issuer.lifetime(),
        Arc::new(Release(Arc::downgrade(registry))),
    );
    let lease = reference
        .downgrade()
        .expect("new MAT authority has a managed lease");
    state.records.insert(
        handle,
        Record {
            source,
            public_path,
            writable,
            lease,
            issuer,
        },
    );
    Ok(reference)
}

pub(crate) fn resolve_matfile(
    runtime: Option<&RuntimeContext>,
    object: &ObjectInstance,
) -> crate::BuiltinResult<AuthorizedMatFile> {
    if !object.is_class(MATFILE_CLASS) {
        return Err(matfile_error("matfile receiver has the wrong class"));
    }
    let reference = object
        .matfile_authority()
        .ok_or_else(|| matfile_error("matfile receiver has no sealed authority"))?;
    if !reference.is_managed()
        || reference.host_identity != HOST
        || reference.generation != GENERATION
    {
        return Err(matfile_error(
            "matfile receiver authority is foreign or detached",
        ));
    }
    let state = registry()
        .lock()
        .map_err(|_| matfile_error("matfile authority registry is unavailable"))?;
    let record = state
        .records
        .get(&reference.handle)
        .ok_or_else(|| matfile_error("matfile receiver authority is stale"))?;
    if !record.issuer.admits(runtime) {
        return Err(matfile_error(
            "matfile receiver belongs to another or ended runtime session",
        ));
    }
    let issued = record
        .lease
        .upgrade()
        .ok_or_else(|| matfile_error("matfile receiver lease has expired"))?;
    if !issued.shares_managed_lease(reference)
        || issued.resource_key() != reference.resource_key()
        || reference.type_identity.family != "runmat.matfile"
        || reference.type_identity.name != MATFILE_CLASS.display_name()
        || reference.type_identity.version != 1
        || reference.ownership != ForeignOwnership::Shared
        || reference.affinity != ForeignAffinity::OriginProcess
        || reference.lifetime != record.issuer.lifetime()
    {
        return Err(matfile_error(
            "matfile receiver authority does not belong to this registry",
        ));
    }
    let expected_source = Value::string(record.public_path.clone());
    let Some(Value::Struct(properties)) = object.properties().get(PROPERTIES_FIELD) else {
        return Err(matfile_error(
            "matfile receiver presentation is missing Properties",
        ));
    };
    if properties.fields.get(SOURCE_FIELD) != Some(&expected_source)
        || properties.fields.get(WRITABLE_FIELD) != Some(&Value::Bool(record.writable))
    {
        return Err(matfile_error(
            "matfile receiver presentation does not match its sealed authority",
        ));
    }
    record.source.check_current().map_err(|err| {
        matfile_error_with_source(
            &MATFILE_ERROR_IO,
            format!("matfile filesystem authority is no longer current: {err}"),
            err,
        )
    })?;
    Ok(AuthorizedMatFile {
        source: record.source.clone(),
        writable: record.writable,
    })
}

#[cfg(test)]
#[path = "handle/tests.rs"]
mod tests;
