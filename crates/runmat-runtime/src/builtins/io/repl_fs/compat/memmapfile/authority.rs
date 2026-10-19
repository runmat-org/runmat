//! Sealed ownership for an opened memmap backing. Public object properties
//! describe controls but never authorize range I/O.

use crate::context::{RuntimeContext, RuntimeContextLocalId};
use runmat_filesystem::CapturedMappedBacking;
use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
use runmat_value::{
    ForeignRef, ForeignResourceKey, ForeignResourceRelease, MemmapReceiverAuthority,
    MemmapReceiverState, WeakForeignRef,
};
use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use super::format_error;
use runmat_builtins::MEMMAPFILE_CLASS;

pub(super) type SharedBacking = Arc<futures::lock::Mutex<CapturedMappedBacking>>;

struct Record {
    backing: SharedBacking,
    lease: WeakForeignRef,
    issuer: Issuer,
}

enum Issuer {
    Session {
        identity: RuntimeContextLocalId,
        lease: Weak<()>,
    },
    Process,
}

impl Issuer {
    fn new(runtime: Option<&RuntimeContext>) -> Self {
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
struct Registry {
    next_handle: u64,
    records: BTreeMap<u64, Record>,
    cleanup_registered: HashSet<RuntimeContextLocalId>,
}

const HOST: &str = "runmat.memmapfile.authority";
const GENERATION: u64 = 1;
static REGISTRY: OnceLock<Arc<Mutex<Registry>>> = OnceLock::new();

fn registry() -> &'static Arc<Mutex<Registry>> {
    REGISTRY.get_or_init(|| Arc::new(Mutex::new(Registry::default())))
}

fn retire_session(identity: RuntimeContextLocalId) {
    let removed = {
        let mut state = registry()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.cleanup_registered.remove(&identity);
        let handles: Vec<_> = state
            .records
            .iter()
            .filter_map(|(handle, record)| {
                matches!(&record.issuer, Issuer::Session { identity: owner, .. } if *owner == identity)
                    .then_some(*handle)
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|handle| state.records.remove(&handle))
            .collect::<Vec<_>>()
    };
    // Provider-owned resources may run cleanup on drop. Never retain the
    // registry lock while releasing those owners.
    drop(removed);
}

struct Release(Weak<Mutex<Registry>>);

impl std::fmt::Debug for Release {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("MemmapAuthorityRelease")
    }
}

impl ForeignResourceRelease for Release {
    fn release(&self, key: &ForeignResourceKey) {
        if key.host_identity != HOST || key.generation != GENERATION {
            return;
        }
        if let Some(registry) = self.0.upgrade() {
            let removed = registry
                .lock()
                .ok()
                .and_then(|mut state| state.records.remove(&key.handle));
            drop(removed);
        }
    }
}

pub(in crate::builtins::io::repl_fs::compat) fn issue(
    runtime: Option<&RuntimeContext>,
    backing: CapturedMappedBacking,
) -> Result<ForeignRef, crate::RuntimeError> {
    let issuer = Issuer::new(runtime);
    let registry = registry();
    let mut state = registry
        .lock()
        .map_err(|_| format_error("memmapfile: authority registry unavailable"))?;
    let handle = state
        .next_handle
        .checked_add(1)
        .ok_or_else(|| format_error("memmapfile: authority handle space exhausted"))?;
    state.next_handle = handle;
    let reference = ForeignRef::managed(
        ForeignResourceKey {
            host_identity: HOST.into(),
            handle,
            generation: GENERATION,
        },
        ForeignTypeIdentity {
            family: "runmat.memmapfile".into(),
            name: MEMMAPFILE_CLASS.display_name().into(),
            version: 1,
        },
        ForeignOwnership::Shared,
        ForeignAffinity::OriginProcess,
        issuer.lifetime(),
        Arc::new(Release(Arc::downgrade(registry))),
    );
    state.records.insert(
        handle,
        Record {
            backing: Arc::new(futures::lock::Mutex::new(backing)),
            lease: reference
                .downgrade()
                .expect("new memmap authority is managed"),
            issuer,
        },
    );
    let register_cleanup =
        runtime.is_some_and(|runtime| state.cleanup_registered.insert(runtime.local_identity()));
    drop(state);
    if register_cleanup {
        let runtime = runtime.expect("session cleanup requires its issuer");
        let identity = runtime.local_identity();
        runtime.register_session_resource_cleanup(move || retire_session(identity));
    }
    Ok(reference)
}

pub(super) fn resolve(
    runtime: Option<&RuntimeContext>,
    receiver: &MemmapReceiverState,
) -> Result<SharedBacking, crate::RuntimeError> {
    let MemmapReceiverAuthority::Opened(reference) = &receiver.authority else {
        return Err(format_error(
            "memmapfile: detached snapshot has no mapped authority",
        ));
    };
    if !reference.is_managed()
        || reference.host_identity != HOST
        || reference.generation != GENERATION
    {
        return Err(format_error("memmapfile: receiver has no sealed authority"));
    }
    let state = registry()
        .lock()
        .map_err(|_| format_error("memmapfile: authority registry unavailable"))?;
    let record = state
        .records
        .get(&reference.handle)
        .ok_or_else(|| format_error("memmapfile: receiver authority is stale"))?;
    if !record.issuer.admits(runtime) {
        return Err(format_error(
            "memmapfile: receiver belongs to another or ended session",
        ));
    }
    let issued = record
        .lease
        .upgrade()
        .ok_or_else(|| format_error("memmapfile: receiver lease expired"))?;
    if !issued.shares_managed_lease(reference)
        || issued.resource_key() != reference.resource_key()
        || reference.type_identity.family != "runmat.memmapfile"
        || reference.type_identity.name != MEMMAPFILE_CLASS.display_name()
        || reference.type_identity.version != 1
        || reference.ownership != ForeignOwnership::Shared
        || reference.affinity != ForeignAffinity::OriginProcess
        || reference.lifetime != record.issuer.lifetime()
    {
        return Err(format_error(
            "memmapfile: receiver authority was not issued here",
        ));
    }
    Ok(record.backing.clone())
}

#[cfg(test)]
#[path = "authority/tests.rs"]
mod tests;
