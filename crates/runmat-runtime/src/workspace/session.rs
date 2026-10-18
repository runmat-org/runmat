use runmat_thread_local::runmat_thread_local;
use runmat_value::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU64, Ordering};

#[path = "session/transaction.rs"]
mod transaction;
pub use globals::SessionGlobalGeneration;
pub use transaction::PreparedSessionGlobalWrite;
pub(crate) use transaction::SessionMutationGate;

#[path = "session/globals.rs"]
mod globals;
use globals::{PreparedGlobalWrite, SessionGlobals};

#[path = "session/persistent.rs"]
mod persistent;
pub use persistent::{
    persistent_named_value, persistent_revision, persistent_slot_value, store_persistent_named,
    store_persistent_slot, update_persistent_slot, PersistentBindingContinuation,
    PersistentInvalidationRevision, PersistentRevision,
};

/// Exact, non-owning capability for one Runtime session's variable store.
/// Executor services retain it across suspension instead of re-resolving an
/// ambient RuntimeContext which may belong to another invocation.
#[derive(Clone, Debug)]
pub struct SessionVariableAuthority {
    state: Weak<crate::context::RuntimeContextState>,
}

impl PartialEq for SessionVariableAuthority {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.state, &other.state)
    }
}

impl Eq for SessionVariableAuthority {}

impl SessionVariableAuthority {
    pub(crate) fn new(state: &Rc<crate::context::RuntimeContextState>) -> Self {
        Self {
            state: Rc::downgrade(state),
        }
    }

    fn with_state<R>(
        &self,
        operation: impl FnOnce(&SessionVariableState) -> R,
    ) -> Result<R, crate::RuntimeError> {
        let state = self.state.upgrade().ok_or_else(session_variables_closed)?;
        state
            .session_variables
            .try_borrow()
            .map(|variables| operation(&variables))
            .map_err(session_variables_read_borrow)
    }

    fn with_state_mut<R>(
        &self,
        operation: impl FnOnce(&mut SessionVariableState) -> R,
    ) -> Result<R, crate::RuntimeError> {
        let state = self.state.upgrade().ok_or_else(session_variables_closed)?;
        state
            .session_variables
            .try_borrow_mut()
            .map(|mut variables| operation(&mut variables))
            .map_err(session_variables_write_borrow)
    }

    pub fn global_value(&self, name: &str) -> Result<Option<Value>, crate::RuntimeError> {
        self.with_state(|state| state.global_value(name))
    }

    /// Test canonical membership without cloning the stored value or tracing
    /// a new result alias. Presence queries do not materialize payloads.
    pub fn contains_global(&self, name: &str) -> Result<bool, crate::RuntimeError> {
        self.with_state(|state| state.globals.named(name).is_some())
    }

    pub fn contains_global_target(
        &self,
        target: &crate::workspace::GlobalBindingTarget,
    ) -> Result<bool, crate::RuntimeError> {
        self.with_state(|state| match target {
            crate::workspace::GlobalBindingTarget::Named(name) => {
                state.globals.named(name.as_str()).is_some()
            }
            crate::workspace::GlobalBindingTarget::LegacySlot(slot) => {
                state.globals.slot(*slot).is_some()
            }
        })
    }

    pub fn global_names(&self) -> Result<Vec<String>, crate::RuntimeError> {
        self.with_state(SessionVariableState::global_names)
    }

    pub fn global_revision(&self) -> Result<GlobalRevision, crate::RuntimeError> {
        self.with_state(|state| state.global_revision)
    }

    /// Capture this exact session's immutable COW global generation. No Value,
    /// name, or table allocation occurs at this boundary.
    pub fn capture_global_generation(
        &self,
        context: &crate::context::RuntimeContext,
    ) -> Result<SessionGlobalGeneration, crate::RuntimeError> {
        self.validate_context(context)?;
        self.with_state(|state| state.globals.capture_generation(state.global_revision))
    }

    pub fn declare_global(&self, name: &str) -> Result<Value, crate::RuntimeError> {
        let context = crate::context::legacy::active().ok_or_else(session_context_required)?;
        self.declare_global_with_context(&context, name)
    }

    pub fn declare_global_with_context(
        &self,
        context: &crate::context::RuntimeContext,
        name: &str,
    ) -> Result<Value, crate::RuntimeError> {
        self.validate_context(context)?;
        if let Some(value) = self.with_state(|state| state.globals.named(name).cloned())? {
            return Ok(value);
        }
        let prepared = self.prepare_declare_global_with_context(context, name)?;
        prepared.checkpoint(context)?;
        prepared.commit();
        self.global_value(name)?
            .ok_or_else(session_variables_closed)
    }

    pub fn store_global(&self, name: &str, value: Value) -> Result<(), crate::RuntimeError> {
        let context = crate::context::legacy::active().ok_or_else(session_context_required)?;
        self.store_global_with_context(&context, name, value)
    }

    pub fn store_global_with_context(
        &self,
        context: &crate::context::RuntimeContext,
        name: &str,
        value: Value,
    ) -> Result<(), crate::RuntimeError> {
        let prepared = self.prepare_store_global_with_context(context, name, value)?;
        prepared.checkpoint(context)?;
        prepared.commit();
        Ok(())
    }

    /// Prepare and publish a borrowed global value without cloning before
    /// finite admission. Unsupported borrowed aggregate aliases fail before
    /// effects; the explicit unbounded policy retains its compatibility clone.
    pub fn store_global_ref_with_context(
        &self,
        context: &crate::context::RuntimeContext,
        name: &str,
        value: &Value,
    ) -> Result<(), crate::RuntimeError> {
        self.validate_context(context)?;
        crate::execution::validate_storable_value(value)?;
        let roots = crate::execution::RootedValueLease::empty();
        let prepared = self.prepare_write(context, roots, |globals, _| {
            globals.prepare_named_ref(context, name, value)
        })?;
        prepared.checkpoint(context)?;
        prepared.commit();
        Ok(())
    }

    pub fn global_slot(&self, slot: usize) -> Result<Option<Value>, crate::RuntimeError> {
        self.with_state(|state| state.globals.slot(slot).cloned())
    }

    pub fn store_global_slot(&self, slot: usize, value: Value) -> Result<(), crate::RuntimeError> {
        let context = crate::context::legacy::active().ok_or_else(session_context_required)?;
        self.store_global_slot_with_context(&context, slot, value)
    }

    pub fn store_global_slot_with_context(
        &self,
        context: &crate::context::RuntimeContext,
        slot: usize,
        value: Value,
    ) -> Result<(), crate::RuntimeError> {
        let prepared = self.prepare_store_global_slot_with_context(context, slot, value)?;
        prepared.checkpoint(context)?;
        prepared.commit();
        Ok(())
    }

    pub fn store_global_slot_ref_with_context(
        &self,
        context: &crate::context::RuntimeContext,
        slot: usize,
        value: &Value,
    ) -> Result<(), crate::RuntimeError> {
        self.validate_context(context)?;
        crate::execution::validate_storable_value(value)?;
        let roots = crate::execution::RootedValueLease::empty();
        let prepared = self.prepare_write(context, roots, |globals, _| {
            globals.prepare_slot_ref(context, slot, value)
        })?;
        prepared.checkpoint(context)?;
        prepared.commit();
        Ok(())
    }

    fn prepare_write(
        &self,
        context: &crate::context::RuntimeContext,
        roots: crate::execution::RootedValueLease,
        prepare: impl FnOnce(
            &SessionGlobals,
            &crate::execution::RootedValueLease,
        ) -> Result<PreparedGlobalWrite, crate::RuntimeError>,
    ) -> Result<PreparedSessionGlobalWrite, crate::RuntimeError> {
        let state = self.validate_context(context)?;
        let cancellation = context.cancellation();
        if cancellation.is_cancelled() {
            return Err(globals::session_allocation_error(
                context,
                crate::context::host_allocation::HostAllocationError::Cancelled,
            ));
        }
        let variables = state
            .session_variables
            .try_borrow_mut()
            .map_err(session_variables_write_borrow)?;
        let expected_revision = variables.global_revision;
        let reservation = state.session_global_mutation.reserve()?;
        let prepared = prepare(&variables.globals, &roots)?;
        let published_revision = prepared
            .changes_session_value()
            .then(|| expected_revision.next())
            .transpose()?;
        drop(variables);
        let token = PreparedSessionGlobalWrite {
            state: Rc::downgrade(&state),
            cancellation,
            expected_revision,
            published_revision,
            prepared: Some(prepared),
            _roots: roots,
        };
        reservation.transfer();
        Ok(token)
    }

    fn validate_context(
        &self,
        context: &crate::context::RuntimeContext,
    ) -> Result<Rc<crate::context::RuntimeContextState>, crate::RuntimeError> {
        let state = self.state.upgrade().ok_or_else(session_variables_closed)?;
        if !Rc::ptr_eq(&state, context.state()) {
            return Err(session_wrong_owner());
        }
        Ok(state)
    }

    pub fn prepare_store_global_with_context(
        &self,
        context: &crate::context::RuntimeContext,
        name: &str,
        value: Value,
    ) -> Result<PreparedSessionGlobalWrite, crate::RuntimeError> {
        self.validate_context(context)?;
        crate::execution::validate_storable_value(&value)?;
        let roots = crate::execution::RootedValueLease::for_value_in_context(&value, context)
            .map_err(|cause| rooted_session_value_error(context, cause))?;
        self.prepare_write(context, roots, |globals, roots| {
            globals.prepare_named(context, name, value, roots.shared_handles())
        })
    }

    pub fn prepare_store_global_slot_with_context(
        &self,
        context: &crate::context::RuntimeContext,
        slot: usize,
        value: Value,
    ) -> Result<PreparedSessionGlobalWrite, crate::RuntimeError> {
        self.validate_context(context)?;
        crate::execution::validate_storable_value(&value)?;
        let roots = crate::execution::RootedValueLease::for_value_in_context(&value, context)
            .map_err(|cause| rooted_session_value_error(context, cause))?;
        self.prepare_write(context, roots, |globals, roots| {
            globals.prepare_slot(context, slot, value, roots.shared_handles())
        })
    }

    pub fn prepare_declare_global_with_context(
        &self,
        context: &crate::context::RuntimeContext,
        name: &str,
    ) -> Result<PreparedSessionGlobalWrite, crate::RuntimeError> {
        self.validate_context(context)?;
        let roots = crate::execution::RootedValueLease::empty();
        self.prepare_write(context, roots, |globals, _roots| {
            globals.prepare_declaration(context, name)
        })
    }

    pub fn prepare_declare_global_slot_with_context(
        &self,
        context: &crate::context::RuntimeContext,
        slot: usize,
    ) -> Result<PreparedSessionGlobalWrite, crate::RuntimeError> {
        self.validate_context(context)?;
        let roots = crate::execution::RootedValueLease::empty();
        self.prepare_write(context, roots, |globals, _roots| {
            globals.prepare_slot_declaration(context, slot)
        })
    }
}

fn rooted_session_value_error(
    context: &crate::context::RuntimeContext,
    cause: crate::execution::RootedValueAdmissionError,
) -> crate::RuntimeError {
    match cause {
        crate::execution::RootedValueAdmissionError::Host(cause) => {
            globals::session_allocation_error(context, cause)
        }
        crate::execution::RootedValueAdmissionError::Gc(cause) => {
            crate::build_runtime_error("session-global value rooting failed")
                .with_identifier(format!("{}:Workspace:Rooting", context.error_namespace()))
                .with_source(cause)
                .build()
        }
    }
}

fn session_wrong_owner() -> crate::RuntimeError {
    crate::runtime_error::semantic_error(
        "RunMat:Workspace:SessionOwner",
        "session variable mutation belongs to a different RuntimeContext",
    )
}

fn session_write_reserved() -> crate::RuntimeError {
    crate::runtime_error::semantic_error(
        "RunMat:Workspace:SessionMutationReserved",
        "another session-global mutation is already prepared",
    )
}

fn session_variables_closed() -> crate::RuntimeError {
    crate::runtime_error::semantic_error(
        "RunMat:Workspace:SessionClosed",
        "session variable authority is closed",
    )
}

fn session_context_required() -> crate::RuntimeError {
    crate::runtime_error::semantic_error(
        "RunMat:Workspace:SessionContextRequired",
        "session-global mutation requires its captured RuntimeContext",
    )
}

fn session_variables_read_borrow(error: std::cell::BorrowError) -> crate::RuntimeError {
    crate::runtime_error::semantic_error(
        "RunMat:Workspace:GlobalBorrow",
        format!("session-global workspace read is reentrant: {error}"),
    )
}

fn session_variables_write_borrow(error: std::cell::BorrowMutError) -> crate::RuntimeError {
    crate::runtime_error::semantic_error(
        "RunMat:Workspace:GlobalBorrow",
        format!("session-global workspace mutation is reentrant: {error}"),
    )
}

/// Checked session identity and generation for one global-store version.
/// Session identities are never reused, and generation exhaustion fails
/// before mutation, so equality cannot acquire an ABA alias.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalRevision {
    session: u64,
    generation: u64,
}

static NEXT_GLOBAL_REVISION_SESSION: AtomicU64 = AtomicU64::new(1);

impl Default for GlobalRevision {
    fn default() -> Self {
        let session = NEXT_GLOBAL_REVISION_SESSION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .expect("session global identity exhausted");
        Self {
            session,
            generation: 0,
        }
    }
}

impl GlobalRevision {
    fn next(self) -> Result<Self, crate::RuntimeError> {
        let generation = self.generation.checked_add(1).ok_or_else(|| {
            crate::runtime_error::semantic_error(
                "RunMat:Workspace:GlobalRevisionExhausted",
                "session global revision space is exhausted",
            )
        })?;
        Ok(Self {
            session: self.session,
            generation,
        })
    }
}

/// Named session variables and legacy bytecode-keyed slot storage.
///
/// Slot-to-local synchronization belongs to the executor. This state owns only
/// values whose lifetime and identity cross an individual VM frame.
#[derive(Debug, Default)]
pub struct SessionVariableState {
    globals: SessionGlobals,
    global_revision: GlobalRevision,
    persistent_revision: PersistentRevision,
    persistent_invalidation_leases: HashMap<String, Weak<()>>,
    persistent_slots: HashMap<(String, usize), Value>,
    persistent_names: HashMap<(String, String), Value>,
}

impl SessionVariableState {
    /// Only RuntimeContext's checked lab fork may snapshot this state.
    /// A prepared global write must exclude the fork before cloning its
    /// formerly unique generation root.
    pub(crate) fn fork_snapshot(&self) -> Self {
        Self {
            globals: self.globals.clone(),
            global_revision: GlobalRevision::default(),
            persistent_revision: self.persistent_revision.clone(),
            persistent_invalidation_leases: self.persistent_invalidation_leases.clone(),
            persistent_slots: self.persistent_slots.clone(),
            persistent_names: self.persistent_names.clone(),
        }
    }
}

impl SessionVariableState {
    pub(crate) fn global_value(&self, name: &str) -> Option<Value> {
        self.globals.named(name).cloned()
    }

    pub(crate) fn global_names(&self) -> Vec<String> {
        self.globals.names()
    }

    fn declare_global(&mut self, name: &str) -> Result<Value, crate::RuntimeError> {
        if let Some(value) = self.globals.named(name) {
            return Ok(value.clone());
        }
        let revision = self.global_revision.next()?;
        let value = canonical_empty_workspace_value();
        self.globals
            .store_named_unbounded(name, value.clone())
            .expect("canonical empty workspace value is storable");
        self.global_revision = revision;
        Ok(value)
    }

    fn store_global(&mut self, name: &str, value: Value) -> Result<(), crate::RuntimeError> {
        let revision = self.global_revision.next()?;
        self.globals.store_named_unbounded(name, value)?;
        self.global_revision = revision;
        Ok(())
    }
}

runmat_thread_local! {
    static LEGACY_SESSION_VARIABLES: RefCell<SessionVariableState> = RefCell::new(SessionVariableState::default());
}

fn with_state<R>(operation: impl FnOnce(&SessionVariableState) -> R) -> R {
    if let Some(context) = crate::context::legacy::active() {
        return operation(&context.state().session_variables.borrow());
    }
    LEGACY_SESSION_VARIABLES.with(|state| operation(&state.borrow()))
}

fn with_state_mut<R>(operation: impl FnOnce(&mut SessionVariableState) -> R) -> R {
    if let Some(context) = crate::context::legacy::active() {
        return operation(&mut context.state().session_variables.borrow_mut());
    }
    LEGACY_SESSION_VARIABLES.with(|state| operation(&mut state.borrow_mut()))
}

pub fn global_names() -> Vec<String> {
    with_state(SessionVariableState::global_names)
}

pub fn global_value(name: &str) -> Option<Value> {
    with_state(|state| state.global_value(name))
}

/// Establish a named global and return its current value. A first declaration
/// initializes the shared value to the canonical empty matrix, so declaring a
/// global never exposes a stale local value from the declaring frame.
pub fn declare_global_named(name: &str) -> Result<Value, crate::RuntimeError> {
    if let Some(context) = crate::context::legacy::active() {
        return context
            .session_variable_authority()
            .declare_global_with_context(&context, name);
    }
    with_state_mut(|state| state.declare_global(name))
}

fn canonical_empty_workspace_value() -> Value {
    Value::Tensor(
        runmat_value::Tensor::new(Vec::new(), vec![0, 0])
            .expect("the canonical empty workspace shape is valid"),
    )
}

/// Revision of the session's named global store. Native frames use this to
/// avoid repeatedly cloning and rooting an unchanged global on every read.
pub fn global_revision() -> GlobalRevision {
    with_state(|state| state.global_revision)
}

/// Store a global through its semantic name.
///
/// Executor-local slot numbers are deliberately not part of this API. Native,
/// bytecode, and browser hosts can therefore share one session global without
/// agreeing on a frame layout.
pub fn store_global_named(name: &str, value: Value) -> Result<(), crate::RuntimeError> {
    if let Some(context) = crate::context::legacy::active() {
        return context
            .session_variable_authority()
            .store_global_with_context(&context, name, value);
    }
    with_state_mut(|state| state.store_global(name, value))
}

pub fn roots() -> Vec<Value> {
    with_state(|state| {
        state
            .globals
            .values()
            .chain(state.persistent_slots.values())
            .chain(state.persistent_names.values())
            .cloned()
            .collect()
    })
}

/// Borrow session roots without copying tensor payloads. An active mutation
/// makes the view unavailable rather than proving there are no retained aliases.
pub fn visit_roots(visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
    fn visit(
        state: &SessionVariableState,
        visitor: &mut dyn FnMut(&Value) -> Option<()>,
    ) -> Option<()> {
        for value in state
            .globals
            .values()
            .chain(state.persistent_slots.values())
            .chain(state.persistent_names.values())
        {
            visitor(value)?;
        }
        Some(())
    }
    if let Some(context) = crate::context::legacy::active() {
        let state = context.state().session_variables.try_borrow().ok()?;
        return visit(&state, visitor);
    }
    LEGACY_SESSION_VARIABLES.with(|state| {
        let state = state.try_borrow().ok()?;
        visit(&state, visitor)
    })
}

/// Store a global addressed by a legacy bytecode slot. This namespace is
/// distinct from user-written global names, even when a name is `var_4`.
pub fn store_global_slot(index: usize, value: Value) -> Result<(), crate::RuntimeError> {
    if let Some(context) = crate::context::legacy::active() {
        return context
            .session_variable_authority()
            .store_global_slot_with_context(&context, index, value);
    }
    with_state_mut(|state| {
        let revision = state.global_revision.next()?;
        state.globals.store_slot_unbounded(index, value)?;
        state.global_revision = revision;
        Ok(())
    })
}

pub fn global_slot_value(index: usize) -> Option<Value> {
    with_state(|state| state.globals.slot(index).cloned())
}

#[doc(hidden)]
pub fn reset_legacy_state_for_tests() {
    LEGACY_SESSION_VARIABLES.with(|state| *state.borrow_mut() = SessionVariableState::default());
}

#[cfg(test)]
#[path = "session/persistent_revision_tests.rs"]
mod persistent_revision_tests;

#[cfg(test)]
#[path = "session/tests.rs"]
mod tests;
