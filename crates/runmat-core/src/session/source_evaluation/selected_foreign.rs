//! Session-owned, unit-scoped foreign hosts. Executable products retain only
//! their immutable archive receipt; no host or RuntimeContext points back into
//! a FunctionBytecode or selected source owner.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use runmat_execution_artifact::archive::{read_bundle, ArchiveLimits};
use runmat_runtime::context::{RuntimeContext, RuntimeForeignService, RuntimeLocalTaskSupervisor};
use runmat_runtime::foreign::MexRuntimeSession;
use runmat_runtime::user_functions::{
    NativeNameObserver, SelectedNativeInvoker, SelectedNativeRoute,
};
use runmat_runtime::{build_runtime_error, RuntimeError};
use runmat_value::FunctionDefinitionOwner;
use runmat_vm::{FunctionBytecode, SelectedUnitInteropReceipt};

use crate::foreign_artifacts::{
    preflight_selected_interop, prepare_selected_cuda, selected_foreign_service,
    PreparedSelectedHostReplacement, SelectedForeignInstallError, SelectedForeignMaterialization,
};
use crate::session::SessionRetirement;
use crate::{ForeignUnitHost, ForeignUnitRetirement};

use super::{
    SelectedDeclarationAdmission, SelectedGraphPrimary, SelectedGraphSnapshot,
    SessionSourceEvaluator,
};

#[derive(Default)]
pub(super) struct SelectedForeignHostCache {
    entries: RefCell<Vec<Rc<SelectedForeignHostEntry>>>,
    installing: RefCell<Vec<SelectedForeignHostKey>>,
}

struct SelectedForeignHostKey {
    receipt: std::rc::Rc<SelectedUnitInteropReceipt>,
    definition_owner: Option<Arc<FunctionDefinitionOwner>>,
    basis: SelectedForeignBasis,
}

#[derive(Clone)]
enum SelectedForeignBasis {
    Graph(Rc<SelectedGraphPrimary>),
    Declaration(Rc<SelectedDeclarationAdmission>),
}

impl SelectedForeignBasis {
    fn same_instance(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Graph(left), Self::Graph(right)) => Rc::ptr_eq(left, right),
            (Self::Declaration(left), Self::Declaration(right)) => Rc::ptr_eq(left, right),
            _ => false,
        }
    }

    fn source_owner(&self) -> &runmat_runtime::source_context::SelectedProgramSourceOwner {
        match self {
            Self::Graph(primary) => primary.source_owner(),
            Self::Declaration(admission) => admission.product().source_owner(),
        }
    }

    fn same_callback_basis(&self, other: &Self) -> Result<bool, RuntimeError> {
        match (self, other) {
            (Self::Graph(left), Self::Graph(right)) => left
                .same_foreign_callback_basis(right)
                .map_err(selected_error),
            (Self::Declaration(left), Self::Declaration(right)) => Ok(Rc::ptr_eq(left, right)),
            _ => Ok(false),
        }
    }

    fn bind_runtime(
        &self,
        runtime: RuntimeContext,
        evaluator: &Rc<SessionSourceEvaluator>,
    ) -> Result<RuntimeContext, RuntimeError> {
        let runtime = self
            .source_owner()
            .bind_runtime(runtime)
            .map_err(|error| selected_error(error.to_string()))?
            .without_selected_graph_program();
        Ok(match self {
            Self::Graph(primary) => runtime.with_selected_graph_program(
                SelectedGraphSnapshot::program(Rc::clone(primary), evaluator),
            ),
            Self::Declaration(_) => runtime,
        })
    }
}

struct InstallingGuard<'a> {
    cache: &'a SelectedForeignHostCache,
    receipt: std::rc::Rc<SelectedUnitInteropReceipt>,
    definition_owner: Option<Arc<FunctionDefinitionOwner>>,
    basis: SelectedForeignBasis,
}

impl Drop for InstallingGuard<'_> {
    fn drop(&mut self) {
        let mut installing = self.cache.installing.borrow_mut();
        if let Some(position) = installing.iter().position(|key| {
            std::rc::Rc::ptr_eq(&key.receipt, &self.receipt)
                && key.basis.same_instance(&self.basis)
                && match (&key.definition_owner, &self.definition_owner) {
                    (None, None) => true,
                    (Some(left), Some(right)) => Arc::ptr_eq(left, right),
                    _ => false,
                }
        }) {
            installing.remove(position);
        }
    }
}

struct SelectedForeignHostEntry {
    receipt: std::rc::Rc<SelectedUnitInteropReceipt>,
    definition_owner: Option<Arc<FunctionDefinitionOwner>>,
    basis: SelectedForeignBasis,
    host: ForeignUnitHost,
    materialized: Rc<SelectedForeignMaterialization>,
    foreign_service: Rc<dyn RuntimeForeignService>,
    native_route: Rc<SelectedNativeRoute>,
    installed: Cell<bool>,
    retired: Cell<bool>,
}

impl Drop for SelectedForeignHostEntry {
    fn drop(&mut self) {
        if !self.retired.get() {
            if let Err(error) = self
                .materialized
                .persist_unverified("selected foreign host was dropped without verified retirement")
            {
                eprintln!("selected foreign host private bytes were retained but the recovery marker failed: {error}");
            }
        }
    }
}

impl SelectedForeignHostEntry {
    fn foreign_service(&self) -> Rc<dyn RuntimeForeignService> {
        Rc::clone(&self.foreign_service)
    }

    fn matches(
        &self,
        receipt: &SelectedUnitInteropReceipt,
        definition_owner: Option<&Arc<FunctionDefinitionOwner>>,
        basis: &SelectedForeignBasis,
    ) -> Result<bool, RuntimeError> {
        Ok(self.receipt.same_selected_admission(receipt)
            && self.basis.same_callback_basis(basis)?
            && match (&self.definition_owner, definition_owner) {
                (None, None) => true,
                (Some(left), Some(right)) => Arc::ptr_eq(left, right),
                _ => false,
            })
    }
}

fn selected_native_route(
    host: &ForeignUnitHost,
    materialized: &Rc<SelectedForeignMaterialization>,
) -> Rc<SelectedNativeRoute> {
    let mex = host.selected_mex_runtime();
    let observed = mex.installed_name_observer();
    let observation_bytes = Rc::clone(materialized);
    let observer: Rc<NativeNameObserver> = Rc::new(move |runtime, name| {
        let bytes = Rc::clone(&observation_bytes);
        let observed = Rc::clone(&observed);
        runmat_runtime::user_functions::admit_native_name_observation_with(
            &runtime.clone(),
            move || async move {
                let _bytes = bytes;
                runmat_runtime::user_functions::await_native_name_observation(observed(
                    runtime, name,
                ))
                .await
            },
        )
    });
    let invocation_bytes = Rc::clone(materialized);
    let invoker: Rc<SelectedNativeInvoker> = Rc::new(move |runtime, name, arguments, outputs| {
        let mex = Rc::clone(&mex);
        let bytes = Rc::clone(&invocation_bytes);
        runmat_runtime::execution::admit_dynamic_function_load_with(
            &runtime.clone(),
            move || async move {
                let _bytes = bytes;
                mex.load_installed_and_call(&name, arguments, outputs, runtime)
                    .await
            },
        )
    });
    Rc::new(SelectedNativeRoute::new(observer, invoker))
}

/// A trusted ordinary session source may admit a path MEX at the call site.
/// A sealed imported/worker product never receives this route. Its existing
/// selected artifact names still precede ordinary path discovery.
pub(in crate::session) fn ordinary_native_route(
    installed: Option<Rc<SelectedNativeRoute>>,
    mex: Rc<MexRuntimeSession>,
    source_owner: runmat_runtime::source_context::SelectedProgramSourceOwner,
) -> Rc<SelectedNativeRoute> {
    let ordinary_observer = mex.ordinary_name_observer();
    let observed_installed = installed.clone();
    let observer: Rc<NativeNameObserver> = Rc::new(move |runtime, name| {
        let installed = observed_installed.clone();
        let ordinary = Rc::clone(&ordinary_observer);
        runmat_runtime::user_functions::admit_native_name_observation_with(
            &runtime.clone(),
            move || async move {
                if let Some(installed) = installed {
                    if let Some(fact) = installed.observe(runtime.clone(), name.clone()).await? {
                        return Ok(Some(fact));
                    }
                }
                runmat_runtime::user_functions::await_native_name_observation(ordinary(
                    runtime, name,
                ))
                .await
            },
        )
    });
    let invoker: Rc<SelectedNativeInvoker> = Rc::new(move |runtime, name, arguments, outputs| {
        let installed = installed.clone();
        let mex = Rc::clone(&mex);
        let source_owner = source_owner.clone();
        runmat_runtime::execution::admit_dynamic_function_load_with(
            &runtime.clone(),
            move || async move {
                if !source_owner.is_bound_in(&runtime) {
                    return std::ops::ControlFlow::Break(Err(selected_error(
                        "ordinary MEX route differs from its issued source",
                    )));
                }
                if let Some(installed) = installed {
                    match installed.observe(runtime.clone(), name.clone()).await {
                        Ok(Some(_)) => {
                            return runmat_runtime::user_functions::await_dynamic_function_load(
                                installed.invoke(runtime, name, arguments, outputs),
                            )
                            .await;
                        }
                        Ok(None) => {}
                        Err(error) => return std::ops::ControlFlow::Break(Err(error)),
                    }
                }
                mex.load_ordinary_admitted_and_call(
                    &name,
                    arguments,
                    outputs,
                    runtime,
                    &source_owner,
                )
                .await
            },
        )
    });
    Rc::new(SelectedNativeRoute::new(observer, invoker))
}

impl SelectedForeignHostCache {
    fn existing(
        &self,
        receipt: &SelectedUnitInteropReceipt,
        definition_owner: Option<&Arc<FunctionDefinitionOwner>>,
        basis: &SelectedForeignBasis,
    ) -> Result<Option<Rc<SelectedForeignHostEntry>>, RuntimeError> {
        for entry in self.entries.borrow().iter() {
            if entry.matches(receipt, definition_owner, basis)? {
                return Ok(Some(Rc::clone(entry)));
            }
        }
        Ok(None)
    }

    fn install(
        &self,
        receipt: std::rc::Rc<SelectedUnitInteropReceipt>,
        definition_owner: Option<Arc<FunctionDefinitionOwner>>,
        basis: SelectedForeignBasis,
        selected_runtime: &RuntimeContext,
    ) -> Result<Rc<SelectedForeignHostEntry>, RuntimeError> {
        if let Some(existing) = self.existing(&receipt, definition_owner.as_ref(), &basis)? {
            return existing.installed.get().then_some(existing).ok_or_else(|| {
                selected_error("selected foreign host installation failed earlier")
            });
        }
        for key in self.installing.borrow().iter() {
            if key.receipt.same_selected_admission(&receipt)
                && key.basis.same_callback_basis(&basis)?
                && match (&key.definition_owner, &definition_owner) {
                    (None, None) => true,
                    (Some(left), Some(right)) => Arc::ptr_eq(left, right),
                    _ => false,
                }
            {
                return Err(selected_error(
                    "selected foreign host installation is already in progress",
                ));
            }
        }
        self.installing.borrow_mut().push(SelectedForeignHostKey {
            receipt: Rc::clone(&receipt),
            definition_owner: definition_owner.clone(),
            basis: basis.clone(),
        });
        let _installing = InstallingGuard {
            cache: self,
            receipt: Rc::clone(&receipt),
            definition_owner: definition_owner.clone(),
            basis: basis.clone(),
        };
        let bundle = read_bundle(receipt.archive().as_ref(), ArchiveLimits::default())
            .map_err(|error| selected_error(error.to_string()))?;
        let materialized =
            Rc::new(SelectedForeignMaterialization::from_bundle(&bundle).map_err(selected_error)?);
        let host = ForeignUnitHost::new()?;
        host.attach_private_byte_lease(&materialized)?;
        let manifest = &bundle
            .manifest
            .recipes
            .first()
            .ok_or_else(|| selected_error("selected foreign archive has no recipe"))?
            .interop;
        // The receipt constructor already authenticated this one-recipe
        // archive against the exact V3 bytes and issuing source owner.
        let native_route = selected_native_route(&host, &materialized);
        // A retained Java/Python callback holds only a weak foreign port.
        // Keep this exact selected service alive until the host is retired,
        // including between separate invocations of the same selected unit.
        let foreign_service = selected_foreign_service(&host, Rc::clone(&materialized));
        let entry = Rc::new(SelectedForeignHostEntry {
            receipt,
            definition_owner,
            basis,
            host,
            native_route,
            materialized,
            foreign_service,
            installed: Cell::new(false),
            retired: Cell::new(false),
        });
        // Publish the owner before any installer callback can run or unwind.
        // An incomplete install remains an explicit retirement obligation.
        self.entries.borrow_mut().push(Rc::clone(&entry));
        let install_runtime = selected_runtime
            .clone()
            .with_selected_native_route(Some(Rc::clone(&entry.native_route)))
            .with_service_ports(
                selected_runtime
                    .service_ports()
                    .clone()
                    .without_foreign()
                    .with_foreign(entry.foreign_service()),
            );
        let _selected_scope = install_runtime.enter();
        let install_result = (|| {
            let plan =
                preflight_selected_interop(&entry.host, manifest, entry.materialized.as_ref())?;
            prepare_selected_cuda(&plan)?;
            PreparedSelectedHostReplacement::prepare(&entry.host, plan)?
                .admit_commit()?
                .commit();
            Ok::<(), SelectedForeignInstallError>(())
        })();
        entry.installed.set(install_result.is_ok());
        install_result.map_err(SelectedForeignInstallError::into_runtime_error)?;
        Ok(entry)
    }

    async fn retire_all(
        &self,
        runtime: &RuntimeContext,
        evaluator: &Rc<SessionSourceEvaluator>,
        local_tasks: &RuntimeLocalTaskSupervisor,
    ) -> SessionRetirement {
        loop {
            let entries = self.entries.borrow().clone();
            if entries.is_empty() {
                return SessionRetirement::Retired(Ok(()));
            }
            let mut first_error = None;
            for entry in entries {
                // A MEX shutdown callback can admit another exact selected
                // unit. Iterate again after this batch so the new host cannot
                // be silently dropped at session close.
                let context = match entry.basis.bind_runtime(runtime.clone(), evaluator) {
                    Ok(context) => context,
                    Err(error) => {
                        first_error = Some(crate::session::retirement::combine_failure(
                            first_error.take(),
                            selected_error(error.to_string()),
                        ));
                        continue;
                    }
                };
                let ports = context
                    .service_ports()
                    .clone()
                    .with_foreign(entry.foreign_service());
                let context = context
                    .with_selected_native_route(Some(Rc::clone(&entry.native_route)))
                    .with_service_ports(ports);
                match entry.host.shutdown_retirement(context, local_tasks).await {
                    ForeignUnitRetirement::Retained(error) => {
                        first_error = Some(crate::session::retirement::combine_failure(
                            first_error.take(),
                            error,
                        ));
                    }
                    ForeignUnitRetirement::Retired(result) => {
                        entry.retired.set(true);
                        self.entries
                            .borrow_mut()
                            .retain(|current| !Rc::ptr_eq(current, &entry));
                        if let Err(error) = result {
                            first_error = Some(crate::session::retirement::combine_failure(
                                first_error.take(),
                                error,
                            ));
                        }
                    }
                }
            }
            if let Some(error) = first_error {
                // Only the cache's remaining exact host entries determine
                // whether a callback error left live adapters behind.
                return if self.entries.borrow().is_empty() {
                    SessionRetirement::Retired(Err(error))
                } else {
                    SessionRetirement::Retained(error)
                };
            }
        }
    }
}

impl SessionSourceEvaluator {
    /// Bind the exact selected host after the caller has selected a body (or
    /// script product) and its admitted V3/source owner. A missing receipt
    /// clears the mutable ambient foreign catalog rather than inheriting it.
    pub(in crate::session) fn bind_selected_foreign(
        self: &Rc<Self>,
        runtime: RuntimeContext,
        primary: Rc<SelectedGraphPrimary>,
        body: Option<&FunctionBytecode>,
        receipt: Option<std::rc::Rc<SelectedUnitInteropReceipt>>,
        ordinary_mex: Option<Rc<MexRuntimeSession>>,
    ) -> Result<RuntimeContext, RuntimeError> {
        if !primary.source_owner().is_issued_in(&runtime) {
            return Err(selected_error(
                "selected foreign unit belongs to another session",
            ));
        }
        if match (primary.selected_interop_receipt(), receipt.as_ref()) {
            (None, None) => false,
            (Some(selected), Some(candidate)) => !selected.same_selected_admission(candidate),
            _ => true,
        } {
            return Err(selected_error(
                "selected foreign receipt differs from its admitted primary",
            ));
        }
        if let Some(receipt) = receipt.as_ref() {
            let canonical = primary.canonical_v3().map_err(selected_error)?;
            if !receipt.matches_selected_unit(&canonical, primary.source_owner()) {
                return Err(selected_error(
                    "selected foreign receipt differs from its code/source owner",
                ));
            }
        }
        let definition_owner = if let Some(body) = body {
            let selected = primary
                .selected_registry()
                .get(body.function)
                .ok_or_else(|| {
                    selected_error("selected foreign body is absent from its registry")
                })?;
            if !Arc::ptr_eq(&body.definition_owner, &selected.definition_owner)
                || body
                    .selected_source_owner
                    .as_ref()
                    .is_none_or(|owner| !owner.same_issuance(primary.source_owner()))
                || match (
                    body.selected_executable_product
                        .as_ref()
                        .and_then(|product| product.interop_receipt()),
                    receipt.as_ref(),
                ) {
                    (None, None) => false,
                    (Some(body_receipt), Some(candidate)) => {
                        !body_receipt.same_selected_admission(candidate)
                    }
                    _ => true,
                }
            {
                return Err(selected_error(
                    "selected foreign body differs from its exact publication",
                ));
            }
            Some(Arc::clone(&body.definition_owner))
        } else {
            None
        };
        let runtime = primary
            .source_owner()
            .bind_runtime(runtime)
            .map_err(|error| selected_error(error.to_string()))?
            .without_selected_graph_program()
            .with_selected_graph_program(SelectedGraphSnapshot::program(Rc::clone(&primary), self));
        let installed = receipt
            .map(|receipt| {
                self.selected_foreign_hosts.install(
                    receipt,
                    definition_owner,
                    SelectedForeignBasis::Graph(Rc::clone(&primary)),
                    &runtime,
                )
            })
            .transpose()?;
        let ports = runtime.service_ports().clone().without_foreign();
        let route = installed
            .as_ref()
            .map(|entry| Rc::clone(&entry.native_route));
        let runtime = match ordinary_mex {
            Some(mex) => runtime.with_ordinary_selected_native_route(
                ordinary_native_route(route, Rc::clone(&mex), primary.source_owner().clone()),
                mex,
                primary.source_owner().clone(),
            ),
            None => runtime.with_selected_native_route(route),
        };
        Ok(runtime.with_service_ports(match installed {
            Some(entry) => ports.with_foreign(entry.foreign_service()),
            None => ports,
        }))
    }

    /// A declaration-only class has source/foreign authority but no callable
    /// or script entrypoint. Bind its exact issued source and checked receipt
    /// without manufacturing a Graph primary or inheriting ambient ports.
    pub(in crate::session) fn bind_selected_declaration_foreign(
        self: &Rc<Self>,
        runtime: RuntimeContext,
        declaration: Rc<SelectedDeclarationAdmission>,
    ) -> Result<RuntimeContext, RuntimeError> {
        let product = declaration.product();
        if !product.source_owner().is_issued_in(&runtime) {
            return Err(selected_error(
                "selected class declaration belongs to another session",
            ));
        }
        let basis = SelectedForeignBasis::Declaration(Rc::clone(&declaration));
        let runtime = basis.bind_runtime(runtime, self)?;
        let installed = product
            .interop_receipt()
            .map(|receipt| {
                if !receipt.matches_selected_unit(product.canonical_bytes(), product.source_owner())
                {
                    return Err(selected_error(
                        "selected class foreign receipt differs from its admitted product",
                    ));
                }
                self.selected_foreign_hosts.install(
                    Rc::clone(receipt),
                    None,
                    basis.clone(),
                    &runtime,
                )
            })
            .transpose()?;
        let ports = runtime.service_ports().clone().without_foreign();
        let route = installed
            .as_ref()
            .map(|entry| Rc::clone(&entry.native_route));
        Ok(runtime
            .with_selected_native_route(route)
            .with_service_ports(match installed {
                Some(entry) => ports.with_foreign(entry.foreign_service()),
                None => ports,
            }))
    }

    /// Ordered retirement retains any host whose MEX callbacks rejected
    /// shutdown, together with its exact materialization and code basis.
    pub(in crate::session) async fn retire_selected_foreign_hosts(
        self: &Rc<Self>,
        runtime: &RuntimeContext,
        local_tasks: &RuntimeLocalTaskSupervisor,
    ) -> Result<(), RuntimeError> {
        self.retire_selected_foreign_hosts_retirement(runtime, local_tasks)
            .await
            .into_result()
    }

    pub(in crate::session) async fn retire_selected_foreign_hosts_retirement(
        self: &Rc<Self>,
        runtime: &RuntimeContext,
        local_tasks: &RuntimeLocalTaskSupervisor,
    ) -> SessionRetirement {
        self.selected_foreign_hosts
            .retire_all(runtime, self, local_tasks)
            .await
    }
}

fn selected_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:SelectedForeignHost")
        .build()
}
