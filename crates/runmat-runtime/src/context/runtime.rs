use super::{
    ContextFuture, RuntimeContextGuard, RuntimeContextState, RuntimeInvocationArity,
    RuntimeServicePorts,
};
use crate::execution::RuntimeExecutionServices;
use runmat_execution::cancellation::{CancellationSource, CancellationToken};
use runmat_execution::profiling::CommandProfilingReceipt;
use runmat_execution::value::GraphStaticReferenceSite;
use runmat_execution::FunctionBindingOrigin;
use std::cell::{Cell, RefCell};
use std::future::Future;
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

mod background_callback;
#[path = "runtime/source_selection.rs"]
mod source_selection;
#[cfg(feature = "plot-core")]
pub(crate) use background_callback::BackgroundCallbackAuthority;
use source_selection::InvocationSourceSelection;
pub use source_selection::ProgramSourceSelectionError;

static NEXT_RUNTIME_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);

mod selected_method;
use selected_method::AdmittedSelectedMethodFrame;
pub use selected_method::SelectedMethodDispatch;

pub type SelectedStaticSiteResolver =
    dyn Fn(
        Option<FunctionBindingOrigin>,
        GraphStaticReferenceSite,
    ) -> Result<Option<runmat_value::BoundFunctionBinding>, &'static str>;

mod selected_function;
pub use selected_function::{SelectedGraphFunction, SelectedGraphFunctionResolver};

/// Process-local identity for one runtime context state allocation.
///
/// This identity lets compatibility adapters partition ambient state while
/// independently polled async sessions share an OS thread. It is not a
/// durable, serialized, or cross-process execution identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RuntimeContextLocalId(u64);

impl RuntimeContextLocalId {
    fn next() -> Self {
        let id = NEXT_RUNTIME_CONTEXT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("runtime context identity space exhausted");
        Self(id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeLanguageMode {
    Matlab,
    RunMat,
    Strict,
}

/// Language and dynamic-source policy bound to one invocation.
///
/// Session defaults live in [`RuntimeContextState`]. An invocation projection
/// lets independently pending work retain the policy captured at admission
/// without mutating those shared defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeLanguagePolicy {
    pub language_mode: RuntimeLanguageMode,
    pub extensions_enabled: bool,
    pub top_level_await_enabled: bool,
    pub dynamic_eval_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeExecutionStack {
    Process,
    Segmented,
}

#[must_use]
pub struct RuntimeExecutionStackGuard {
    state: Rc<RuntimeContextState>,
    previous: RuntimeExecutionStack,
}

/// Restores the invocation assignment that was active before a nested program
/// execution entered this runtime context.
#[must_use]
pub struct ProgramExecutionAssignmentGuard {
    state: Rc<RuntimeContextState>,
    previous: Option<runmat_execution::ProgramExecutionAssignment>,
}

#[must_use]
pub struct ProgramExecutionJobGuard {
    state: Rc<RuntimeContextState>,
    previous: Option<runmat_execution::JobId>,
}

impl Drop for ProgramExecutionJobGuard {
    fn drop(&mut self) {
        *self.state.execution_job.borrow_mut() = self.previous.take();
    }
}

impl Drop for ProgramExecutionAssignmentGuard {
    fn drop(&mut self) {
        *self.state.execution_assignment.borrow_mut() = self.previous.take();
    }
}

impl Drop for RuntimeExecutionStackGuard {
    fn drop(&mut self) {
        self.state.execution_stack.set(self.previous);
    }
}

pub const DEFAULT_CALLSTACK_LIMIT: usize = 200;
pub const DEFAULT_ERROR_NAMESPACE: &str = "RunMat";

/// Complete explicit runtime authority for one session/invocation tree.
#[derive(Clone)]
pub struct RuntimeContext {
    cleanup_completion: Option<
        crate::builtins::introspection::on_cleanup::lifecycle::registry::CleanupCompletionId,
    >,
    acceleration_policy: Option<super::RuntimeAccelerationPolicy>,
    graphics_callback_mutation: Option<super::GraphicsCallbackMutationScope>,
    execution: Rc<dyn RuntimeExecutionServices>,
    /// Call-local selected V3 graph authority. This carrier retains no
    /// execution port and snapshots Values only when a task is spawned.
    selected_graph_program: Option<crate::execution::DeferredGraphProgram>,
    /// Call-local target admission for native MIR futures. The selected Core
    /// registry is captured without a RuntimeContext or execution service.
    selected_graph_function_resolver: Option<Rc<SelectedGraphFunctionResolver>>,
    /// Selected graph admission supplies one unit-local site map. The caller
    /// origin changes with the VM frame; no numeric ID or display name selects
    /// a cross-unit target.
    selected_static_site_resolver: Option<Rc<SelectedStaticSiteResolver>>,
    selected_static_caller: Option<FunctionBindingOrigin>,
    native_resolution_scope: crate::user_functions::NativeResolutionScope,
    pub(super) environment: Rc<std::cell::RefCell<super::environment::EnvironmentState>>,
    // Immutable program selection is retained across invocation projections;
    // cloning a context must not copy its domain-contribution vector.
    program_revision: Option<Arc<runmat_execution::ProgramRevision>>,
    program_sources: Option<Rc<crate::source_context::ProgramSourceBinding>>,
    source_current: Option<
        super::poll_projection::PollProjection<crate::source_context::SourceCurrentProjection>,
    >,
    source_selection: InvocationSourceSelection,
    source_catalog_view: Option<
        super::poll_projection::PollProjection<crate::source_context::SourceCatalogProjection>,
    >,
    debug_projection:
        super::poll_projection::PollProjection<crate::debug_context::DebugFrameProjection>,
    output_projection: super::poll_projection::PollProjection<
        super::output_projection::InvocationOutputProjection,
    >,
    call_projection: super::poll_projection::PollProjection<super::call_projection::CallProjection>,
    call_projection_lineage: Rc<()>,
    console_capture_projection:
        super::poll_projection::PollProjection<crate::console::ConsoleCaptureProjection>,
    #[cfg(feature = "plot-core")]
    plot_receipts: crate::builtins::plotting::state::FigureReceiptOwner,
    // Clones admitted for one command share this exact origin. The next
    // command installs a new owner; deferred Send feedback retains only its
    // linear contribution, not this Rc-backed RuntimeContext.
    command_profiling: Option<CommandProfilingReceipt>,
    /// Core's exact, already-registered WGPU façade at command admission.
    /// This is a negative comparison fact, not public completeness authority.
    command_device_provider: Option<&'static dyn runmat_accelerate_api::AccelProvider>,
    command_device_access: Option<runmat_accelerate_api::WgpuDeviceAccess>,
    /// Poll-local differentiation authority for this exact invocation tree.
    /// An unrelated context selects no tape even when entered beneath one.
    tape_poll_authority: Option<crate::builtins::deep_learning::autodiff::TapePollAuthority>,
    search_path: Option<Arc<crate::builtins::common::path_state::SearchPath>>,
    services: RuntimeServicePorts,
    state: Rc<RuntimeContextState>,
    /// Deterministic parfor task RNG only; all other mutable language state
    /// remains on the original session owner.
    parallel_random_scope: Option<Rc<crate::builtins::common::random::ParallelRandomScope>>,
    invocation_language_policy: Option<Rc<Cell<RuntimeLanguagePolicy>>>,
    invocation_arity: Option<RuntimeInvocationArity>,
    // Captured at semantic-call admission before a child executor can be
    // polled outside the caller's constructor-receiver selection.
    constructor_receiver: Option<crate::execution::RootedValue>,
    selected_classes: Option<Rc<crate::class_registry::SelectedClassProjection>>,
    selected_object_class: Option<Rc<runmat_value::SelectedObjectClassFacts>>,
    selected_method_body: Option<(
        Rc<runmat_value::SelectedObjectClassFacts>,
        runmat_types::ClassMethodOwner,
        runmat_value::BoundFunctionBinding,
        SelectedMethodDispatch,
    )>,
    selected_method_frame: Option<AdmittedSelectedMethodFrame>,
    retained_super_constructors: Option<Rc<[super::RetainedSuperConstructorBinding]>>,
    retained_super_caller_origin: Option<runmat_execution::FunctionBindingOrigin>,
    // One exact superclass-body dispatch, never an ambient constructor mode.
    retained_super_body: Option<(
        runmat_types::ClassMethodOwner,
        runmat_value::BoundFunctionBinding,
    )>,
    // A typed invocation-lineage fact, propagated by context clones. It is
    // never inferred from a class name, TLS depth, or a session-global flag.
    pub(super) native_invocation_lineage: Option<Rc<()>>,
    /// Invocation-local logical MEX callback ancestry, never a count of
    /// suspended sibling callbacks or a session-global reentry flag.
    #[cfg(not(target_arch = "wasm32"))]
    mex_callback_ancestry: crate::foreign::MexCallbackAncestry,
    #[cfg(not(target_arch = "wasm32"))]
    python_reentry: Option<
        Rc<super::foreign_callback_lineage::CallbackLineage<runmat_python::PythonReentryAuthority>>,
    >,
    #[cfg(not(target_arch = "wasm32"))]
    java_reentry: Option<
        Rc<super::foreign_callback_lineage::CallbackLineage<runmat_java::JavaReentryAuthority>>,
    >,
    // VM-owned, poll-local execution evidence. Clones may carry it only within
    // the selected instruction; nested invocations explicitly clear it.
    vm_fusion_host_route: Option<super::VmFusionHostRoute>,
    native_region_host_route: Option<super::NativeRegionHostRoute>,
    prepared_host_execution: Option<super::preparation::PreparedHostExecution>,
    vm_fusion_selected_binding: Option<runmat_builtins::BuiltinBindingIdentity>,
    // Invocation-scoped, unlike shared session language state. Scoped guards
    // affect this invocation's clones, never an earlier/later generation.
    cancellation: Rc<RefCell<CancellationToken>>,
}

impl std::fmt::Debug for RuntimeContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeContext")
            .field("scope_id", &self.execution.scope_id())
            .field("program_revision", &self.program_revision)
            .field("services", &self.services)
            .finish_non_exhaustive()
    }
}

impl RuntimeContext {
    pub fn with_acceleration_policy(mut self, policy: super::RuntimeAccelerationPolicy) -> Self {
        self.acceleration_policy = Some(policy);
        self
    }

    pub fn acceleration_policy(&self) -> Option<super::RuntimeAccelerationPolicy> {
        self.acceleration_policy
    }
    pub(crate) fn with_graphics_callback_mutation(
        mut self,
        scope: Option<super::GraphicsCallbackMutationScope>,
    ) -> Self {
        self.graphics_callback_mutation = scope;
        self
    }

    #[cfg(feature = "plot-core")]
    pub(crate) fn graphics_callback_mutation(
        &self,
    ) -> Option<super::GraphicsCallbackMutationScope> {
        self.graphics_callback_mutation.clone()
    }
    pub fn new(execution: Rc<dyn RuntimeExecutionServices>) -> Self {
        Self::with_cancellation(execution, CancellationSource::new().token())
    }

    pub fn with_cancellation(
        execution: Rc<dyn RuntimeExecutionServices>,
        cancellation: CancellationToken,
    ) -> Self {
        Self::with_environment_state(
            execution,
            cancellation,
            super::environment::EnvironmentState::from_host(),
        )
    }

    pub(super) fn with_environment_state(
        execution: Rc<dyn RuntimeExecutionServices>,
        cancellation: CancellationToken,
        environment: super::environment::EnvironmentState,
    ) -> Self {
        Self::with_environment_and_services(
            execution,
            cancellation,
            environment,
            RuntimeServicePorts::default(),
        )
    }

    pub(super) fn with_environment_and_services(
        execution: Rc<dyn RuntimeExecutionServices>,
        cancellation: CancellationToken,
        environment: super::environment::EnvironmentState,
        services: RuntimeServicePorts,
    ) -> Self {
        runmat_accelerate_api::register_execution_provider_authorizer(
            super::scope::execution_provider_authorization,
        );
        runmat_accelerate_api::register_device_allocation_scope_resolver(
            super::gpu_device::current_device_allocation_scope,
        );
        runmat_filesystem::register_invocation_filesystem_resolver(selected_invocation_filesystem)
            .expect("Runtime filesystem selection callback must be unique");
        let state = Rc::new(RuntimeContextState::new(RuntimeContextLocalId::next()));
        crate::class_registry::register_context_state(&state);
        let context = Self {
            cleanup_completion: None,
            graphics_callback_mutation: None,
            execution,
            environment: Rc::new(std::cell::RefCell::new(environment)),
            selected_graph_program: None,
            selected_graph_function_resolver: None,
            selected_static_site_resolver: None,
            selected_static_caller: None,
            native_resolution_scope: crate::user_functions::NativeResolutionScope::Session,
            program_revision: None,
            program_sources: None,
            source_current: None,
            source_selection: InvocationSourceSelection::Inherited,
            source_catalog_view: None,
            debug_projection: super::poll_projection::PollProjection::legacy(Default::default()),
            output_projection: super::poll_projection::PollProjection::legacy(Default::default()),
            call_projection: super::poll_projection::PollProjection::legacy(Default::default()),
            call_projection_lineage: Rc::new(()),
            console_capture_projection: super::poll_projection::PollProjection::legacy(
                Default::default(),
            ),
            #[cfg(feature = "plot-core")]
            plot_receipts: Default::default(),
            command_profiling: None,
            command_device_provider: None,
            command_device_access: None,
            tape_poll_authority: None,
            search_path: None,
            services,
            state,
            parallel_random_scope: None,
            invocation_language_policy: None,
            acceleration_policy: None,
            invocation_arity: None,
            constructor_receiver: None,
            selected_classes: None,
            selected_object_class: None,
            selected_method_body: None,
            selected_method_frame: None,
            retained_super_constructors: None,
            retained_super_caller_origin: None,
            retained_super_body: None,
            native_invocation_lineage: None,
            #[cfg(not(target_arch = "wasm32"))]
            mex_callback_ancestry: crate::foreign::MexCallbackAncestry::ROOT,
            #[cfg(not(target_arch = "wasm32"))]
            python_reentry: None,
            #[cfg(not(target_arch = "wasm32"))]
            java_reentry: None,
            vm_fusion_host_route: None,
            native_region_host_route: None,
            prepared_host_execution: None,
            vm_fusion_selected_binding: None,
            cancellation: Rc::new(RefCell::new(cancellation)),
        };
        {
            let _guard = context.enter();
            crate::builtins::deep_learning::model::ensure_network_classes_registered();
        }
        context
    }

    pub fn execution(&self) -> &Rc<dyn RuntimeExecutionServices> {
        &self.execution
    }

    pub fn with_parallel_random_scope(mut self) -> crate::BuiltinResult<Self> {
        self.parallel_random_scope = Some(
            crate::builtins::common::random::ParallelRandomScope::new(&self)?,
        );
        Ok(self)
    }

    pub(crate) fn parallel_random_scope(
        &self,
    ) -> Option<&Rc<crate::builtins::common::random::ParallelRandomScope>> {
        self.parallel_random_scope.as_ref()
    }

    pub fn with_selected_graph_program(
        mut self,
        program: crate::execution::DeferredGraphProgram,
    ) -> Self {
        self.selected_graph_program = Some(program);
        self.selected_graph_function_resolver = None;
        self.selected_static_site_resolver = None;
        self.selected_static_caller = None;
        self
    }

    /// An admitted nested product must not inherit its caller's selected V3
    /// graph authority while its own exact product is being bound.
    pub fn without_selected_graph_program(mut self) -> Self {
        self.selected_graph_program = None;
        self.selected_graph_function_resolver = None;
        self.selected_static_site_resolver = None;
        self.selected_static_caller = None;
        self
    }

    pub fn with_selected_native_route(
        mut self,
        route: Option<Rc<crate::user_functions::SelectedNativeRoute>>,
    ) -> Self {
        self.native_resolution_scope = match route {
            Some(route) => crate::user_functions::NativeResolutionScope::Selected(route),
            None => crate::user_functions::NativeResolutionScope::SelectedClosed,
        };
        self
    }

    pub fn native_resolution_scope(&self) -> &crate::user_functions::NativeResolutionScope {
        &self.native_resolution_scope
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_ordinary_selected_native_route(
        mut self,
        route: Rc<crate::user_functions::SelectedNativeRoute>,
        mex: Rc<crate::foreign::MexRuntimeSession>,
        source_owner: crate::source_context::SelectedProgramSourceOwner,
    ) -> Self {
        self.native_resolution_scope =
            crate::user_functions::NativeResolutionScope::SelectedOrdinary {
                route,
                mex,
                source_owner,
            };
        self
    }

    /// Install a short-lived, already-admitted site resolver for one selected
    /// V3 unit. It is carried by this invocation context, never by bytecode or
    /// a durable definition product.
    pub fn with_selected_static_site_resolver(
        mut self,
        resolver: Rc<SelectedStaticSiteResolver>,
    ) -> Self {
        self.selected_static_site_resolver = Some(resolver);
        self.selected_static_caller = None;
        self
    }

    /// The VM sets this from the exact body it entered; None denotes the
    /// selected script stream, not an arbitrary unnamed function.
    pub fn with_selected_static_caller(mut self, caller: Option<FunctionBindingOrigin>) -> Self {
        self.selected_static_caller = caller;
        self
    }

    pub fn selected_static_caller(&self) -> Option<FunctionBindingOrigin> {
        self.selected_static_caller
    }

    pub fn selected_static_binding(
        &self,
        site: GraphStaticReferenceSite,
    ) -> Result<Option<runmat_value::BoundFunctionBinding>, &'static str> {
        let Some(resolve) = self.selected_static_site_resolver.as_ref() else {
            return Ok(None);
        };
        resolve(self.selected_static_caller, site)
    }

    pub fn has_selected_static_site_resolver(&self) -> bool {
        self.selected_static_site_resolver.is_some()
    }

    pub fn selected_graph_program(&self) -> Option<&crate::execution::DeferredGraphProgram> {
        self.selected_graph_program.as_ref()
    }

    /// Bind Native MIR function IDs through the exact selected registry for
    /// this invocation. Installing a different selected program clears this
    /// resolver, preventing a nested unit from inheriting caller authority.
    pub fn with_selected_graph_function_resolver(
        mut self,
        resolver: Rc<SelectedGraphFunctionResolver>,
    ) -> Self {
        if self.selected_graph_program.is_some() {
            self.selected_graph_function_resolver = Some(resolver);
        }
        self
    }

    pub fn selected_graph_function_binding(
        &self,
        caller: runmat_types::FunctionId,
        identity: &runmat_types::CallableIdentity,
    ) -> Option<SelectedGraphFunction> {
        self.selected_graph_program.as_ref()?;
        self.selected_graph_function_resolver.as_ref()?(caller, identity)
    }

    pub fn has_selected_graph_function_resolver(&self) -> bool {
        self.selected_graph_function_resolver.is_some()
    }

    pub fn service_ports(&self) -> &RuntimeServicePorts {
        &self.services
    }

    /// Publish application-pipe text using this caller's console and allocation
    /// authority. Process owners pass bytes to the caller; they must not retain
    /// a context in an execution service or publish from its worker thread.
    pub fn record_process_console(
        &self,
        stream: crate::console::ConsoleStream,
        text: &str,
    ) -> Result<(), crate::RuntimeError> {
        super::with_original_scope(Some(self), || {
            crate::console::record_console_output(stream, text)
        })?;
        if let Some(host) = self.service_ports().host() {
            host.check_execution()?;
        }
        Ok(())
    }

    /// This restricted view can publish console text, not dispatch source or
    /// foreign calls. Removing its foreign edge must not invalidate the actual
    /// command's device-route evidence as a service reconfiguration would.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn console_scope(&self) -> super::RuntimeConsoleScope {
        let mut context = self.clone();
        context.services = context.services.without_foreign();
        super::RuntimeConsoleScope::new(context)
    }

    /// Capture the typed constructor instance at call admission. The active
    /// receiver stack is poll-local, while a semantic child may be scheduled
    /// for a later poll. Ordinary methods explicitly clear any inherited seed.
    pub fn with_constructor_receiver_for_method(
        mut self,
        owner: Option<&runmat_types::ClassMethodOwner>,
    ) -> Result<Self, crate::RuntimeError> {
        let Some(owner) = owner else {
            self.constructor_receiver = None;
            return Ok(self);
        };
        if owner.role != runmat_types::ClassMethodRole::Constructor {
            self.constructor_receiver = None;
            return Ok(self);
        }
        self.constructor_receiver = self.capture_constructor_receiver(owner)?;
        Ok(self)
    }

    fn capture_constructor_receiver(
        &self,
        owner: &runmat_types::ClassMethodOwner,
    ) -> Result<Option<crate::execution::RootedValue>, crate::RuntimeError> {
        if owner.is_static {
            return Err(crate::build_runtime_error(
                "a static method cannot own the constructor instance",
            )
            .with_identifier("RunMat:ConstructorOwnerInvalid")
            .build());
        }
        if let Some(receiver) = crate::constructor_receiver_in_context(self, &owner.declaring_class)
        {
            Ok(Some(
                crate::execution::RootedValue::new(receiver)
                    .map_err(|error| error.into_runtime_error())?,
            ))
        } else {
            Ok(self
                .constructor_receiver
                .as_ref()
                .filter(|rooted| {
                    crate::constructor_receiver_matches_class(
                        rooted.value(),
                        &owner.declaring_class,
                    )
                })
                .cloned())
        }
    }

    /// The call-local constructor instance remains rooted by this context.
    pub fn captured_constructor_receiver(&self) -> Option<&runmat_value::Value> {
        self.constructor_receiver
            .as_ref()
            .map(|owner| owner.value())
    }

    /// Select one already-admitted class product for this invocation only.
    /// Sibling futures may retain different revisions without changing the
    /// session's ordinary name-resolved class registry.
    pub fn with_selected_class_projection(
        mut self,
        projection: Rc<crate::class_registry::SelectedClassProjection>,
    ) -> Self {
        self.selected_classes = Some(projection);
        self
    }

    pub fn with_selected_object_class(
        mut self,
        selected: Rc<runmat_value::SelectedObjectClassFacts>,
    ) -> Self {
        self.selected_object_class = Some(selected);
        self
    }

    pub fn selected_object_class(&self) -> Option<&Rc<runmat_value::SelectedObjectClassFacts>> {
        self.selected_object_class.as_ref()
    }

    /// A nested ordinary invocation does not inherit its caller's receiver
    /// facts or one-shot method intent. Selected invocations install their own
    /// checked product before entering the body.
    pub fn without_selected_object_class(mut self) -> Self {
        self.selected_classes = None;
        self.selected_object_class = None;
        self.selected_method_body = None;
        self.selected_method_frame = None;
        self
    }

    pub(crate) fn selected_class_projection(
        &self,
    ) -> Option<&Rc<crate::class_registry::SelectedClassProjection>> {
        self.selected_classes.as_ref()
    }

    pub fn with_retained_super_constructors(
        mut self,
        bindings: Vec<super::RetainedSuperConstructorBinding>,
        caller_origin: runmat_execution::FunctionBindingOrigin,
    ) -> Result<Self, crate::RuntimeError> {
        for (index, binding) in bindings.iter().enumerate() {
            let valid_target = binding.target.is_admitted_in(&self);
            if !valid_target
                || binding.target.parent() == &binding.requesting_class
                || binding.caller_origin != caller_origin
                || bindings[..index].iter().any(|previous| {
                    previous.requesting_class == binding.requesting_class
                        && previous.target.parent() == binding.target.parent()
                        && previous.caller_origin == binding.caller_origin
                })
            {
                return Err(crate::build_runtime_error(
                    "retained super constructor has invalid or ambiguous authority",
                )
                .with_identifier("RunMat:ConstructorOwnerInvalid")
                .build());
            }
        }
        self.retained_super_constructors = Some(Rc::from(bindings));
        self.retained_super_caller_origin = Some(caller_origin);
        Ok(self)
    }

    /// Select the calling definition, not a same-named later class product.
    pub fn with_retained_super_caller_origin(
        mut self,
        origin: runmat_execution::FunctionBindingOrigin,
    ) -> Self {
        self.retained_super_caller_origin = Some(origin);
        self
    }

    pub(crate) fn with_retained_super_body(
        mut self,
        owner: runmat_types::ClassMethodOwner,
        binding: runmat_value::BoundFunctionBinding,
    ) -> Self {
        self.retained_super_body = Some((owner, binding));
        self
    }

    /// Consume the exact superclass-body intent before entering a child
    /// executor. A sibling or nested ordinary constructor cannot inherit it.
    pub fn take_retained_super_body_for(
        mut self,
        binding: &runmat_value::BoundFunctionBinding,
    ) -> Result<(Self, Option<runmat_types::ClassMethodOwner>), crate::RuntimeError> {
        match self.retained_super_body.take() {
            None => Ok((self, None)),
            Some((owner, expected)) if &expected == binding => Ok((self, Some(owner))),
            Some(_) => Err(crate::build_runtime_error(
                "superclass-body dispatch selected a different definition",
            )
            .with_identifier("RunMat:ConstructorOwnerInvalid")
            .build()),
        }
    }

    pub(crate) fn retained_super_constructor(
        &self,
        requesting_class: &runmat_types::ClassIdentity,
        super_class: &runmat_types::ClassIdentity,
    ) -> Result<Option<&super::RetainedSuperConstructorBinding>, crate::RuntimeError> {
        let Some(bindings) = &self.retained_super_constructors else {
            return Ok(None);
        };
        let selected = self.retained_super_caller_origin.and_then(|caller_origin| {
            bindings
                .iter()
                .find(|binding| binding.matches(caller_origin, requesting_class, super_class))
        });
        if selected.is_none() {
            return Err(crate::build_runtime_error(
                "retained superclass call has no exact admitted target for its caller",
            )
            .with_identifier("RunMat:ConstructorOwnerInvalid")
            .build());
        }
        Ok(selected)
    }

    /// Numeric presentation policy shared by this session's invocation frames.
    pub fn display_format_owner(&self) -> &runmat_value::DisplayFormatOwner {
        &self.state.display_format
    }

    /// Start a command with its own figure-change receipt. Clones made by
    /// nested calls share it; previously selected callbacks retain theirs.
    pub fn with_fresh_plot_receipts(self) -> Self {
        #[cfg(feature = "plot-core")]
        {
            let mut context = self;
            context.plot_receipts = Default::default();
            context
        }
        #[cfg(not(feature = "plot-core"))]
        {
            self
        }
    }

    pub fn with_command_profiling(mut self, receipt: CommandProfilingReceipt) -> Self {
        self.command_profiling = Some(receipt);
        self
    }

    pub fn with_command_device_provider(
        mut self,
        provider: Option<&'static dyn runmat_accelerate_api::AccelProvider>,
    ) -> Self {
        self.command_device_provider = provider;
        self.command_device_access = provider
            .and_then(|provider| provider.wgpu_device_access())
            .cloned();
        self
    }

    pub fn command_device_provider(
        &self,
    ) -> Option<&'static dyn runmat_accelerate_api::AccelProvider> {
        self.command_device_provider
    }

    pub fn command_device_access(&self) -> Option<&runmat_accelerate_api::WgpuDeviceAccess> {
        self.command_device_access.as_ref()
    }

    /// Select an exact queue-admission origin, including an explicit unknown
    /// origin that must mask any later drainer's command.
    pub(crate) fn with_cleanup_command_origin(
        mut self,
        origin: Option<CommandProfilingReceipt>,
    ) -> Self {
        self.command_profiling = origin;
        self
    }

    pub(crate) fn with_cleanup_completion(
        mut self,
        completion: Option<
            crate::builtins::introspection::on_cleanup::lifecycle::registry::CleanupCompletionId,
        >,
    ) -> Self {
        self.cleanup_completion = completion;
        self
    }

    pub(crate) fn cleanup_completion(
        &self,
    ) -> Option<crate::builtins::introspection::on_cleanup::lifecycle::registry::CleanupCompletionId>
    {
        self.cleanup_completion
    }

    pub fn command_profiling(&self) -> Option<&CommandProfilingReceipt> {
        self.command_profiling.as_ref()
    }

    #[cfg(feature = "plot-core")]
    pub(crate) fn plot_receipts(&self) -> &crate::builtins::plotting::state::FigureReceiptOwner {
        &self.plot_receipts
    }

    pub(crate) fn state(&self) -> &Rc<RuntimeContextState> {
        &self.state
    }

    pub(crate) fn cleanup_registry(
        &self,
    ) -> Option<Rc<crate::builtins::introspection::on_cleanup::lifecycle::registry::CleanupRegistry>>
    {
        self.state
            .cleanup_registry
            .try_borrow()
            .ok()?
            .as_ref()?
            .upgrade()
    }

    /// Read one value from this exact session's global namespace without
    /// relying on ambient runtime activation.
    pub fn session_global_value(
        &self,
        name: &str,
    ) -> Result<Option<runmat_value::Value>, crate::RuntimeError> {
        self.state
            .session_variables
            .try_borrow()
            .map(|variables| variables.global_value(name))
            .map_err(|error| {
                crate::runtime_error::semantic_error(
                    "RunMat:Workspace:GlobalBorrow",
                    format!("session-global workspace read is reentrant: {error}"),
                )
            })
    }

    /// Read a legacy slot from this selected session, not from ambient state
    /// or a coincidentally named `var_N` global.
    pub fn session_global_slot_value(
        &self,
        slot: usize,
    ) -> Result<Option<runmat_value::Value>, crate::RuntimeError> {
        self.session_variable_authority().global_slot(slot)
    }

    /// Enumerate this exact session's global namespace without relying on
    /// ambient runtime activation.
    pub fn session_global_names(&self) -> Result<Vec<String>, crate::RuntimeError> {
        self.state
            .session_variables
            .try_borrow()
            .map(|variables| variables.global_names())
            .map_err(|error| {
                crate::runtime_error::semantic_error(
                    "RunMat:Workspace:GlobalBorrow",
                    format!("session-global workspace read is reentrant: {error}"),
                )
            })
    }

    pub fn session_variable_authority(
        &self,
    ) -> crate::workspace::session::SessionVariableAuthority {
        crate::workspace::session::SessionVariableAuthority::new(&self.state)
    }

    pub(super) fn state_identity(&self) -> *const RuntimeContextState {
        Rc::as_ptr(&self.state)
    }

    pub fn local_identity(&self) -> RuntimeContextLocalId {
        self.state.local_identity
    }
    pub(crate) fn gpu_device_session_key(&self) -> u64 {
        self.state.local_identity.0
    }

    pub(in crate::context) fn capture_host_allocation_context(
        &self,
    ) -> Option<super::host_allocation::HostAllocationContext> {
        Some(super::host_allocation::HostAllocationContext::new(
            Rc::clone(&self.state),
            Rc::clone(&self.cancellation),
            Rc::clone(self.services.parallel()?),
        ))
    }

    pub(in crate::context) fn capture_host_effect_stage_authority(
        &self,
    ) -> super::host_effect::HostEffectStageAuthority {
        super::host_effect::HostEffectStageAuthority::new(
            Rc::clone(&self.state),
            Rc::clone(&self.cancellation),
            Rc::clone(&self.execution),
            self.program_revision.clone(),
            self.services.placement().cloned(),
            self.services.host_effect_resources().cloned(),
            self.services.parallel().cloned(),
        )
    }

    /// Process-local, non-serialized authority for this session's lifetime.
    /// Values and foreign registries must retain only a weak copy.
    pub(crate) fn session_lease(&self) -> std::sync::Weak<()> {
        Arc::downgrade(&self.state.session_lease)
    }

    #[cfg(feature = "plot-core")]
    pub(crate) fn retain_plot_callback(
        &self,
        slot: u64,
        root: crate::builtins::plotting::state::callbacks::PlotCallbackRoot,
    ) {
        let (retired, replaced) = {
            let mut callbacks = self.state.plot_callbacks.borrow_mut();
            let retired = crate::builtins::plotting::state::callbacks::take_revoked(&mut callbacks);
            let replaced = callbacks.insert(slot, root);
            (retired, replaced)
        };
        drop(replaced);
        drop(retired);
    }

    #[cfg(feature = "plot-core")]
    pub(crate) fn plot_callback(&self, slot: u64) -> Option<runmat_value::Value> {
        let (retired, value) = {
            let mut callbacks = self.state.plot_callbacks.borrow_mut();
            let retired = crate::builtins::plotting::state::callbacks::take_revoked(&mut callbacks);
            let value = callbacks.get(&slot).map(|root| root.value.clone());
            (retired, value)
        };
        drop(retired);
        value
    }

    #[cfg(all(test, feature = "plot-core"))]
    pub(crate) fn plot_callback_root_count(&self) -> usize {
        let (retired, count) = {
            let mut callbacks = self.state.plot_callbacks.borrow_mut();
            let retired = crate::builtins::plotting::state::callbacks::take_revoked(&mut callbacks);
            (retired, callbacks.len())
        };
        drop(retired);
        count
    }

    #[cfg(feature = "plot-core")]
    pub(crate) fn prepare_plot_behavior(&self, slot: u64) -> crate::BuiltinResult<Option<Self>> {
        let authority = {
            let callbacks = self.state.plot_callbacks.borrow();
            let root = callbacks
                .get(&slot)
                .filter(|root| root.live.load(std::sync::atomic::Ordering::Acquire))
                .ok_or_else(|| {
                    crate::build_runtime_error("graphics callback root has retired")
                        .with_identifier("RunMat:graphics:CallbackSourceRetired")
                        .build()
                })?;
            root.behavior_authority.clone()
        };
        authority.map(|authority| authority(self)).transpose()
    }

    #[cfg(feature = "plot-core")]
    pub(crate) fn plot_behavior_source(&self, slot: u64) -> Option<String> {
        let callbacks = self.state.plot_callbacks.borrow();
        let root = callbacks.get(&slot)?;
        match &root.value {
            runmat_value::Value::String(text) => text.as_str().map(str::to_owned),
            // Optional diagnostic source is not executable authority.
            runmat_value::Value::CharArray(chars) => chars.try_row_string().unwrap_or_default(),
            _ => None,
        }
    }

    /// Invalidate only this session's staged host-effect invocations when its
    /// installed resource authority changes. Shared ledger owners remain
    /// usable by other sessions and retain prior physical output charges.
    pub fn invalidate_host_effect_session_authority(&self) {
        self.state
            .host_effect_generation
            .fetch_add(1, Ordering::AcqRel);
    }

    pub(crate) fn host_effect_session_generation(&self) -> (Arc<AtomicU64>, u64) {
        let owner = self.state.host_effect_generation.clone();
        let generation = owner.load(Ordering::Acquire);
        (owner, generation)
    }

    pub(crate) fn register_session_resource_cleanup(&self, cleanup: impl FnOnce() + 'static) {
        self.state
            .resource_cleanup
            .0
            .borrow_mut()
            .push(Box::new(cleanup));
    }

    /// A standalone worker invocation owns newly created, untransferred
    /// resources. Ordinary interactive sessions do not select this policy.
    #[cfg(target_arch = "wasm32")]
    pub fn enable_worker_invocation_resources(&self) {
        self.state.worker_invocation_resources.set(true);
    }

    /// Non-owning proof check for the fresh invocation's context and execution
    /// roots. Evaluate after dropping every invocation-owned strong handle.
    #[cfg(target_arch = "wasm32")]
    pub fn worker_invocation_retirement_check(&self) -> impl Fn() -> bool {
        let state = Rc::downgrade(&self.state);
        let execution = Rc::downgrade(&self.execution);
        move || state.strong_count() == 0 && execution.strong_count() == 0
    }

    #[cfg(target_arch = "wasm32")]
    pub fn collect_retired_worker_values() -> Result<(), crate::RuntimeError> {
        runmat_gc::gc_collect_major().map(|_| ()).map_err(|error| {
            crate::build_runtime_error("worker GC retirement failed")
                .with_identifier("RunMat:RuntimeContext:WorkerGcRetirement")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .with_source(error)
                .build()
        })
    }

    /// Call only after foreground execution, cleanup callbacks and local child
    /// frames have retired. The same registered owners also run on state drop.
    #[cfg(target_arch = "wasm32")]
    pub fn retire_worker_invocation_resources(&self) {
        if !self.state.worker_invocation_resources.get() {
            return;
        }
        loop {
            let cleanups = std::mem::take(&mut *self.state.resource_cleanup.0.borrow_mut());
            if cleanups.is_empty() {
                break;
            }
            for cleanup in cleanups {
                cleanup();
            }
        }
    }

    pub fn cancellation(&self) -> CancellationToken {
        self.cancellation.borrow().clone()
    }

    /// Bind lifecycle cancellation until the returned scope retires. This
    /// shares only the token slot, not workspace values or context projections.
    pub fn scoped_cancellation_token(
        &self,
        token: CancellationToken,
    ) -> super::RuntimeCancellationScope {
        super::RuntimeCancellationScope::new(Rc::clone(&self.cancellation), token)
    }

    pub(crate) fn cancellation_context(&self) -> super::RuntimeCancellationContext {
        super::RuntimeCancellationContext::new(self.cancellation(), self.state.clone())
    }

    /// Project the selected call's arity without changing sibling futures or
    /// the parent invocation. The VM/native caller owns the actual counts.
    pub fn with_invocation_arity(mut self, arity: RuntimeInvocationArity) -> Self {
        self.invocation_arity = Some(arity);
        self
    }

    pub fn invocation_arity(&self) -> Option<RuntimeInvocationArity> {
        self.invocation_arity
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn mex_callback_ancestry(&self) -> crate::foreign::MexCallbackAncestry {
        self.mex_callback_ancestry
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn with_mex_callback_ancestry(
        mut self,
        ancestry: crate::foreign::MexCallbackAncestry,
    ) -> Self {
        self.mex_callback_ancestry = ancestry;
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn python_reentry_for(
        &self,
        session: &runmat_python::PythonSession,
    ) -> Option<&runmat_python::PythonReentryAuthority> {
        self.python_reentry.as_deref().and_then(|lineage| {
            lineage.newest_for(|authority| session.owns_reentry_authority(authority))
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn with_python_reentry(
        mut self,
        authority: runmat_python::PythonReentryAuthority,
    ) -> Self {
        self.python_reentry = Some(super::foreign_callback_lineage::CallbackLineage::push(
            self.python_reentry.take(),
            authority,
        ));
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn java_reentry_for(
        &self,
        lane: &runmat_java::JavaLane,
    ) -> Option<&runmat_java::JavaReentryAuthority> {
        self.java_reentry.as_deref().and_then(|lineage| {
            lineage.newest_for(|authority| lane.owns_reentry_authority(authority))
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn with_java_reentry(
        mut self,
        authority: runmat_java::JavaReentryAuthority,
    ) -> Self {
        self.java_reentry = Some(super::foreign_callback_lineage::CallbackLineage::push(
            self.java_reentry.take(),
            authority,
        ));
        self
    }

    /// Carry only callback ancestry into a newly allocated child session.
    /// The child keeps its own execution and cancellation owners; this is not
    /// the same-owner invocation projection used for captured callback targets.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn try_inherit_foreign_callback_lineage_for_child(
        mut self,
        parent: &RuntimeContext,
    ) -> Option<Self> {
        if self.local_identity() == parent.local_identity()
            || Rc::ptr_eq(&self.execution, &parent.execution)
            || Rc::ptr_eq(&self.cancellation, &parent.cancellation)
        {
            return None;
        }
        self.mex_callback_ancestry = parent.mex_callback_ancestry;
        self.python_reentry = parent.python_reentry.clone();
        self.java_reentry = parent.java_reentry.clone();
        Some(self)
    }

    /// Project the current command's revocable callback and execution facts
    /// onto a captured target. The target retains its code, source, policy,
    /// foreign and base-workspace ports, and language owner. The workspace
    /// frame belongs to the physical caller invocation, not the registration:
    /// a callback registered in an earlier frame must not reenter that frame
    /// after it closes. An older registration must not reuse stale
    /// cancellation, profiling, device or native-placement authority.
    pub fn try_with_inherited_callback_invocation_authority_from(
        mut self,
        parent: &RuntimeContext,
    ) -> Option<Self> {
        if self.local_identity() != parent.local_identity() {
            return None;
        }
        self.services = match parent.services.workspace() {
            Some(workspace) => self.services.with_workspace(Rc::clone(workspace)),
            None => self.services.without_workspace(),
        };
        self.execution = Rc::clone(&parent.execution);
        self.cancellation = Rc::clone(&parent.cancellation);
        // Code and binding owners belong to the retained target; the live
        // environment and debug ancestry belong to its physical caller.
        self.environment = Rc::clone(&parent.environment);
        self.debug_projection = parent.debug_projection.clone();
        self.command_profiling = parent.command_profiling.clone();
        // Differentiation belongs to the physical callback invocation. A
        // retained target must neither reuse a completed tape nor lose the
        // current one when the VM polls it as a separate continuation.
        self.tape_poll_authority = parent.tape_poll_authority.clone();
        if self.services.same_device_route_owners(&parent.services) {
            self.command_device_provider = parent.command_device_provider;
            self.command_device_access = parent.command_device_access.clone();
        } else {
            if let Some(owner) = &self.command_profiling {
                owner.mark_unproven_device_route();
            }
            self.command_device_provider = None;
            self.command_device_access = None;
        }
        self.native_invocation_lineage = parent.native_invocation_lineage.clone();
        self.invocation_arity = None;
        self.prepared_host_execution = None;
        self.vm_fusion_host_route = None;
        self.native_region_host_route = None;
        self.vm_fusion_selected_binding = None;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.mex_callback_ancestry = parent.mex_callback_ancestry;
            self.python_reentry = parent.python_reentry.clone();
            self.java_reentry = parent.java_reentry.clone();
        }
        Some(self)
    }

    /// Seed a checked logical MEX ancestry in host integration tests. The
    /// ordinary invocation and wire paths cannot construct an unchecked fact.
    #[cfg(all(feature = "interaction-test-hooks", not(target_arch = "wasm32")))]
    #[doc(hidden)]
    pub fn with_test_mex_callback_ancestry_depth(mut self, depth: u16) -> Option<Self> {
        self.mex_callback_ancestry = crate::foreign::MexCallbackAncestry::from_wire(depth)?;
        Some(self)
    }

    /// Create work accounting for this invocation generation. The adapter
    /// owns the quantum and the actual scheduling primitive.
    pub fn cpu_cooperation(&self) -> crate::execution::CpuCooperationBudget {
        crate::execution::CpuCooperationBudget::new(self.execution.clone(), self.cancellation())
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.borrow().is_cancelled()
    }

    /// Retain this exact invocation generation for ongoing work. Unlike a
    /// fresh invocation, all selected host/VM/native projections stay intact.
    /// The independent slot cannot observe a caller's later lifecycle rebind.
    pub(crate) fn with_pinned_cancellation(mut self) -> Self {
        let cancellation = self.cancellation();
        self.cancellation = Rc::new(RefCell::new(cancellation));
        self
    }

    /// Bind a fresh invocation generation while retaining the session's
    /// language state and services. Previously cloned contexts stay bound to
    /// their own invocation generation.
    pub fn with_cancellation_token(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = Rc::new(RefCell::new(cancellation));
        self.prepared_host_execution = None;
        self.vm_fusion_host_route = None;
        self.native_region_host_route = None;
        self.vm_fusion_selected_binding = None;
        self
    }

    /// Install an exact VM-selected host route on one instruction's context.
    /// It never changes provider availability or a separate invocation.
    pub fn with_vm_fusion_host_route(mut self, route: Option<super::VmFusionHostRoute>) -> Self {
        self.prepared_host_execution = None;
        self.vm_fusion_host_route = route;
        self.native_region_host_route = None;
        self.vm_fusion_selected_binding = None;
        self
    }

    pub fn without_vm_fusion_host_route(mut self) -> Self {
        self.prepared_host_execution = None;
        self.vm_fusion_host_route = None;
        self.native_region_host_route = None;
        self.vm_fusion_selected_binding = None;
        self
    }

    pub fn vm_fusion_host_route(&self) -> Option<&super::VmFusionHostRoute> {
        self.vm_fusion_host_route.as_ref()
    }

    /// Scope one selected native occurrence, never an ambient host policy.
    pub fn with_native_region_host_route(mut self, route: super::NativeRegionHostRoute) -> Self {
        self.prepared_host_execution = None;
        self.vm_fusion_host_route = None;
        self.vm_fusion_selected_binding = None;
        self.native_region_host_route = Some(route);
        self
    }

    pub fn native_region_host_route(&self) -> Option<&super::NativeRegionHostRoute> {
        self.native_region_host_route.as_ref()
    }

    pub(crate) fn with_prepared_host_execution(
        mut self,
        proof: Option<super::preparation::PreparedHostExecution>,
    ) -> Self {
        self.prepared_host_execution = proof;
        self
    }

    pub(crate) fn prepared_host_execution(
        &self,
    ) -> Option<&super::preparation::PreparedHostExecution> {
        self.prepared_host_execution.as_ref()
    }

    /// The dispatcher projects its actual selected canonical binding into
    /// the call context. Legacy invocations never inherit a sibling identity.
    pub(crate) fn with_vm_fusion_selected_binding(
        mut self,
        binding: Option<runmat_builtins::BuiltinBindingIdentity>,
    ) -> Self {
        self.prepared_host_execution = None;
        if self.vm_fusion_selected_binding.is_some() {
            // An inner canonical invocation may select the same binding, but
            // it cannot certify the enclosing VM instruction's host branch.
            self.vm_fusion_host_route = None;
            self.native_region_host_route = None;
        }
        self.vm_fusion_selected_binding = binding;
        self
    }

    pub(crate) fn vm_fusion_selected_binding(
        &self,
    ) -> Option<runmat_builtins::BuiltinBindingIdentity> {
        self.vm_fusion_selected_binding
    }

    pub(crate) fn replace_cancellation_token(
        &self,
        cancellation: CancellationToken,
    ) -> CancellationToken {
        self.cancellation.replace(cancellation)
    }

    pub fn program_revision(&self) -> Option<&runmat_execution::ProgramRevision> {
        self.program_revision.as_deref()
    }

    pub(in crate::context) fn program_revision_owner(
        &self,
    ) -> Option<&Arc<runmat_execution::ProgramRevision>> {
        self.program_revision.as_ref()
    }

    /// Issue session-local source leases for a complete immutable program.
    /// No caller-provided numeric source identity is imported into the issuer.
    pub fn with_program_sources(
        self,
        records: impl IntoIterator<Item = crate::source_context::ProgramSourceRecord>,
    ) -> Result<Self, crate::source_context::ProgramSourceBindingError> {
        let binding = crate::source_context::bind_program_sources(&self, records)?;
        Ok(self.with_bound_program_sources(binding))
    }

    /// Project one already-issued source into a new selected invocation
    /// without allocating a second session source identity.
    pub fn with_existing_program_source(
        self,
        ordinal: runmat_types::ProgramSourceId,
        lease: crate::source_context::RuntimeSourceLease,
    ) -> Result<Self, crate::source_context::ProgramSourceBindingError> {
        let binding = crate::source_context::bind_existing_program_source(&self, ordinal, lease)?;
        Ok(self.with_bound_program_sources(binding))
    }

    fn with_bound_program_sources(
        self,
        binding: crate::source_context::ProgramSourceBinding,
    ) -> Self {
        self.with_selected_program_source_binding(Rc::new(binding))
    }

    /// Reuse an already-issued, admitted source table for an exact selected
    /// invocation or its orderly cleanup. This never allocates source IDs.
    pub(crate) fn with_selected_program_source_binding(
        mut self,
        binding: Rc<crate::source_context::ProgramSourceBinding>,
    ) -> Self {
        self.program_sources = Some(binding);
        self.source_current = Some(super::poll_projection::PollProjection::legacy(
            crate::source_context::SourceCurrentProjection::new(None),
        ));
        self.source_selection = InvocationSourceSelection::Inherited;
        self.source_catalog_view = Some(super::poll_projection::PollProjection::legacy(
            crate::source_context::SourceCatalogProjection::new(
                self.state.source.borrow().catalog.clone(),
            ),
        ));
        self
    }

    pub fn portable_program_source_map(&self) -> Option<&runmat_execution::ExecutableSourceMapV2> {
        self.program_sources
            .as_ref()
            .and_then(|binding| binding.portable_map())
            .map(|map| map.as_ref())
    }

    pub(crate) fn selected_program_source_binding(
        &self,
    ) -> Option<Rc<crate::source_context::ProgramSourceBinding>> {
        self.program_sources.as_ref().map(Rc::clone)
    }

    pub fn portable_ordinal_for_source(
        &self,
        source: crate::source_context::RuntimeSourceRef,
    ) -> Option<runmat_types::ProgramSourceId> {
        source.resolve_in(self)?;
        self.program_sources.as_ref()?.source_ordinal(source)
    }

    pub(crate) fn retain_portable_program_source_map(
        &mut self,
        admitted: &crate::source_context::ValidatedPortableSourceMap,
    ) -> Result<(), crate::source_context::PortableSourceIdentityError> {
        let binding =
            self.program_sources.as_mut().and_then(Rc::get_mut).expect(
                "newly issued program binding is not shared before portable map attachment",
            );
        binding.retain_portable_map(admitted)
    }

    pub(crate) fn source_catalog_view(
        &self,
    ) -> Option<
        &super::poll_projection::PollProjection<crate::source_context::SourceCatalogProjection>,
    > {
        self.source_catalog_view.as_ref()
    }

    pub(crate) fn same_source_projection_owner(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.state, &other.state)
            && self.source_selection.same_lineage(&other.source_selection)
            && match (&self.program_sources, &other.program_sources) {
                (Some(left), Some(right)) => Rc::ptr_eq(left, right),
                (None, None) => true,
                _ => false,
            }
    }

    pub(super) fn capture_poll_projections(&mut self) {
        self.capture_poll_source_catalog();
        self.capture_poll_debug_frames();
        self.capture_poll_call_projection();
        self.capture_poll_console_capture_projection();
        self.capture_poll_output_projection();
    }

    fn capture_poll_debug_frames(&mut self) {
        let snapshot = self.debug_projection.borrow().snapshot();
        self.debug_projection = super::poll_projection::PollProjection::legacy(snapshot);
    }

    pub(in crate::context) fn poll_source_debug_quote(
        &self,
    ) -> Option<super::poll_projection::PollProjectionQuote> {
        use super::poll_projection::{PollProjection, PollProjectionQuote};
        let frames = self.debug_projection.borrow();
        let node_bytes = u64::try_from(frames.snapshot_requested_bytes()?).ok()?;
        let node_claims = frames.snapshot_retained_claims()?;
        let source =
            PollProjection::<crate::source_context::SourceCurrentProjection>::quote(0).ok()?;
        let catalog =
            PollProjection::<crate::source_context::SourceCatalogProjection>::quote(0).ok()?;
        let debug = PollProjection::<crate::debug_context::DebugFrameProjection>::quote(0).ok()?;
        source
            .checked_add(catalog)?
            .checked_add(debug)?
            .checked_add(PollProjectionQuote {
                scratch_bytes: 0,
                retained_bytes: node_bytes,
                retained_claims: node_claims,
            })
    }

    pub(in crate::context) fn with_admitted_poll_debug_frames(
        mut self,
        permit: &super::host_allocation::HostOutputPermit,
        scratch: Arc<super::host_effect::ManagedHostEffectStageReservation>,
    ) -> Result<Self, runmat_value::ExactHostNumericAllocationError> {
        let snapshot = self
            .debug_projection
            .borrow()
            .snapshot_with_permit(permit)?;
        self.debug_projection =
            super::poll_projection::PollProjection::prepare(snapshot, permit, scratch)?;
        Ok(self)
    }

    fn capture_poll_output_projection(&mut self) {
        let owner = self.selected_output_projection();
        let snapshot = owner.borrow().snapshot();
        self.output_projection = super::poll_projection::PollProjection::legacy(snapshot);
    }

    pub(crate) fn selected_output_projection(
        &self,
    ) -> super::poll_projection::PollProjection<super::output_projection::InvocationOutputProjection>
    {
        super::scope::with_active_runtime_context(|active| {
            active
                .filter(|active| Rc::ptr_eq(active.state(), self.state()))
                .map(|active| active.output_projection().clone())
                .unwrap_or_else(|| self.output_projection.clone())
        })
    }

    pub(super) fn poll_local_projection_quote(
        &self,
    ) -> Option<super::poll_projection::PollProjectionQuote> {
        let call_owner = self.selected_call_projection();
        let call_state = self.state.call.borrow();
        let (call_retained, call_claims) = call_owner
            .borrow()
            .poll_snapshot_retained_quote(&call_state)?;
        drop(call_state);
        let console_scratch = self
            .selected_console_capture_projection()
            .borrow()
            .poll_snapshot_scratch_bytes()?;
        let output = super::poll_projection::PollProjection::<
            super::output_projection::InvocationOutputProjection,
        >::quote(0)
        .ok()?;
        let call = super::poll_projection::PollProjection::<
            super::call_projection::CallProjection,
        >::quote(0)
        .ok()?;
        let console = super::poll_projection::PollProjection::<
            crate::console::ConsoleCaptureProjection,
        >::quote(0)
        .ok()?;
        output.checked_add(call)?.checked_add(console)?.checked_add(
            super::poll_projection::PollProjectionQuote {
                scratch_bytes: console_scratch,
                retained_bytes: call_retained,
                retained_claims: call_claims,
            },
        )
    }

    pub(super) fn with_admitted_poll_local_projections(
        mut self,
        permit: &super::host_allocation::HostOutputPermit,
        scratch: std::sync::Arc<super::host_effect::ManagedHostEffectStageReservation>,
    ) -> Result<Self, super::host_allocation::HostAllocationError> {
        let output = self
            .output_projection
            .borrow()
            .admitted_poll_snapshot(&self, permit.clone())?;
        let call_owner = self.selected_call_projection();
        let call = {
            let call_state = self.state.call.borrow();
            let snapshot = call_owner
                .borrow()
                .poll_snapshot(&call_state, &self, permit.clone());
            snapshot?
        };
        let console = self
            .selected_console_capture_projection()
            .borrow()
            .poll_snapshot(&self, permit.clone())?;
        self.output_projection =
            super::poll_projection::PollProjection::prepare(output, permit, scratch.clone())
                .map_err(super::host_allocation::HostAllocationError::Numeric)?;
        self.call_projection =
            super::poll_projection::PollProjection::prepare(call, permit, scratch.clone())
                .map_err(super::host_allocation::HostAllocationError::Numeric)?;
        self.console_capture_projection =
            super::poll_projection::PollProjection::prepare(console, permit, scratch)
                .map_err(super::host_allocation::HostAllocationError::Numeric)?;
        Ok(self)
    }

    pub(crate) fn output_projection(
        &self,
    ) -> &super::poll_projection::PollProjection<super::output_projection::InvocationOutputProjection>
    {
        &self.output_projection
    }

    /// Retain the selected builtin's original output-count owner at dispatch.
    /// Sharing this admitted projection adds no allocation and keeps later
    /// sibling poll scopes from substituting their invocation controls.
    pub(crate) fn with_selected_builtin_output_projection(mut self) -> Self {
        self.output_projection = self.selected_output_projection();
        self
    }

    pub(crate) fn call_projection(
        &self,
    ) -> &super::poll_projection::PollProjection<super::call_projection::CallProjection> {
        &self.call_projection
    }

    pub(crate) fn console_capture_projection(
        &self,
    ) -> &super::poll_projection::PollProjection<crate::console::ConsoleCaptureProjection> {
        &self.console_capture_projection
    }

    fn capture_poll_console_capture_projection(&mut self) {
        let snapshot = self
            .selected_console_capture_projection()
            .borrow()
            .snapshot();
        self.console_capture_projection = super::poll_projection::PollProjection::legacy(snapshot);
    }

    pub(crate) fn selected_console_capture_projection(
        &self,
    ) -> super::poll_projection::PollProjection<crate::console::ConsoleCaptureProjection> {
        super::scope::with_active_runtime_context(|active| {
            active
                .filter(|active| {
                    Rc::ptr_eq(active.state(), &self.state)
                        && Rc::ptr_eq(
                            &active.call_projection_lineage,
                            &self.call_projection_lineage,
                        )
                })
                .map(|active| active.console_capture_projection().clone())
                .unwrap_or_else(|| self.console_capture_projection.clone())
        })
    }

    fn capture_poll_call_projection(&mut self) {
        let snapshot = self.selected_call_projection().borrow().snapshot();
        self.call_projection = super::poll_projection::PollProjection::legacy(snapshot);
    }

    pub(crate) fn selected_call_projection(
        &self,
    ) -> super::poll_projection::PollProjection<super::call_projection::CallProjection> {
        super::scope::with_active_runtime_context(|active| {
            active
                .filter(|active| {
                    Rc::ptr_eq(active.state(), &self.state)
                        && Rc::ptr_eq(
                            &active.call_projection_lineage,
                            &self.call_projection_lineage,
                        )
                })
                .map(|active| active.call_projection().clone())
                .unwrap_or_else(|| self.call_projection.clone())
        })
    }

    pub(crate) fn debug_projection(
        &self,
    ) -> &super::poll_projection::PollProjection<crate::debug_context::DebugFrameProjection> {
        &self.debug_projection
    }

    /// Add an immutable VM frame to this invocation only. Independently
    /// pending siblings keep their own linked frame heads.
    pub fn with_debug_frame(mut self, frame: crate::debug_context::DebugFrame) -> Self {
        let projection = self.debug_projection.borrow().with_frame(frame);
        self.debug_projection = super::poll_projection::PollProjection::legacy(projection);
        self
    }

    pub(crate) fn source_current(
        &self,
    ) -> Option<
        &super::poll_projection::PollProjection<crate::source_context::SourceCurrentProjection>,
    > {
        self.source_current.as_ref()
    }

    pub fn program_source_id(
        &self,
        ordinal: runmat_types::ProgramSourceId,
    ) -> Option<runmat_types::SourceId> {
        self.program_sources.as_ref()?.get(ordinal)
    }

    pub fn program_source_ref(
        &self,
        ordinal: runmat_types::ProgramSourceId,
    ) -> Option<crate::source_context::RuntimeSourceRef> {
        self.program_sources.as_ref()?.source_ref(ordinal)
    }

    /// The local error-frame basis of this exact admitted source table. A
    /// selected worker response must still validate its portable ordinal
    /// against the admitted product before reattaching this token.
    pub fn call_frame_source_owner_for_program_source(
        &self,
        ordinal: runmat_types::ProgramSourceId,
    ) -> Option<crate::CallFrameSourceOwner> {
        self.program_sources.as_ref()?.frame_owner_for(ordinal)
    }

    pub fn with_program_revision(
        mut self,
        revision: Option<runmat_execution::ProgramRevision>,
    ) -> Self {
        if self.program_revision() == revision.as_ref() {
            return self;
        }
        if let (Some(service), Some(previous)) =
            (self.services.placement(), self.program_revision.take())
        {
            service.invalidate(runmat_execution::PlacementInvalidation::Program {
                revision: Arc::unwrap_or_clone(previous),
            });
        }
        self.program_revision = revision.map(Arc::new);
        self
    }

    pub fn with_execution(mut self, execution: Rc<dyn RuntimeExecutionServices>) -> Self {
        self.execution = execution;
        self
    }

    pub fn with_search_path(
        mut self,
        search_path: Arc<crate::builtins::common::path_state::SearchPath>,
    ) -> Self {
        self.search_path = Some(search_path);
        self
    }

    pub fn search_path(&self) -> Option<&Arc<crate::builtins::common::path_state::SearchPath>> {
        self.search_path.as_ref()
    }

    pub fn set_dynamic_function_loader(
        &self,
        loader: Option<Rc<crate::user_functions::DynamicFunctionLoader>>,
    ) {
        self.state.call.borrow_mut().dynamic_loader = loader;
    }

    /// Install the native loader's read-only selection view for introspection.
    pub fn set_native_name_observer(
        &self,
        observer: Option<Rc<crate::user_functions::NativeNameObserver>>,
    ) {
        self.state.call.borrow_mut().native_name_observer = observer;
    }

    /// Install the selected session's exact-definition invocation route.
    /// This does not affect the ordinary name or numeric-function resolver.
    pub fn set_bound_function_invoker(
        &self,
        invoker: Option<Rc<crate::user_functions::BoundFunctionInvoker>>,
    ) {
        self.state.call.borrow_mut().bound_invoker = invoker;
    }

    pub fn set_dynamic_function_clearer(
        &self,
        clearer: Option<Rc<crate::user_functions::DynamicFunctionClearer>>,
    ) {
        self.state.call.borrow_mut().dynamic_clearer = clearer;
    }

    pub fn set_dynamic_function_refresher(
        &self,
        refresher: Option<Rc<crate::user_functions::DynamicFunctionRefresher>>,
    ) {
        self.state.call.borrow_mut().dynamic_refresher = refresher;
    }

    pub fn runmat_extensions_enabled(&self) -> bool {
        self.invocation_language_policy
            .as_ref()
            .map(|policy| policy.get().extensions_enabled)
            .unwrap_or_else(|| self.state.runmat_extensions_enabled.get())
    }

    pub fn set_runmat_extensions_enabled(&self, enabled: bool) {
        if let Some(policy) = &self.invocation_language_policy {
            let mut updated = policy.get();
            updated.extensions_enabled = enabled;
            policy.set(updated);
        } else {
            self.state.runmat_extensions_enabled.set(enabled);
        }
    }

    pub fn language_mode(&self) -> RuntimeLanguageMode {
        self.invocation_language_policy
            .as_ref()
            .map(|policy| policy.get().language_mode)
            .unwrap_or_else(|| self.state.language_mode.get())
    }

    pub fn set_language_mode(&self, mode: RuntimeLanguageMode) {
        if let Some(policy) = &self.invocation_language_policy {
            let mut updated = policy.get();
            updated.language_mode = mode;
            policy.set(updated);
        } else {
            self.state.language_mode.set(mode);
        }
    }

    pub fn top_level_await_enabled(&self) -> bool {
        self.invocation_language_policy
            .as_ref()
            .map(|policy| policy.get().top_level_await_enabled)
            .unwrap_or_else(|| self.state.top_level_await_enabled.get())
    }

    pub fn set_top_level_await_enabled(&self, enabled: bool) {
        if let Some(policy) = &self.invocation_language_policy {
            let mut updated = policy.get();
            updated.top_level_await_enabled = enabled;
            policy.set(updated);
        } else {
            self.state.top_level_await_enabled.set(enabled);
        }
    }

    pub fn dynamic_eval_enabled(&self) -> bool {
        self.invocation_language_policy
            .as_ref()
            .map(|policy| policy.get().dynamic_eval_enabled)
            .unwrap_or_else(|| self.state.dynamic_eval_enabled.get())
    }

    pub fn execution_stack(&self) -> RuntimeExecutionStack {
        self.state.execution_stack.get()
    }

    pub fn enter_execution_stack(
        &self,
        stack: RuntimeExecutionStack,
    ) -> RuntimeExecutionStackGuard {
        let previous = self.state.execution_stack.replace(stack);
        RuntimeExecutionStackGuard {
            state: Rc::clone(&self.state),
            previous,
        }
    }

    pub fn execution_assignment(&self) -> Option<runmat_execution::ProgramExecutionAssignment> {
        self.state.execution_assignment.borrow().clone()
    }

    /// Installs scheduler-owned invocation identity for one execution extent.
    /// Dropping the returned guard restores the previous identity, which keeps
    /// nested and cancelled executions from leaking worker state into their
    /// caller's runtime context.
    pub fn enter_execution_assignment(
        &self,
        assignment: Option<runmat_execution::ProgramExecutionAssignment>,
    ) -> ProgramExecutionAssignmentGuard {
        let previous = self.state.execution_assignment.replace(assignment);
        ProgramExecutionAssignmentGuard {
            state: Rc::clone(&self.state),
            previous,
        }
    }

    pub fn execution_job(&self) -> Option<runmat_execution::JobId> {
        *self.state.execution_job.borrow()
    }

    pub fn enter_execution_job(
        &self,
        job_id: Option<runmat_execution::JobId>,
    ) -> ProgramExecutionJobGuard {
        let previous = self.state.execution_job.replace(job_id);
        ProgramExecutionJobGuard {
            state: Rc::clone(&self.state),
            previous,
        }
    }

    pub fn set_dynamic_eval_enabled(&self, enabled: bool) {
        if let Some(policy) = &self.invocation_language_policy {
            let mut updated = policy.get();
            updated.dynamic_eval_enabled = enabled;
            policy.set(updated);
        } else {
            self.state.dynamic_eval_enabled.set(enabled);
        }
    }

    pub fn callstack_limit(&self) -> usize {
        self.state.callstack_limit.get()
    }

    pub fn set_callstack_limit(&self, limit: usize) {
        self.state.callstack_limit.set(limit);
    }

    pub fn error_namespace(&self) -> String {
        self.state.error_namespace.borrow().clone()
    }

    pub fn set_error_namespace(&self, namespace: impl Into<String>) {
        let namespace = namespace.into();
        let namespace = if namespace.trim().is_empty() {
            DEFAULT_ERROR_NAMESPACE.to_string()
        } else {
            namespace
        };
        *self.state.error_namespace.borrow_mut() = namespace;
    }

    pub fn with_service_ports(mut self, services: RuntimeServicePorts) -> Self {
        if !self.services.same_device_route_owners(&services) {
            if let Some(owner) = &self.command_profiling {
                owner.mark_unproven_device_route();
            }
        }
        self.services = services;
        self
    }

    /// Bind language policy to this invocation and its clones. This does not
    /// change the session defaults observed by sibling or later invocations.
    pub fn with_language_policy(mut self, policy: RuntimeLanguagePolicy) -> Self {
        self.invocation_language_policy = Some(Rc::new(Cell::new(policy)));
        self
    }

    /// Create an isolated worker runtime that retains immutable resolution and
    /// host capabilities while owning independent mutable language state.
    pub fn fork_parallel_lab(
        &self,
        services: RuntimeServicePorts,
    ) -> Result<Self, crate::RuntimeError> {
        // A prepared session-global commit relies on its root remaining
        // unique. Refuse a fork before it clones that root; the reservation
        // is released by its transaction token on commit or cancellation.
        self.state.session_global_mutation.ensure_idle()?;
        // A lab is an isolated mutable-language-state owner. Caller-provided
        // ports cannot confer either its current frame or session base
        // workspace; a lab workspace must be installed explicitly afterward.
        let services = services
            .without_workspace()
            .without_base_workspace()
            .without_graphics_presentation()
            .without_callable_artifacts();
        let mut child = Self::with_cancellation(self.execution.clone(), self.cancellation())
            .with_service_ports(services);
        child.program_revision = self.program_revision.clone();
        child.acceleration_policy = self.acceleration_policy;
        let child = if let Some(search_path) = &self.search_path {
            child.with_search_path(Arc::clone(search_path))
        } else {
            child
        };
        child
            .display_format_owner()
            .set(self.display_format_owner().get());
        *child.state.source.borrow_mut() = self.state.source.borrow().fork_snapshot();
        let mut child = child;
        child.command_profiling = self.command_profiling.clone();
        child.invocation_arity = self.invocation_arity;
        #[cfg(not(target_arch = "wasm32"))]
        {
            child.mex_callback_ancestry = self.mex_callback_ancestry;
            child.python_reentry = self.python_reentry.clone();
            child.java_reentry = self.java_reentry.clone();
        }
        if let Some(sources) = &self.program_sources {
            child.program_sources = Some(Rc::new(sources.fork_snapshot()));
        }
        if let Some(current) = &self.source_current {
            child.source_current = Some(super::poll_projection::PollProjection::legacy(
                crate::source_context::SourceCurrentProjection::new(current.borrow().snapshot()),
            ));
        }
        if let Some(view) = &self.source_catalog_view {
            child.source_catalog_view = Some(super::poll_projection::PollProjection::legacy(
                view.borrow()
                    .fork_with_inherited_sources(&child.state.source.borrow()),
            ));
        }
        *child.state.call.borrow_mut() = self.state.call.borrow().clone();
        child.call_projection = super::poll_projection::PollProjection::legacy(
            self.selected_call_projection().borrow().snapshot(),
        );
        child.console_capture_projection = super::poll_projection::PollProjection::legacy(
            self.selected_console_capture_projection()
                .borrow()
                .snapshot(),
        );
        *child.state.classes.borrow_mut() = self.state.classes.borrow().clone();
        self.state.session_global_mutation.ensure_idle()?;
        *child.state.session_variables.borrow_mut() =
            self.state.session_variables.borrow().fork_snapshot();
        child
            .state
            .runmat_extensions_enabled
            .set(self.runmat_extensions_enabled());
        child.state.language_mode.set(self.language_mode());
        child
            .state
            .top_level_await_enabled
            .set(self.top_level_await_enabled());
        child
            .state
            .dynamic_eval_enabled
            .set(self.dynamic_eval_enabled());
        child
            .state
            .execution_stack
            .set(self.state.execution_stack.get());
        child
            .state
            .callstack_limit
            .set(self.state.callstack_limit.get());
        *child.state.error_namespace.borrow_mut() = self.state.error_namespace.borrow().clone();
        Ok(child)
    }

    /// Scope every poll of `future` to this context. This is the supported
    /// bridge for async code that still reaches ambient compatibility APIs.
    pub fn scope<F: Future>(&self, future: F) -> ContextFuture<F> {
        self.scope_pinned(Box::pin(future))
    }

    /// Adopt an already-pinned execution owner without moving its state
    /// through another by-value scope constructor or allocating a second box.
    pub fn scope_pinned<F: Future>(&self, future: std::pin::Pin<Box<F>>) -> ContextFuture<F> {
        ContextFuture::new(self.clone(), future)
    }

    /// Select the exact differentiation tape for polls of this invocation.
    /// The tape remains private to autodiff; sibling continuations inherit it
    /// through their captured context, not through a suspended thread-local.
    pub(crate) fn with_tape_poll_authority(
        mut self,
        authority: crate::builtins::deep_learning::autodiff::TapePollAuthority,
    ) -> Self {
        self.tape_poll_authority = Some(authority);
        self
    }

    pub(super) fn tape_poll_authority(
        &self,
    ) -> Option<&crate::builtins::deep_learning::autodiff::TapePollAuthority> {
        self.tape_poll_authority.as_ref()
    }

    /// Carry this frame's lexical catch exception through every poll, never
    /// across a yield. `None` explicitly masks an enclosing frame's catch.
    pub fn scope_active_catch<F: Future>(
        &self,
        exception: Option<runmat_value::MException>,
        future: F,
    ) -> super::ActiveCatchFuture<F> {
        super::ActiveCatchFuture::new(self, exception, future)
    }

    pub fn scope_active_catch_shared<F: Future>(
        &self,
        exception: Option<Rc<runmat_value::MException>>,
        future: F,
    ) -> super::ActiveCatchFuture<F> {
        super::ActiveCatchFuture::new_shared(self, exception, future)
    }

    /// Scope a pinned execution child without allocating another future box.
    /// The child is also dropped under the catch authority on abandonment.
    pub fn scope_active_catch_unboxed_shared<F: Future>(
        &self,
        exception: Option<Rc<runmat_value::MException>>,
        future: F,
    ) -> super::UnboxedActiveCatchFuture<F> {
        super::UnboxedActiveCatchFuture::new(self, exception, future)
    }

    pub fn enter_active_catch(
        &self,
        exception: Option<runmat_value::MException>,
    ) -> super::ActiveCatchGuard {
        super::ActiveCatchGuard::enter(self.local_identity(), exception.map(Rc::new))
    }

    /// Enter one synchronous poll with an already-owned exception. This is
    /// used by execution engines to avoid cloning exception payloads per poll.
    pub fn enter_active_catch_shared(
        &self,
        exception: Option<Rc<runmat_value::MException>>,
    ) -> super::ActiveCatchGuard {
        super::ActiveCatchGuard::enter(self.local_identity(), exception)
    }

    pub fn active_catch_exception(&self) -> Option<runmat_value::MException> {
        super::catch::active_for(self)
    }

    /// Snapshot a caller's active catch for a nested invocation without
    /// copying its payload. The callee's own lexical catch may override it.
    pub fn active_catch_exception_shared(&self) -> Option<Rc<runmat_value::MException>> {
        super::catch::active_shared_for(self)
    }

    /// Activate this context for one synchronous executor or foreign-host
    /// extent. Async code should use [`Self::scope`] so the context is removed
    /// across yields.
    pub fn enter(&self) -> RuntimeContextGuard {
        let mut context = self.clone();
        // A synchronous host reentering an original handle during a scoped
        // poll must see that invocation's overlays, not the original handle's
        // pre-poll projection. Other sessions and isolated lineages keep their
        // own selection. This shares the active owner; it does not snapshot it.
        context.call_projection = self.selected_call_projection();
        context.console_capture_projection = self.selected_console_capture_projection();
        context.output_projection = self.selected_output_projection();
        RuntimeContextGuard::enter(context)
    }
}

fn selected_invocation_filesystem() -> Option<runmat_filesystem::InvocationFilesystemSelection> {
    let active = super::scope::active_runtime_context()?;
    active
        .service_ports()
        .selected_filesystem_for_active_context()
}
