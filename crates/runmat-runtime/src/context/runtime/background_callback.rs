//! Original background-call authority without ownership of its session state.

use super::*;

struct BackgroundCallbackServices {
    services: RuntimeServicePorts,
    policy: RuntimeLanguagePolicy,
}

/// Ports and language rules shared by timers and persistent graphics callbacks.
/// A caller frame is never a persistent background workspace authority.
impl RuntimeContext {
    pub(crate) fn background_callback_runtime(
        &self,
        requires_source_evaluation: bool,
        operation: &'static str,
    ) -> Result<Self, crate::RuntimeError> {
        let captured = self.background_callback_services(requires_source_evaluation, operation)?;
        Ok(self
            .clone()
            .with_cleanup_completion(None)
            .with_graphics_callback_mutation(None)
            .with_language_policy(captured.policy)
            .with_service_ports(captured.services))
    }

    fn background_callback_services(
        &self,
        requires_source_evaluation: bool,
        operation: &'static str,
    ) -> Result<BackgroundCallbackServices, crate::RuntimeError> {
        let policy = RuntimeLanguagePolicy {
            language_mode: self.language_mode(),
            extensions_enabled: self.runmat_extensions_enabled(),
            top_level_await_enabled: self.top_level_await_enabled(),
            dynamic_eval_enabled: self.dynamic_eval_enabled(),
        };
        let mut services = self.service_ports().clone().without_workspace();
        if let Some(base) = self.service_ports().base_workspace() {
            services = services.with_workspace(Rc::new(
                crate::workspace::BaseWorkspaceService::new(base.owner()),
            ));
        } else if requires_source_evaluation {
            self.service_ports()
                .require_base_workspace(operation)
                .map_err(super::super::RuntimeCapabilityError::into_runtime_error)?;
        }
        if let Some(source) = self.service_ports().source_evaluation() {
            services = services.with_source_evaluation(source.bind_background_authority()?);
        } else if requires_source_evaluation {
            self.service_ports()
                .require_source_evaluation(operation)
                .map_err(super::super::RuntimeCapabilityError::into_runtime_error)?;
        }
        Ok(BackgroundCallbackServices { services, policy })
    }
}

/// A sealed projection reconstructs only its original invocation selection.
/// The exact state/executor are weak; GUI attachment cannot grant other ports.
#[cfg(feature = "plot-core")]
type BackgroundCallbackResume =
    dyn Fn(&RuntimeContext) -> Result<RuntimeContext, crate::RuntimeError>;

#[cfg(feature = "plot-core")]
pub(crate) struct BackgroundCallbackAuthority {
    resume: Box<BackgroundCallbackResume>,
}

#[cfg(feature = "plot-core")]
impl std::fmt::Debug for BackgroundCallbackAuthority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BackgroundCallbackAuthority { original: true, .. }")
    }
}

#[cfg(feature = "plot-core")]
impl std::ops::Deref for BackgroundCallbackAuthority {
    type Target = BackgroundCallbackResume;

    fn deref(&self) -> &Self::Target {
        self.resume.as_ref()
    }
}

#[cfg(feature = "plot-core")]
impl RuntimeContext {
    pub(crate) fn capture_background_callback(
        &self,
        requires_source_evaluation: bool,
    ) -> Result<BackgroundCallbackAuthority, crate::RuntimeError> {
        let captured =
            self.background_callback_services(requires_source_evaluation, "graphics callback")?;
        // Do not clone a caller's constructor/Value payload merely to clear it.
        let mut selected = RuntimeContext {
            cleanup_completion: None,
            acceleration_policy: self.acceleration_policy,
            graphics_callback_mutation: None,
            execution: self.execution.clone(),
            // Callbacks run in the session's own environment.
            environment: self.environment.clone(),
            selected_graph_program: self.selected_graph_program.clone(),
            selected_graph_function_resolver: self.selected_graph_function_resolver.clone(),
            selected_static_site_resolver: self.selected_static_site_resolver.clone(),
            selected_static_caller: self.selected_static_caller,
            native_resolution_scope: self.native_resolution_scope.clone(),
            program_revision: self.program_revision.clone(),
            program_sources: self.program_sources.clone(),
            source_current: self.source_current.clone(),
            source_selection: self.source_selection.clone(),
            source_catalog_view: self.source_catalog_view.clone(),
            debug_projection: super::super::poll_projection::PollProjection::legacy(
                Default::default(),
            ),
            output_projection: super::super::poll_projection::PollProjection::legacy(
                Default::default(),
            ),
            call_projection: super::super::poll_projection::PollProjection::legacy(
                self.selected_call_projection().borrow().snapshot(),
            ),
            call_projection_lineage: Rc::new(()),
            console_capture_projection: super::super::poll_projection::PollProjection::legacy(
                Default::default(),
            ),
            #[cfg(feature = "plot-core")]
            plot_receipts: self.plot_receipts.clone(),
            command_profiling: None,
            command_device_provider: None,
            command_device_access: None,
            tape_poll_authority: None,
            search_path: self.search_path.clone(),
            services: captured.services.without_graphics_presentation(),
            state: self.state.clone(),
            parallel_random_scope: self.parallel_random_scope.clone(),
            invocation_language_policy: Some(Rc::new(Cell::new(captured.policy))),
            invocation_arity: None,
            constructor_receiver: None,
            selected_classes: self.selected_classes.clone(),
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
            cancellation: Rc::new(RefCell::new(self.cancellation())),
        };
        selected.capture_poll_source_catalog();
        // A later same-session drainer is not this source selection's owner.
        selected.source_selection = if selected.program_sources.is_some() {
            InvocationSourceSelection::Selected(Rc::new(()))
        } else {
            InvocationSourceSelection::Unbound(Rc::new(()))
        };
        // Exhaustive decomposition makes a new context field an explicit
        // capture/reconstruction decision, rather than inheriting a drainer.
        let RuntimeContext {
            cleanup_completion: _,
            acceleration_policy,
            graphics_callback_mutation: _,
            execution,
            environment,
            selected_graph_program,
            selected_graph_function_resolver,
            selected_static_site_resolver,
            selected_static_caller,
            native_resolution_scope,
            program_revision,
            program_sources,
            source_current,
            source_selection,
            source_catalog_view,
            debug_projection,
            output_projection,
            call_projection,
            call_projection_lineage,
            console_capture_projection,
            #[cfg(feature = "plot-core")]
            plot_receipts,
            command_profiling,
            command_device_provider,
            command_device_access,
            tape_poll_authority,
            search_path,
            services,
            state,
            parallel_random_scope,
            invocation_language_policy,
            invocation_arity,
            constructor_receiver,
            selected_classes,
            selected_object_class,
            selected_method_body,
            selected_method_frame,
            retained_super_constructors,
            retained_super_caller_origin,
            retained_super_body,
            native_invocation_lineage,
            #[cfg(not(target_arch = "wasm32"))]
            mex_callback_ancestry,
            #[cfg(not(target_arch = "wasm32"))]
            python_reentry,
            #[cfg(not(target_arch = "wasm32"))]
            java_reentry,
            vm_fusion_host_route,
            native_region_host_route: _,
            prepared_host_execution,
            vm_fusion_selected_binding,
            cancellation,
        } = selected;
        let state = Rc::downgrade(&state);
        let execution = Rc::downgrade(&execution);
        Ok(BackgroundCallbackAuthority {
            resume: Box::new(move |attachment| {
                let fail = || {
                    crate::build_runtime_error(
                        "graphics property belongs to another or ended runtime authority",
                    )
                    .with_identifier("RunMat:graphics:CallbackSessionMismatch")
                    .build()
                };
                let state = state.upgrade().ok_or_else(fail)?;
                let execution = execution.upgrade().ok_or_else(fail)?;
                if !Rc::ptr_eq(&state, attachment.state())
                    || !Rc::ptr_eq(&execution, attachment.execution())
                {
                    return Err(fail());
                }
                if cancellation.borrow().is_cancelled() {
                    return Err(
                        crate::execution::ExecutionServiceError::Cancelled.into_runtime_error()
                    );
                }
                let mut services = services.clone();
                if let Some(queue) = attachment.service_ports().graphics_callback_queue() {
                    services = services.with_graphics_callback_queue(queue.clone());
                }
                if let Some(presentation) = attachment.service_ports().graphics_presentation() {
                    services = services.with_graphics_presentation(presentation.clone());
                }
                Ok(RuntimeContext {
                    cleanup_completion: None,
                    acceleration_policy,
                    graphics_callback_mutation: None,
                    execution,
                    environment: environment.clone(),
                    selected_graph_program: selected_graph_program.clone(),
                    selected_graph_function_resolver: selected_graph_function_resolver.clone(),
                    selected_static_site_resolver: selected_static_site_resolver.clone(),
                    selected_static_caller,
                    native_resolution_scope: native_resolution_scope.clone(),
                    program_revision: program_revision.clone(),
                    program_sources: program_sources.clone(),
                    source_current: source_current.clone(),
                    source_selection: source_selection.clone(),
                    source_catalog_view: source_catalog_view.clone(),
                    debug_projection: debug_projection.clone(),
                    output_projection: output_projection.clone(),
                    call_projection: call_projection.clone(),
                    call_projection_lineage: call_projection_lineage.clone(),
                    console_capture_projection: console_capture_projection.clone(),
                    #[cfg(feature = "plot-core")]
                    plot_receipts: plot_receipts.clone(),
                    command_profiling: command_profiling.clone(),
                    command_device_provider,
                    command_device_access: command_device_access.clone(),
                    tape_poll_authority: tape_poll_authority.clone(),
                    search_path: search_path.clone(),
                    services,
                    state,
                    parallel_random_scope: parallel_random_scope.clone(),
                    invocation_language_policy: invocation_language_policy.clone(),
                    invocation_arity,
                    constructor_receiver: constructor_receiver.clone(),
                    selected_classes: selected_classes.clone(),
                    selected_object_class: selected_object_class.clone(),
                    selected_method_body: selected_method_body.clone(),
                    selected_method_frame: selected_method_frame.clone(),
                    retained_super_constructors: retained_super_constructors.clone(),
                    retained_super_caller_origin,
                    retained_super_body: retained_super_body.clone(),
                    native_invocation_lineage: native_invocation_lineage.clone(),
                    #[cfg(not(target_arch = "wasm32"))]
                    mex_callback_ancestry,
                    #[cfg(not(target_arch = "wasm32"))]
                    python_reentry: python_reentry.clone(),
                    #[cfg(not(target_arch = "wasm32"))]
                    java_reentry: java_reentry.clone(),
                    vm_fusion_host_route: vm_fusion_host_route.clone(),
                    native_region_host_route: None,
                    prepared_host_execution: prepared_host_execution.clone(),
                    vm_fusion_selected_binding,
                    cancellation: cancellation.clone(),
                })
            }),
        })
    }
}
