use super::RunMatSession;

use runmat_execution::profiling::CommandProfilingRoot;

pub(super) struct AdmittedCommandProfiling {
    pub(super) root: CommandProfilingRoot,
    /// Core-private positive admission fact; the execution receipt itself
    /// remains incomplete until this is joined with sealed physical evidence.
    pub(super) canonical_device_route: bool,
}

impl RunMatSession {
    /// A test child borrows the originating command, while a host entrypoint
    /// alone receives seal authority. Both source and executable entrypoints
    /// must use this admission boundary before running user code.
    pub(super) fn admit_command_profiling(&mut self) -> Option<AdmittedCommandProfiling> {
        let root = self.inherited_command_profiling.is_none().then(|| {
            CommandProfilingRoot::new_with_thread_cpu_sampler(
                self.runtime_context
                    .service_ports()
                    .timing()
                    .ok()
                    .and_then(|timing| timing.command_thread_cpu_sampler()),
            )
        });
        let receipt = self
            .inherited_command_profiling
            .as_ref()
            .cloned()
            .or_else(|| root.as_ref().map(CommandProfilingRoot::receipt))
            .expect("root or inherited command receipt");
        let provider = runmat_accelerate::registered_wgpu_provider_for_profiling();
        if provider
            .and_then(|provider| provider.wgpu_device_access())
            .is_some_and(|access| access.has_unmeasured_implicit_transfer())
        {
            receipt.mark_unproven_device_time();
        }
        let canonical_device_route = root.is_some()
            && provider
                .and_then(|provider| provider.wgpu_device_access())
                .is_some_and(|access| !access.has_raw_export())
            && self
                .runtime_context
                .service_ports()
                .same_device_route_owners(&self.canonical_device_route_ports);
        self.runtime_context = self
            .runtime_context
            .clone()
            .with_command_profiling(receipt)
            .with_command_device_provider(provider);
        root.map(|root| AdmittedCommandProfiling {
            root,
            canonical_device_route,
        })
    }

    /// Recheck the same access at seal: an implicit transfer can race after
    /// admission and before the command's final physical evidence is frozen.
    pub(super) fn observe_command_device_time_at_seal(&self, root: &CommandProfilingRoot) {
        if self
            .runtime_context
            .command_device_access()
            .is_some_and(|access| access.has_unmeasured_implicit_transfer())
        {
            root.receipt().mark_unproven_device_time();
        }
    }
}
