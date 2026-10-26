//! Original-pool eligibility and immutable target selection for callback waves.
use super::*;

enum SelectedTarget<'a> {
    Dynamic(&'a Value),
    Retained(&'a crate::call::descriptor::RepeatableCallable),
}

impl CallbackBatch {
    /// Auto falls back only on known worker eligibility/no current pool.
    /// Once selected, source/provider/callback failures never replay serially.
    pub fn select(
        runtime: Option<&RuntimeContext>,
        target: &Value,
        mode: CallbackParallelMode,
    ) -> Result<Option<Self>, RuntimeError> {
        Self::select_target(runtime, SelectedTarget::Dynamic(target), mode)
    }

    pub(crate) fn select_retained(
        runtime: Option<&RuntimeContext>,
        target: &crate::call::descriptor::RepeatableCallable,
        mode: CallbackParallelMode,
    ) -> Result<Option<Self>, RuntimeError> {
        Self::select_target(runtime, SelectedTarget::Retained(target), mode)
    }

    fn select_target(
        runtime: Option<&RuntimeContext>,
        target: SelectedTarget<'_>,
        mode: CallbackParallelMode,
    ) -> Result<Option<Self>, RuntimeError> {
        if mode == CallbackParallelMode::Off {
            return Ok(None);
        }
        let Some(runtime) = runtime else {
            return if mode == CallbackParallelMode::Auto {
                Ok(None)
            } else {
                Err(unavailable(
                    runtime,
                    "parallel callbacks require an explicit selected runtime",
                ))
            };
        };
        let service = runtime.execution();
        if !service.supports_callback_batches() {
            return if mode == CallbackParallelMode::Auto {
                Ok(None)
            } else {
                Err(unavailable(
                    Some(runtime),
                    "selected adapter does not provide worker callback batches",
                ))
            };
        }
        let pool = service
            .current_pool()
            .map_err(|error| execution_error(runtime, error))?;
        let pool = match (pool, mode) {
            (Some(pool), _) => pool,
            (None, CallbackParallelMode::On) => service
                .ensure_pool(PoolRequest::automatic())
                .map_err(|error| execution_error(runtime, error))?,
            (None, _) => return Ok(None),
        };
        if pool.state != runmat_execution::PoolState::Ready
            || pool.workers == 0
            || !matches!(
                pool.backend,
                PoolBackend::LocalProcesses | PoolBackend::BrowserWorkers
            )
        {
            return if mode == CallbackParallelMode::Auto {
                Ok(None)
            } else {
                Err(unavailable(
                    Some(runtime),
                    "parallel callbacks require an eligible native or browser worker pool",
                ))
            };
        }
        let selected_program = match &target {
            SelectedTarget::Retained(selected) => selected
                .admitted_reentry_program()
                .or_else(|| runtime.selected_graph_program()),
            SelectedTarget::Dynamic(_) => runtime.selected_graph_program(),
        };
        let program = selected_program.cloned().ok_or_else(|| {
            unavailable(
                Some(runtime),
                "parallel callbacks lack selected Graph/source authority",
            )
        })?;
        let selected = {
            let _entry = runtime.enter();
            let result = match target {
                SelectedTarget::Dynamic(target) => {
                    let target = crate::canonicalize_callback_handle_for_semantic_resolution(
                        target.clone(),
                    )?;
                    program.select_callable(runtime, RootedValue::in_context(target, runtime)?)
                }
                SelectedTarget::Retained(target) => {
                    program.select_retained_worker_callable(runtime, target.clone())
                }
            };
            result.map_err(|mut error| {
                error.gpu_gather_retry = GpuGatherRetry::Never;
                error
            })?
        };
        Ok(Some(Self {
            runtime: runtime
                .clone()
                .with_cancellation_token(runtime.cancellation()),
            selected,
            program,
            pool: pool.handle,
            width: pool.workers as usize,
        }))
    }
}
