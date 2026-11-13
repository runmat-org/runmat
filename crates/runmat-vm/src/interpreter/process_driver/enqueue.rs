//! Branch-selected queue construction on the same authentic driver owner.
//! Separate physical preparation frames, not a second queue or scheduler.
use super::*;

pub(super) fn unbounded<T, F, M>(
    handle: &ProcessDriverHandle,
    make_future: M,
) -> Result<ProcessReplyTicket<T>, RuntimeError>
where
    T: 'static,
    F: Future<Output = Result<T, RuntimeError>> + 'static,
    M: FnOnce() -> Result<F, RuntimeError>,
{
    handle.enqueue_typed_unbounded(make_future()?)
}

pub(super) fn managed<T, F, M>(
    context: &RuntimeContext,
    owner: DriverStateOwner,
    additional_scratch_bytes: u64,
    make_future: M,
) -> Result<ProcessReplyTicket<T>, RuntimeError>
where
    T: 'static,
    F: Future<Output = Result<T, RuntimeError>> + 'static,
    M: FnOnce() -> Result<F, RuntimeError>,
{
    let (geometry, expected_id, parent, depth, wake_hub) = {
        let state = owner.borrow();
        state.ensure_context(context)?;
        if !state.accepting || state.retiring {
            return Err(driver_unavailable());
        }
        let hub = state
            .wake_hub
            .get()
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let (parent, depth) = state.current.map_or((None, 1), |(parent, depth)| {
            (Some(parent), depth.saturating_add(1))
        });
        let geometry = QueueGrowthGeometry {
            task_capacity: state.tasks.capacity(),
            task_index: Some(state.tasks.required_index()),
            ready_capacity: hub.ready.capacity(),
            ready_len: hub.ready.len(),
        };
        (
            geometry,
            state
                .tasks
                .next_id(state.next_id)
                .map_err(|_| driver_overflow())?,
            parent,
            depth,
            state.wake_hub.clone(),
        )
    };
    let scratch_bytes = request_scratch_bytes::<F, T>()?
        .checked_add(additional_scratch_bytes)
        .ok_or_else(driver_overflow)?;
    let mut prepared = PreparedQueueGrowth::prepare(
        context,
        geometry,
        scratch_bytes,
        managed_request_control_bytes::<T>()?,
        3,
    )
    .map_err(|error| queue_allocation_error(context, error))?;
    let scratch = Some(prepared.retained_scratch());

    // Only after the complete request and queue capacity are admitted may
    // caller-owned child state enter the queued future.
    let request = prepared
        .with_output_construction(context, |permit| {
            make_future().map(|future| {
                prepare_managed_request(permit, wake_hub.clone(), expected_id, depth, future)
            })
        })
        .map_err(|error| queue_allocation_error(context, error))?;
    let (completion, control, wake, queued) = request?
        .map_err(|error| queue_allocation_error(context, HostAllocationError::Numeric(error)))?;
    prepared
        .checkpoint(context)
        .map_err(|error| queue_allocation_error(context, error))?;
    let (wake, retirement) = {
        let mut state = owner.borrow_mut();
        state.ensure_context(context)?;
        if state
            .tasks
            .next_id(state.next_id)
            .map_err(|_| driver_overflow())?
            != expected_id
            || !state.accepting
            || state.retiring
        {
            return Err(driver_unavailable());
        }
        let mut hub = wake_hub
            .get()
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let retirement = prepared.install(&mut state.tasks, &mut hub.ready);
        if hub.ready.len() >= hub.ready.capacity() {
            (None, retirement)
        } else {
            state.register_prepared(
                PreparedRegistration {
                    id: expected_id,
                    parent,
                    depth,
                    wake: wake.clone(),
                    future: queued,
                    reply: control,
                    scratch: scratch.clone(),
                },
                &mut hub.ready,
            );
            (Some(wake.clone()), retirement)
        }
    };
    drop(retirement);
    let wake = wake.ok_or_else(driver_unavailable)?;
    let host = {
        let state = owner.borrow();
        let mut hub = state
            .wake_hub
            .get()
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        // `register_prepared` installed the ready entry after the exact
        // backing replacement was committed.
        hub.host.take()
    };
    if let Some(host) = host {
        host.wake();
    }
    Ok(ProcessReplyTicket {
        completion,
        wake,
        _scratch: scratch,
    })
}
