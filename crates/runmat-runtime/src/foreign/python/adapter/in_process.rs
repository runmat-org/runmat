use super::*;

/// A cancelled async release may already be queued on the interpreter lane.
/// Release is idempotent there, so keep every unacknowledged identity for a
/// later retry rather than losing it after the queue has been drained.
struct PendingReleases {
    queue: Arc<ReleaseQueue>,
    remaining: std::collections::VecDeque<u64>,
}

impl Drop for PendingReleases {
    fn drop(&mut self) {
        let mut queue = self
            .queue
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        queue.extend(self.remaining.drain(..));
    }
}

impl PythonAdapter {
    async fn drain_releases_async(
        &self,
        session: &PythonSession,
        context: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        let mut pending = PendingReleases {
            queue: Arc::clone(&self.released),
            remaining: self.take_releases()?.into(),
        };
        while let Some(&foreign_handle) = pending.remaining.front() {
            let python_handle = { self.resources.borrow().get(&foreign_handle).copied() };
            if let Some(python_handle) = python_handle {
                session
                    .invoke_with_async_callback(
                        PythonCall::Release {
                            handle: python_handle,
                        },
                        context.python_reentry_for(session),
                        Some(Box::new(
                            crate::foreign::native_lane_profiling::NativeLaneProfiling::admit(
                                context,
                            ),
                        )),
                        |_, _| async {
                            Err(PythonError::host(
                                "PythonCallbackError",
                                "release cannot invoke a RunMat callback",
                            ))
                        },
                    )
                    .await
                    .map_err(python_runtime_error)?;
                self.resources.borrow_mut().remove(&foreign_handle);
                self.python_to_foreign.borrow_mut().remove(&python_handle);
            }
            pending.remaining.pop_front();
        }
        Ok(())
    }

    pub(super) async fn invoke_in_process(
        self: Rc<Self>,
        context: RuntimeContext,
        call: ForeignCall,
    ) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
        let _input_roots = crate::execution::RootedValueLease::for_values(call.arguments.iter())
            .map_err(|error| error.into_runtime_error())?;
        let cancellation = context.cancellation();
        let operation = async {
            let prepared = prepare_python_call(&call.symbol, call.arguments)?;
            let session = self.ensure_session()?;
            self.drain_releases_async(&session, &context).await?;
            let (python_call, assigned_receiver) = self.prepare_invocation(&context, prepared)?;
            // The queued PythonCall owns converted COW buffers and values;
            // callbacks are separately rooted in the adapter registry. The
            // lane keeps its admission and private bytes until physical
            // completion even if this caller cancels below.
            let callback_owner = Rc::clone(&self);
            let callback_session = session.clone();
            let callback_context = context.clone();
            let values = session
                .invoke_with_async_callback(
                    python_call,
                    context.python_reentry_for(&session),
                    Some(Box::new(
                        crate::foreign::native_lane_profiling::NativeLaneProfiling::admit(&context),
                    )),
                    move |invocation, authority| {
                        let owner = Rc::clone(&callback_owner);
                        let session = callback_session.clone();
                        let context = callback_context.clone();
                        async move {
                            owner
                                .invoke_callback_async(&session, &context, invocation, authority)
                                .await
                        }
                    },
                )
                .await
                .map_err(python_runtime_error)?;
            self.finish_invocation_async(
                &session,
                &context,
                call.requested_outputs,
                assigned_receiver,
                values,
                context.python_reentry_for(&session),
            )
            .await
        };
        use futures::FutureExt;
        let operation = operation.fuse();
        let cancelled = cancellation.cancelled().fuse();
        futures::pin_mut!(operation, cancelled);
        futures::select! {
            result = operation => result,
            _ = cancelled => Err(python_runtime_error(PythonError::host(
                "PythonCancellationError",
                "Python operation was cancelled; interpreter retirement remains unverified until the lane completes",
            ))),
        }
    }

    async fn finish_invocation_async(
        self: &Rc<Self>,
        session: &PythonSession,
        context: &RuntimeContext,
        requested_outputs: usize,
        assigned_receiver: Option<ForeignRef>,
        values: Vec<PythonValue>,
        authority: Option<&runmat_python::PythonReentryAuthority>,
    ) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
        if let Some(reference) = assigned_receiver {
            let value = crate::foreign::single_output_sequence(Value::Foreign(reference))?;
            return crate::execution::RootedValueSequence::new(value)
                .map_err(|error| error.into_runtime_error());
        }
        let mut converted = crate::execution::RootedValuesBuilder::with_capacity(values.len());
        for value in values {
            let value = self
                .result_from_python_async(session, context, value, authority)
                .await?;
            converted
                .push(value)
                .map_err(|error| error.into_runtime_error())?;
        }
        converted
            .finish()
            .map_err(|error| error.into_runtime_error())?
            .with_owned_output_storage(|values| {
                let value = Self::shape_python_outputs(requested_outputs, values)?;
                crate::execution::RootedValueSequence::new(value)
                    .map_err(|error| error.into_runtime_error())
            })
    }

    async fn result_from_python_async(
        self: &Rc<Self>,
        session: &PythonSession,
        context: &RuntimeContext,
        value: PythonValue,
        authority: Option<&runmat_python::PythonReentryAuthority>,
    ) -> Result<Value, RuntimeError> {
        let PythonValue::Object(handle) = value else {
            return value_from_python(value, Some(context));
        };
        if let Some(reference) = self
            .python_to_foreign
            .borrow()
            .get(&handle)
            .and_then(WeakForeignRef::upgrade)
        {
            return Ok(Value::Foreign(reference));
        }
        let callback_owner = Rc::clone(self);
        let callback_session = session.clone();
        let callback_context = (*context).clone();
        let metadata = session
            .metadata_with_reentry_async(
                handle,
                authority,
                Some(Box::new(
                    crate::foreign::native_lane_profiling::NativeLaneProfiling::admit(context),
                )),
                move |invocation, authority| {
                    let owner = Rc::clone(&callback_owner);
                    let session = callback_session.clone();
                    let context = callback_context.clone();
                    async move {
                        owner
                            .invoke_callback_async(&session, &context, invocation, authority)
                            .await
                    }
                },
            )
            .await
            .map_err(python_runtime_error)?;
        self.register_python_object(handle, metadata)
    }

    async fn invoke_callback_async(
        self: &Rc<Self>,
        session: &PythonSession,
        invocation_context: &RuntimeContext,
        invocation: PythonCallbackInvocation,
        authority: runmat_python::PythonReentryAuthority,
    ) -> Result<PythonValue, PythonError> {
        let (registration, context) =
            self.callback_context(invocation.callback, invocation_context, Some(&authority))?;
        let mut arguments =
            crate::execution::RootedValuesBuilder::with_capacity(invocation.arguments.len());
        for value in invocation.arguments {
            let value = self
                .result_from_python_async(session, &context, value, Some(&authority))
                .await
                .map_err(|error| PythonError::host("PythonCallbackError", error.to_string()))?;
            arguments
                .push(value)
                .map_err(|error| PythonError::host("PythonCallbackError", error.to_string()))?;
        }
        arguments
            .finish()
            .map_err(|error| PythonError::host("PythonCallbackError", error.to_string()))?
            .with_unbounded_values_async(|arguments| async move {
                let request =
                    foreign_callback_request(&context, &registration.callback, arguments, 1)
                        .map_err(|error| {
                            PythonError::host("PythonCallbackError", error.to_string())
                        })?;
                let sequence = invoke_foreign_callback(context.clone(), request)
                    .await
                    .map_err(|error| PythonError::host("PythonCallbackError", error.to_string()))?;
                self.finish_callback(&context, sequence)
            })
            .await
            .map_err(|error| PythonError::host("PythonCallbackError", error.to_string()))?
    }
}
