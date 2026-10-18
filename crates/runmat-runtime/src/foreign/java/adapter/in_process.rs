use super::lane_call::JavaLaneResult;
use super::prepare::JavaOutput;
use super::*;
use futures::future::{FutureExt, LocalBoxFuture};
use futures::stream::{FuturesUnordered, StreamExt};

/// A cancelled release keeps every unacknowledged handle for the next call.
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

impl JavaAdapter {
    pub(super) async fn invoke_in_process(
        self: Rc<Self>,
        context: RuntimeContext,
        call: ForeignCall,
    ) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
        let _input_roots = crate::execution::RootedValueLease::for_values(call.arguments.iter())
            .map_err(|error| error.into_runtime_error())?;
        let cancellation = context.cancellation();
        let operation = async {
            if self.retirement_requested.get() || self.project_retirement.is_pending() {
                return Err(invalid_conversion("Java selected adapter is retiring"));
            }
            if call.symbol == "usejava" {
                let feature = string_argument(call.arguments.into_iter().next(), "Java feature")?;
                let parsed = runmat_builtins::JavaFeature::parse(&feature).ok_or_else(|| {
                    invalid_conversion("Java feature must be 'jvm', 'awt', 'swing', or 'desktop'")
                })?;
                if matches!(
                    parsed,
                    runmat_builtins::JavaFeature::Jvm | runmat_builtins::JavaFeature::Desktop
                ) {
                    return self.root_single(self.usejava_value(vec![Value::String(feature)])?);
                }
                let session = self.started_lane()?;
                let result = self
                    .run_lane(&session, &context, |owner| {
                        super::lane_call::JavaLaneCall::Headless.execute(owner)
                    })
                    .await?;
                let JavaLaneResult::Headless(headless) = result else {
                    return Err(invalid_conversion(
                        "Java graphics capability response is invalid",
                    ));
                };
                return self.root_single(Value::Bool(!headless));
            }
            if matches!(call.symbol.as_str(), "classpath" | "has_classpath") && !self.is_running() {
                let snapshot = self.initial_classpath.borrow().snapshot();
                let value = if call.symbol == "classpath" {
                    self.classpath_value(call.arguments, snapshot)?
                } else {
                    Value::Bool(
                        !snapshot.bootstrap.is_empty()
                            || !snapshot.project.is_empty()
                            || !snapshot.dynamic.is_empty(),
                    )
                };
                return self.root_single(value);
            }
            let prepared = self.prepare_java_call(&context, call)?;
            let session = self.started_lane()?;
            self.drain_releases_async(&session, &context).await?;
            let super::prepare::PreparedJavaCall { command, output } = prepared;
            let result = self
                .run_lane(&session, &context, move |owner| command.execute(owner))
                .await?;
            self.complete_java_call(output, result)
        };
        let operation = operation.fuse();
        let cancelled = cancellation.cancelled().fuse();
        futures::pin_mut!(operation, cancelled);
        futures::select! {
            result = operation => result,
            _ = cancelled => Err(java_invocation_error(JavaInvocationError::Callback(
                "Java operation was cancelled; lane retirement remains unverified until physical completion".into(),
            ))),
        }
    }

    fn started_lane(&self) -> Result<Rc<JavaLane>, RuntimeError> {
        self.ensure_session()?;
        Ok(self
            .session
            .borrow()
            .as_ref()
            .expect("Java lane initialized")
            .clone())
    }

    fn root_single(
        &self,
        value: Value,
    ) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
        crate::foreign::single_output_sequence(value).and_then(|sequence| {
            crate::execution::RootedValueSequence::new(sequence)
                .map_err(|error| error.into_runtime_error())
        })
    }

    fn complete_java_call(
        &self,
        output: JavaOutput,
        result: JavaLaneResult,
    ) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
        match (output, result) {
            (JavaOutput::Value, JavaLaneResult::Value(value)) => {
                self.root_single(self.value_from_java(value)?)
            }
            (JavaOutput::Assigned(reference), JavaLaneResult::Empty) => {
                self.root_single(Value::Foreign(reference))
            }
            (JavaOutput::Empty, JavaLaneResult::Empty) => {
                crate::execution::RootedValueSequence::new(runmat_value::ValueSequence::empty())
                    .map_err(|error| error.into_runtime_error())
            }
            (JavaOutput::Classpath(arguments), JavaLaneResult::Classpath(snapshot)) => {
                self.root_single(self.classpath_value(arguments, snapshot)?)
            }
            (JavaOutput::HasClasspath, JavaLaneResult::Classpath(snapshot)) => {
                self.root_single(Value::Bool(
                    !snapshot.bootstrap.is_empty()
                        || !snapshot.project.is_empty()
                        || !snapshot.dynamic.is_empty(),
                ))
            }
            _ => Err(invalid_conversion(
                "Java lane returned a response for a different operation",
            )),
        }
    }

    async fn drain_releases_async(
        self: &Rc<Self>,
        lane: &JavaLane,
        context: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        let remaining = {
            let mut queue = self.released.0.lock().map_err(|_| {
                foreign_error(
                    ForeignErrorKind::HostUnavailable,
                    "Java release queue lock was poisoned",
                )
            })?;
            std::mem::take(&mut *queue).into()
        };
        let mut pending = PendingReleases {
            queue: Arc::clone(&self.released),
            remaining,
        };
        while let Some(&foreign_handle) = pending.remaining.front() {
            let java_handle = self.resources.borrow().get(&foreign_handle).copied();
            if let Some(java_handle) = java_handle {
                self.run_lane(lane, context, move |owner| {
                    owner.release_registered(java_handle)
                })
                .await?;
                self.resources.borrow_mut().remove(&foreign_handle);
                self.java_to_foreign.borrow_mut().remove(&java_handle);
                lane.acknowledge_release(java_handle);
            }
            pending.remaining.pop_front();
        }
        Ok(())
    }

    async fn run_lane<T: Send + 'static>(
        self: &Rc<Self>,
        lane: &JavaLane,
        context: &RuntimeContext,
        operation: impl FnOnce(&runmat_java::JavaLaneSession<'_>) -> Result<T, JavaInvocationError>
            + Send
            + 'static,
    ) -> Result<T, RuntimeError> {
        let profiling = crate::foreign::native_lane_profiling::NativeLaneProfiling::admit(context);
        let operation =
            move |owner: &runmat_java::JavaLaneSession<'_>| profiling.execute(|| operation(owner));
        if let Some(authority) = context.java_reentry_for(lane) {
            let admitted = lane
                .submit_reentry(authority, operation)
                .map_err(java_invocation_error)?;
            return self.drive_java_events(admitted, (*context).clone()).await;
        }
        let admitted = lane.submit(operation).map_err(java_invocation_error)?;
        self.drive_java_events(admitted, (*context).clone()).await
    }

    async fn drive_java_events<T: Send + 'static>(
        self: &Rc<Self>,
        admitted: runmat_java::JavaLaneOperation<T>,
        invocation_context: RuntimeContext,
    ) -> Result<T, RuntimeError> {
        let (result, mut events) = admitted.into_parts();
        let mut result = result.fuse();
        let mut callbacks: FuturesUnordered<LocalBoxFuture<'_, ()>> = FuturesUnordered::new();
        loop {
            let event = if callbacks.is_empty() {
                futures::select_biased! {
                    response = result => return response.map_err(|_| java_invocation_error(JavaInvocationError::Callback("Java lane ended before replying".into())))?.map_err(java_invocation_error),
                    event = events.next().fuse() => event,
                }
            } else {
                futures::select_biased! {
                    response = result => return response.map_err(|_| java_invocation_error(JavaInvocationError::Callback("Java lane ended before replying".into())))?.map_err(java_invocation_error),
                    event = events.next().fuse() => event,
                    _ = callbacks.next().fuse() => continue,
                }
            };
            let Some(mut event) = event else {
                return result
                    .await
                    .map_err(|_| {
                        java_invocation_error(JavaInvocationError::Callback(
                            "Java lane ended before replying".into(),
                        ))
                    })?
                    .map_err(java_invocation_error);
            };
            let (token, invocation, completion, authority) = event.take_parts();
            let owner = Rc::clone(self);
            let context = invocation_context.clone();
            callbacks.push(Box::pin(async move {
                let answer = owner
                    .dispatch_java_callback(token, invocation, authority, context)
                    .await;
                completion.finish(answer);
            }));
        }
    }

    async fn dispatch_java_callback(
        self: Rc<Self>,
        token: u64,
        invocation: JavaCallbackInvocation,
        authority: runmat_java::JavaReentryAuthority,
        invocation_context: RuntimeContext,
    ) -> Result<JavaValue, JavaInvocationError> {
        let registration = self
            .callbacks
            .borrow()
            .get(&token)
            .cloned()
            .ok_or_else(|| {
                JavaInvocationError::Callback("Java callback registration is stale".into())
            })?;
        if registration.context.local_identity() != invocation_context.local_identity() {
            return Err(JavaInvocationError::Callback(
                "Java callback invocation belongs to another Runtime owner".into(),
            ));
        }
        let foreign = registration.foreign.upgrade().ok_or_else(|| {
            JavaInvocationError::Callback("callback's originating foreign runtime has ended".into())
        })?;
        let context = registration
            .context
            .clone()
            .with_service_ports(
                registration
                    .context
                    .service_ports()
                    .clone()
                    .with_foreign(foreign),
            )
            .try_with_inherited_callback_invocation_authority_from(&invocation_context)
            .expect("callback Runtime owner checked before authority projection")
            .with_java_reentry(authority);
        let mut arguments =
            crate::execution::RootedValuesBuilder::with_capacity(invocation.arguments.len());
        for value in invocation.arguments {
            arguments
                .push(self.value_from_java(value).map_err(callback_error)?)
                .map_err(|error| JavaInvocationError::Callback(error.to_string()))?;
        }
        let requested_outputs = usize::from(invocation.returns_value);
        arguments
            .finish()
            .map_err(|error| JavaInvocationError::Callback(error.to_string()))?
            .with_unbounded_values_async(|arguments| async move {
                let request = foreign_callback_request(
                    &context,
                    &registration.callable,
                    arguments,
                    requested_outputs,
                )
                .map_err(callback_error)?;
                let sequence = invoke_foreign_callback(context.clone(), request)
                    .await
                    .map_err(callback_error)?;
                sequence.with_owned_sequence(|sequence| {
                    let values = crate::sequence::ResolveValueSequence::resolve(
                        sequence,
                        runmat_types::SequenceUse::SelectPrefix {
                            count: requested_outputs,
                        },
                        crate::sequence::SequenceResolutionContext::default(),
                    )
                    .map_err(|error| JavaInvocationError::Callback(error.to_string()))?;
                    if requested_outputs == 0 {
                        return Ok(JavaValue::Null);
                    }
                    self.argument_to_java(
                        &context,
                        values
                            .try_into_first()
                            .map_err(|_| {
                                JavaInvocationError::Callback(
                                    "shared output storage requires separate admission".into(),
                                )
                            })?
                            .expect("one selected Java callback result"),
                    )
                    .map_err(callback_error)
                })
            })
            .await
            .map_err(callback_error)?
    }
}

fn callback_error(error: RuntimeError) -> JavaInvocationError {
    JavaInvocationError::Callback(error.to_string())
}
