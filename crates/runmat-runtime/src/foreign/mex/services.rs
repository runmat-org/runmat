use runmat_execution::cancellation::CancellationToken;
use runmat_mex::{
    MexCallOutput, MexCancellationScope, MexDiagnostic, MexEngineCompletion, MexHostServices,
};
use runmat_value::Value;

use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease, RootedValueSequence};
mod origin;
pub(super) use origin::{
    OriginMexBoundaryAdapter, OriginMexBoundaryServices, OriginMexCompletionFuture,
    OriginMexOperation,
};

/// Runtime-owned callback authority for one active MEX invocation.
///
/// The exact invocation context is retained so native execution sees its
/// scoped workspace service instead of falling back to a process-global
/// interpreter workspace.
pub struct RuntimeMexHostServices {
    runtime: RuntimeContext,
}

impl RuntimeMexHostServices {
    pub fn new(runtime: RuntimeContext) -> Self {
        Self { runtime }
    }

    pub(crate) async fn call_rooted(
        &self,
        function: &str,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> Result<RootedValueSequence, crate::RuntimeError> {
        let input_roots = RootedValueLease::for_values(arguments.iter())
            .map_err(|error| error.into_runtime_error())?;
        let result = crate::call_feval_sequence_async_with_outputs_in_context(
            &self.runtime,
            Value::from(function),
            &arguments,
            requested_outputs,
        )
        .await;
        drop(input_roots);
        result
    }

    pub(crate) fn with_request_cancellation(&self, cancellation: CancellationToken) -> Self {
        let combined = self.runtime.cancellation().or(&cancellation);
        Self::new(self.runtime.clone().with_cancellation_token(combined))
    }

    pub(crate) async fn eval_rooted(&self, command: String) -> Result<(), crate::RuntimeError> {
        crate::call_builtin_async_with_outputs_in_context(
            &self.runtime,
            "eval",
            &[Value::string(command)],
            0,
        )
        .await
        .map(|_| ())
    }

    pub(crate) async fn get_object_property_rooted(
        &self,
        object: Value,
        index: usize,
        name: String,
    ) -> Result<RootedValue, MexDiagnostic> {
        let object = RootedValue::new(object)
            .map_err(|error| runtime_diagnostic(error.into_runtime_error()))?
            .try_map_owned(select_object_element_at(index), |error| {
                runtime_diagnostic(error.into_runtime_error())
            })?;
        object
            .with_owned_async(|object| {
                self.runtime
                    .scope(crate::object::resolve::load_member_with_context(
                        Some(&self.runtime),
                        object,
                        name,
                        false,
                        None,
                    ))
            })
            .await
            .map_err(runtime_diagnostic)
    }

    pub(crate) async fn set_object_property_rooted(
        &self,
        object: Value,
        index: usize,
        name: String,
        value: Value,
    ) -> Result<RootedValue, MexDiagnostic> {
        let object = RootedValue::new(object)
            .map_err(|error| runtime_diagnostic(error.into_runtime_error()))?;
        let value = RootedValue::new(value)
            .map_err(|error| runtime_diagnostic(error.into_runtime_error()))?;
        object
            .with_owned_async(|object| async {
                match object {
                    Value::ObjectArray(array) => {
                        let selected = array
                            .data()
                            .get(index)
                            .cloned()
                            .ok_or_else(|| object_index_diagnostic(index, array.len()))?;
                        let selected = RootedValue::new(selected)
                            .map_err(|error| runtime_diagnostic(error.into_runtime_error()))?;
                        let updated = selected
                            .with_owned_async(|selected| {
                                value.with_owned_async(|value| {
                                    self.runtime.scope(
                                        crate::object::resolve::store_member_in_context(
                                            &self.runtime,
                                            selected,
                                            name,
                                            value,
                                            None,
                                            |_, _| {},
                                        ),
                                    )
                                })
                            })
                            .await
                            .map_err(runtime_diagnostic)?;
                        updated.with_owned(|updated| {
                            let mut output =
                                crate::execution::object_array::ObjectArrayConstruction::new_retaining_shape(
                                    Some(&self.runtime),
                                    crate::execution::object_array::Class::Owner(
                                        array.class_name_owner(),
                                    ),
                                    array.len(),
                                    array.shape_owner(),
                                    array.selected_provenance().iter(),
                                    array.has_unselected_provenance(),
                                )
                                .map_err(runtime_diagnostic)?;
                            for (ordinal, child) in array.data().iter().enumerate() {
                                output
                                    .push(if ordinal == index {
                                        updated.clone()
                                    } else {
                                        child.clone()
                                    })
                                    .map_err(runtime_diagnostic)?;
                            }
                            output.finish().map_err(runtime_diagnostic)
                        })
                    }
                    object if index == 0 => value
                        .with_owned_async(|value| {
                            self.runtime
                                .scope(crate::object::resolve::store_member_in_context(
                                    &self.runtime,
                                    object,
                                    name,
                                    value,
                                    None,
                                    |_, _| {},
                                ))
                        })
                        .await
                        .map_err(runtime_diagnostic),
                    _ => Err(object_index_diagnostic(index, 1)),
                }
            })
            .await
    }
}

fn select_object_element_at(index: usize) -> impl FnOnce(Value) -> Result<Value, MexDiagnostic> {
    move |object| select_object_element(object, index)
}

struct RuntimeMexCallOutput {
    sequence: RootedValueSequence,
    requested_outputs: usize,
}

struct RuntimeMexValueOutput(RootedValue);

impl MexCallOutput for RuntimeMexValueOutput {
    fn publish(
        self: Box<Self>,
        sink: &mut dyn FnMut(Value) -> Result<(), MexDiagnostic>,
    ) -> Result<(), MexDiagnostic> {
        self.0.with_owned(sink)
    }
}

fn rooted_output(value: Value) -> Result<Box<dyn MexCallOutput>, MexDiagnostic> {
    Ok(Box::new(RuntimeMexValueOutput(
        RootedValue::new(value).map_err(|error| runtime_diagnostic(error.into_runtime_error()))?,
    )))
}

impl MexCallOutput for RuntimeMexCallOutput {
    fn publish(
        self: Box<Self>,
        sink: &mut dyn FnMut(Value) -> Result<(), MexDiagnostic>,
    ) -> Result<(), MexDiagnostic> {
        let Self {
            sequence,
            requested_outputs,
        } = *self;
        sequence.with_owned_sequence(|sequence| {
            let values = crate::sequence::ResolveValueSequence::resolve(
                sequence,
                runmat_types::SequenceUse::SelectPrefix {
                    count: requested_outputs,
                },
                crate::sequence::SequenceResolutionContext::default(),
            )
            .map_err(runtime_diagnostic)?;
            let values = values.try_into_owned().map_err(|_| {
                runtime_diagnostic(crate::sequence::sequence_error_to_runtime(
                    runmat_value::ValueSequenceError::SharedOutputStorage,
                ))
            })?;
            for value in values {
                sink(value)?;
            }
            Ok(())
        })
    }
}

impl MexHostServices for RuntimeMexHostServices {
    fn cancellation_scope(&self, cancellation: CancellationToken) -> Box<dyn MexCancellationScope> {
        let combined = self.runtime.cancellation().or(&cancellation);
        let previous = self.runtime.replace_cancellation_token(combined);
        Box::new(RuntimeMexCancellationScope {
            runtime: self.runtime.clone(),
            previous: Some(previous),
        })
    }

    fn eval(&self, command: &str) -> Result<(), MexDiagnostic> {
        pollster::block_on(self.eval_rooted(command.to_string())).map_err(runtime_diagnostic)
    }

    fn eval_captured(&self, command: &str) -> MexEngineCompletion<()> {
        captured_completion(&self.runtime, || self.eval(command))
    }

    fn call(
        &self,
        function: &str,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> Result<Box<dyn MexCallOutput>, MexDiagnostic> {
        let sequence = pollster::block_on(self.call_rooted(function, arguments, requested_outputs))
            .map_err(runtime_diagnostic)?;
        Ok(Box::new(RuntimeMexCallOutput {
            sequence,
            requested_outputs,
        }))
    }

    fn call_captured(
        &self,
        function: &str,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> MexEngineCompletion<Box<dyn MexCallOutput>> {
        captured_completion(&self.runtime, || {
            self.call(function, arguments, requested_outputs)
        })
    }

    fn get_variable(
        &self,
        workspace: &str,
        name: &str,
    ) -> Result<Option<Box<dyn MexCallOutput>>, MexDiagnostic> {
        let workspace = MexWorkspace::parse(workspace)?;
        let _scope = self.runtime.enter();
        let value = match workspace {
            MexWorkspace::Caller => self
                .runtime
                .service_ports()
                .require_workspace("read the MEX caller workspace")
                .map_err(capability_diagnostic)?
                .try_lookup_with_context(&self.runtime, name)
                .map_err(runtime_diagnostic),
            MexWorkspace::Base => {
                let owner = self
                    .runtime
                    .service_ports()
                    .require_base_workspace("read the MEX base workspace")
                    .map_err(capability_diagnostic)?
                    .owner();
                crate::workspace::lookup_logical_base_value(
                    &owner,
                    &self.runtime.session_variable_authority(),
                    name,
                )
                .map_err(runtime_diagnostic)
            }
            MexWorkspace::Global => self
                .runtime
                .session_variable_authority()
                .global_value(name)
                .map_err(runtime_diagnostic),
        }?;
        value.map(rooted_output).transpose()
    }

    fn put_variable(&self, workspace: &str, name: &str, value: Value) -> Result<(), MexDiagnostic> {
        let workspace = MexWorkspace::parse(workspace)?;
        let _scope = self.runtime.enter();
        match workspace {
            MexWorkspace::Caller => self
                .runtime
                .service_ports()
                .require_workspace("write the MEX caller workspace")
                .map_err(capability_diagnostic)?
                .assign_with_context(&self.runtime, name, value)
                .map_err(runtime_diagnostic),
            MexWorkspace::Base => {
                let base = self
                    .runtime
                    .service_ports()
                    .require_base_workspace("write the MEX base workspace")
                    .map_err(capability_diagnostic)?;
                base.assign_logical_ref_with_context(
                    &self.runtime.session_variable_authority(),
                    &self.runtime,
                    name,
                    &value,
                )
                .map_err(runtime_diagnostic)
            }
            MexWorkspace::Global => {
                crate::execution::validate_storable_value(&value).map_err(runtime_diagnostic)?;
                self.runtime
                    .session_variable_authority()
                    .store_global_with_context(&self.runtime, name, value)
                    .map_err(runtime_diagnostic)
            }
        }
    }

    fn get_object_property(
        &self,
        object: Value,
        name: &str,
    ) -> Result<Box<dyn MexCallOutput>, MexDiagnostic> {
        let value = pollster::block_on(self.get_object_property_rooted(object, 0, name.into()))?;
        Ok(Box::new(RuntimeMexValueOutput(value)))
    }

    fn get_object_property_at(
        &self,
        object: Value,
        index: usize,
        name: &str,
    ) -> Result<Box<dyn MexCallOutput>, MexDiagnostic> {
        let value =
            pollster::block_on(self.get_object_property_rooted(object, index, name.into()))?;
        Ok(Box::new(RuntimeMexValueOutput(value)))
    }

    fn set_object_property(
        &self,
        object: Value,
        name: &str,
        value: Value,
    ) -> Result<Box<dyn MexCallOutput>, MexDiagnostic> {
        let value =
            pollster::block_on(self.set_object_property_rooted(object, 0, name.into(), value))?;
        Ok(Box::new(RuntimeMexValueOutput(value)))
    }

    fn set_object_property_at(
        &self,
        object: Value,
        index: usize,
        name: &str,
        value: Value,
    ) -> Result<Box<dyn MexCallOutput>, MexDiagnostic> {
        let object =
            pollster::block_on(self.set_object_property_rooted(object, index, name.into(), value))?;
        Ok(Box::new(RuntimeMexValueOutput(object)))
    }
}

fn captured_completion<T>(
    runtime: &RuntimeContext,
    operation: impl FnOnce() -> Result<T, MexDiagnostic>,
) -> MexEngineCompletion<T> {
    if let Err(error) = prepare_legacy_capture_materialization(runtime) {
        return MexEngineCompletion::from_result(Err(runtime_diagnostic(error)));
    }
    let capture = match crate::console::begin_capture() {
        Ok(capture) => capture,
        Err(error) => return MexEngineCompletion::from_result(Err(runtime_diagnostic(error))),
    };
    let result = operation();
    let entries = capture.finish_entries();
    let mut stdout = String::new();
    let mut stderr = String::new();
    for entry in entries {
        match entry.stream {
            crate::console::ConsoleStream::Stdout => stdout.push_str(&entry.text),
            crate::console::ConsoleStream::Stderr => stderr.push_str(&entry.text),
            crate::console::ConsoleStream::ClearScreen => {}
        }
    }
    MexEngineCompletion {
        result,
        stdout,
        stderr,
    }
}

fn prepare_legacy_capture_materialization(
    runtime: &RuntimeContext,
) -> Result<(), crate::RuntimeError> {
    use crate::context::host_allocation::{
        host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
    };
    let policy = HostAllocationReceipt::policy(Some(runtime)).map_err(|error| {
        host_allocation_runtime_error(
            Some(runtime),
            "RunMat:mex:CaptureUnavailable",
            "MEX captured text policy could not be established",
            error,
        )
    })?;
    if policy == HostAllocationPolicy::Managed {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:mex:ManagedCaptureUnavailable",
            "MEX captured text requires an admitted language-string owner",
        ));
    }
    Ok(())
}

struct RuntimeMexCancellationScope {
    runtime: RuntimeContext,
    previous: Option<CancellationToken>,
}

impl MexCancellationScope for RuntimeMexCancellationScope {}

impl Drop for RuntimeMexCancellationScope {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            self.runtime.replace_cancellation_token(previous);
        }
    }
}

fn select_object_element(object: Value, index: usize) -> Result<Value, MexDiagnostic> {
    match object {
        Value::ObjectArray(array) => array
            .get_linear(index)
            .cloned()
            .ok_or_else(|| object_index_diagnostic(index, array.len())),
        object if index == 0 => Ok(object),
        _ => Err(object_index_diagnostic(index, 1)),
    }
}

fn object_index_diagnostic(index: usize, length: usize) -> MexDiagnostic {
    MexDiagnostic {
        identifier: Some("RunMat:MEX:ObjectIndex".into()),
        message: format!(
            "zero-based object index {index} is outside an object array with {length} elements"
        ),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MexWorkspace {
    Base,
    Caller,
    Global,
}

impl MexWorkspace {
    fn parse(workspace: &str) -> Result<Self, MexDiagnostic> {
        match workspace {
            "base" => Ok(Self::Base),
            "caller" => Ok(Self::Caller),
            "global" => Ok(Self::Global),
            _ => Err(MexDiagnostic {
                identifier: Some("RunMat:MEX:Workspace".into()),
                message: format!("unknown MEX workspace '{workspace}'"),
            }),
        }
    }
}

fn capability_diagnostic(error: crate::context::RuntimeCapabilityError) -> MexDiagnostic {
    runtime_diagnostic(error.into_runtime_error())
}

pub(crate) fn runtime_diagnostic(error: crate::RuntimeError) -> MexDiagnostic {
    MexDiagnostic {
        identifier: error.identifier().map(str::to_string),
        message: error.message().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{RuntimeServicePorts, RuntimeWorkspaceService};
    use crate::execution::RuntimeExecutionService;
    use runmat_value::{HandleRef, ObjectInstance};
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::fs;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    fn published_root(output: Box<dyn MexCallOutput>) -> Result<RootedValue, MexDiagnostic> {
        let mut rooted = None;
        output.publish(&mut |value| {
            rooted = Some(RootedValue::new(value).map_err(|error| MexDiagnostic {
                identifier: Some("RunMat:MEX:ValueRoot".into()),
                message: error.to_string(),
            })?);
            Ok(())
        })?;
        rooted.ok_or_else(|| MexDiagnostic {
            identifier: Some("RunMat:MEX:CallbackProtocol".into()),
            message: "expected one published value".into(),
        })
    }

    impl RuntimeMexHostServices {
        fn get_variable_rooted(
            &self,
            workspace: &str,
            name: &str,
        ) -> Result<Option<RootedValue>, MexDiagnostic> {
            self.get_variable(workspace, name)?
                .map(published_root)
                .transpose()
        }
    }

    #[test]
    fn mex_output_publication_precedes_collecting_root_retirement() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("mex output".into()))
                .expect("allocate managed output");
            let handle = initial.handle();
            let sequence = RootedValueSequence::new(
                runmat_value::ValueSequence::single(Value::HandleObject(HandleRef {
                    class_name: "MexOutput".into(),
                    target: handle,
                    valid: true,
                }))
                .expect("single output"),
            )
            .expect("root callback output");
            initial.unroot().expect("transfer original root");

            let collected_during_transfer = Arc::new(AtomicBool::new(false));
            let observed = Arc::clone(&collected_during_transfer);
            let identity = (handle.addr(), handle.epoch());
            let _observer = runmat_gc::RootRetirementObserver::register(Arc::new(move |event| {
                let runmat_gc::RootRetirement::Explicit(retired) = event else {
                    return;
                };
                if (retired.addr(), retired.epoch()) == identity
                    && !observed.swap(true, Ordering::SeqCst)
                {
                    runmat_gc::gc_collect_major().expect("collect during output retirement");
                    assert!(runmat_gc::gc_clone_value(&retired).is_ok());
                }
            }))
            .expect("register collecting observer");

            let output = Box::new(RuntimeMexCallOutput {
                sequence,
                requested_outputs: 1,
            });
            let mut sink_root = None;
            let mut sink_value = None;
            output
                .publish(&mut |value| {
                    sink_root = Some(
                        RootedValueLease::for_values(std::iter::once(&value))
                            .expect("register sink before source retirement"),
                    );
                    sink_value = Some(value);
                    Ok(())
                })
                .expect("publish managed output");
            assert!(collected_during_transfer.load(Ordering::SeqCst));
            assert!(matches!(sink_value, Some(Value::HandleObject(_))));
            drop(sink_root);
            runmat_gc::gc_collect_major().expect("collect after sink retirement");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }

    #[test]
    fn mex_property_output_registers_sink_root_before_source_retirement() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("property".into()))
                .expect("allocate managed property");
            let handle = initial.handle();
            let value = Value::HandleObject(HandleRef {
                class_name: "MexProperty".into(),
                target: handle,
                valid: true,
            });
            let output = Box::new(RuntimeMexValueOutput(
                RootedValue::new(value).expect("root producer property"),
            ));
            initial.unroot().expect("transfer original root");

            let observed = Arc::new(AtomicBool::new(false));
            let collected = Arc::clone(&observed);
            let identity = (handle.addr(), handle.epoch());
            let _observer = runmat_gc::RootRetirementObserver::register(Arc::new(move |event| {
                let runmat_gc::RootRetirement::Explicit(retired) = event else {
                    return;
                };
                if (retired.addr(), retired.epoch()) == identity
                    && !collected.swap(true, Ordering::SeqCst)
                {
                    runmat_gc::gc_collect_major().expect("collect at property source retirement");
                    assert!(runmat_gc::gc_clone_value(&retired).is_ok());
                }
            }))
            .expect("register collecting observer");

            let sink = published_root(output).expect("publish rooted property");
            assert!(observed.load(Ordering::SeqCst));
            assert!(matches!(sink.value(), Value::HandleObject(_)));
            drop(sink);
            runmat_gc::gc_collect_major().expect("collect after sink retirement");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }

    #[derive(Default)]
    struct ScopedWorkspace {
        values: RefCell<BTreeMap<String, Value>>,
    }

    impl RuntimeWorkspaceService for ScopedWorkspace {
        fn visit_retained_values(
            &self,
            visitor: &mut dyn FnMut(&Value) -> Option<()>,
        ) -> Option<()> {
            for value in self.values.try_borrow().ok()?.values() {
                visitor(value)?;
            }
            Some(())
        }
        fn lookup(&self, name: &str) -> Option<Value> {
            self.values.borrow().get(name).cloned()
        }

        fn snapshot(&self) -> Vec<(String, Value)> {
            self.values
                .borrow()
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect()
        }

        fn global_names(&self) -> Vec<String> {
            Vec::new()
        }

        fn assign(&self, name: &str, value: Value) -> Result<(), crate::RuntimeError> {
            self.values.borrow_mut().insert(name.to_string(), value);
            Ok(())
        }

        fn clear(&self) -> Result<(), crate::RuntimeError> {
            self.values.borrow_mut().clear();
            Ok(())
        }

        fn remove(&self, name: &str) -> Result<(), crate::RuntimeError> {
            self.values.borrow_mut().remove(name);
            Ok(())
        }
    }

    #[test]
    fn workspace_callbacks_keep_base_caller_and_global_namespaces_distinct() {
        let caller = Rc::new(ScopedWorkspace::default());
        let base_owner = crate::workspace::BaseWorkspaceOwner::new();
        let base = Rc::new(crate::workspace::BaseWorkspaceService::new(
            base_owner.clone(),
        ));
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(
                RuntimeServicePorts::default()
                    .with_workspace(caller.clone())
                    .with_base_workspace(base),
            );
        let services = RuntimeMexHostServices::new(runtime);

        services
            .put_variable("caller", "answer", Value::Num(42.0))
            .expect("write scoped caller workspace");
        services
            .put_variable("base", "answer", Value::Num(7.0))
            .expect("write session base workspace");
        services
            .put_variable("global", "answer", Value::Num(3.0))
            .expect("write session global workspace");
        assert_eq!(
            services
                .get_variable_rooted("base", "answer")
                .expect("read scoped base workspace")
                .as_ref()
                .map(|rooted| rooted.value()),
            Some(&Value::Num(7.0))
        );
        assert_eq!(
            services
                .get_variable_rooted("caller", "answer")
                .expect("read scoped caller workspace")
                .as_ref()
                .map(|rooted| rooted.value()),
            Some(&Value::Num(42.0))
        );
        assert_eq!(
            services
                .get_variable_rooted("global", "answer")
                .expect("read scoped global workspace")
                .as_ref()
                .map(|rooted| rooted.value()),
            Some(&Value::Num(3.0))
        );
        assert_eq!(caller.lookup("answer"), Some(Value::Num(42.0)));
        assert_eq!(base_owner.lookup("answer").unwrap(), Some(Value::Num(7.0)));
    }

    #[test]
    fn base_callbacks_resolve_bound_globals_through_the_session_value_owner() {
        let base_owner = crate::workspace::BaseWorkspaceOwner::new();
        let mut binding = crate::workspace::WorkspaceDelta::new();
        binding.bind_global("answer");
        base_owner.apply(binding).unwrap();
        let base = Rc::new(crate::workspace::BaseWorkspaceService::new(
            base_owner.clone(),
        ));
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_base_workspace(base));
        let services = RuntimeMexHostServices::new(runtime);

        services
            .put_variable("global", "answer", Value::Num(3.0))
            .expect("initialize session-global value");
        assert_eq!(
            services
                .get_variable_rooted("base", "answer")
                .expect("base binding resolves through session-global storage")
                .as_ref()
                .map(|rooted| rooted.value()),
            Some(&Value::Num(3.0))
        );

        services
            .put_variable("base", "answer", Value::Num(11.0))
            .expect("base write updates its bound global");
        assert_eq!(
            services
                .get_variable_rooted("global", "answer")
                .expect("read updated session-global value")
                .as_ref()
                .map(|rooted| rooted.value()),
            Some(&Value::Num(11.0))
        );
        assert!(base_owner.is_global("answer").unwrap());
        assert_eq!(base_owner.lookup("answer").unwrap(), None);
    }

    #[test]
    fn base_workspace_callbacks_never_fall_back_to_the_caller() {
        let caller = Rc::new(ScopedWorkspace::default());
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_workspace(caller.clone()));
        let services = RuntimeMexHostServices::new(runtime);

        services
            .put_variable("caller", "answer", Value::Num(42.0))
            .expect("caller workspace remains available");
        let error = services
            .get_variable_rooted("base", "answer")
            .expect_err("missing base authority must not read the caller");
        assert_eq!(
            error.identifier.as_deref(),
            Some(crate::context::RuntimeCapabilityError::IDENTIFIER)
        );
        assert_eq!(caller.lookup("answer"), Some(Value::Num(42.0)));
    }

    #[test]
    fn base_revocation_does_not_revoke_the_retained_caller_service() {
        let caller = Rc::new(ScopedWorkspace::default());
        let base_owner = crate::workspace::BaseWorkspaceOwner::new();
        let base = Rc::new(crate::workspace::BaseWorkspaceService::new(
            base_owner.clone(),
        ));
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(
                RuntimeServicePorts::default()
                    .with_workspace(caller.clone())
                    .with_base_workspace(base),
            );
        let services = RuntimeMexHostServices::new(runtime);
        base_owner.close().unwrap();

        services
            .put_variable("caller", "answer", Value::Num(42.0))
            .expect("caller remains independently live");
        let error = services
            .put_variable("base", "answer", Value::Num(7.0))
            .expect_err("closed base authority must reject mutation");
        assert_eq!(error.identifier.as_deref(), Some("RunMat:Workspace:Closed"));
        assert_eq!(caller.lookup("answer"), Some(Value::Num(42.0)));
    }

    #[test]
    fn missing_caller_authority_does_not_hide_the_explicit_base_port() {
        let base_owner = crate::workspace::BaseWorkspaceOwner::new();
        let base = Rc::new(crate::workspace::BaseWorkspaceService::new(
            base_owner.clone(),
        ));
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_base_workspace(base));
        let services = RuntimeMexHostServices::new(runtime);

        services
            .put_variable("base", "answer", Value::Num(7.0))
            .expect("explicit base authority remains available");
        let error = services
            .get_variable_rooted("caller", "answer")
            .expect_err("missing caller authority must not fall back to base");
        assert_eq!(
            error.identifier.as_deref(),
            Some(crate::context::RuntimeCapabilityError::IDENTIFIER)
        );
        assert_eq!(base_owner.lookup("answer").unwrap(), Some(Value::Num(7.0)));
    }

    #[test]
    fn global_workspace_callbacks_are_session_scoped() {
        let first = RuntimeMexHostServices::new(RuntimeContext::new(Rc::new(
            RuntimeExecutionService::new(),
        )));
        let second = RuntimeMexHostServices::new(RuntimeContext::new(Rc::new(
            RuntimeExecutionService::new(),
        )));

        first
            .put_variable("global", "answer", Value::Num(3.0))
            .unwrap();
        assert_eq!(
            first
                .get_variable_rooted("global", "answer")
                .unwrap()
                .as_ref()
                .map(|rooted| rooted.value()),
            Some(&Value::Num(3.0))
        );
        assert_eq!(
            second
                .get_variable_rooted("global", "answer")
                .unwrap()
                .as_ref()
                .map(|rooted| rooted.value()),
            None
        );
    }

    #[test]
    fn workspace_callbacks_reject_unknown_workspace_names() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let services = RuntimeMexHostServices::new(runtime);
        let error = services
            .get_variable_rooted("workspace-typo", "answer")
            .expect_err("unknown workspace must fail");

        assert_eq!(error.identifier.as_deref(), Some("RunMat:MEX:Workspace"));
    }

    #[test]
    fn asynchronous_callback_cancellation_is_scoped_and_restored() {
        let original = runmat_execution::cancellation::CancellationSource::new();
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(RuntimeExecutionService::new()),
            original.token(),
        );
        let services = RuntimeMexHostServices::new(runtime.clone());
        let request = runmat_execution::cancellation::CancellationSource::new();

        {
            let _scope = services.cancellation_scope(request.token());
            assert!(!runtime.is_cancelled());
            request.cancel(runmat_execution::CancellationReason::User);
            assert!(runtime.is_cancelled());
        }

        assert!(!runtime.is_cancelled());
        original.cancel(runmat_execution::CancellationReason::Shutdown);
        assert!(runtime.is_cancelled());
    }

    #[test]
    fn asynchronous_callback_keeps_parent_and_nested_cancellation_authorities() {
        let parent = runmat_execution::cancellation::CancellationSource::new();
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(RuntimeExecutionService::new()),
            parent.token(),
        );
        let services = RuntimeMexHostServices::new(runtime.clone());
        let request = runmat_execution::cancellation::CancellationSource::new();
        let nested = runmat_execution::cancellation::CancellationSource::new();
        {
            let _request = services.cancellation_scope(request.token());
            {
                let _nested = services.cancellation_scope(nested.token());
                nested.cancel(runmat_execution::CancellationReason::Deadline);
                assert!(runtime.is_cancelled());
            }
            assert!(!runtime.is_cancelled(), "nested cancellation stays scoped");
            parent.cancel(runmat_execution::CancellationReason::Shutdown);
            assert!(
                runtime.is_cancelled(),
                "parent remains visible during request"
            );
        }
        assert_eq!(
            runtime.cancellation().reason(),
            Some(runmat_execution::CancellationReason::Shutdown)
        );
        assert!(
            !request.token().is_cancelled(),
            "composition does not cancel siblings"
        );
    }

    fn register_fixture_handle_class(runtime: &RuntimeContext) {
        pollster::block_on(runtime.scope(async {
            crate::compatibility::set_runmat_extensions_enabled(false);
            let mut class = crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: "FixtureHandle".into(),
                parent: None,
                properties: Default::default(),
                methods: Default::default(),
            };
            class.properties.insert(
                "Value".into(),
                crate::class_registry::RuntimeProperty {
                    name: "Value".into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: runmat_types::MemberAccess::Public,
                    set_access: runmat_types::MemberAccess::Public,
                    default_value: None,
                },
            );
            crate::class_registry::register_class(class);
        }));
    }

    #[test]
    fn object_callbacks_preserve_handle_identity_and_mutate_its_target() {
        runmat_gc::gc_test_context(|| {
            let mut target = ObjectInstance::new("FixtureHandle");
            target.replace_property_unbounded("Value".into(), Value::Num(4.0));
            let target_root =
                runmat_gc::gc_allocate_rooted(Value::Object(target)).expect("root handle target");
            let handle = HandleRef {
                class_name: "FixtureHandle".into(),
                target: target_root.handle(),
                valid: true,
            };
            let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
            register_fixture_handle_class(&runtime);
            let services = RuntimeMexHostServices::new(runtime);

            assert_eq!(
                pollster::block_on(services.get_object_property_rooted(
                    Value::HandleObject(handle.clone()),
                    0,
                    "Value".into(),
                ))
                .expect("read handle property")
                .value(),
                &Value::Num(4.0)
            );
            let updated = pollster::block_on(services.set_object_property_rooted(
                Value::HandleObject(handle.clone()),
                0,
                "Value".into(),
                Value::Num(9.0),
            ))
            .expect("write handle property");
            let Value::HandleObject(updated) = updated.value() else {
                panic!("handle property assignment must return the same handle kind");
            };
            assert_eq!(updated, &handle);
            assert_eq!(
                pollster::block_on(services.get_object_property_rooted(
                    Value::HandleObject(handle),
                    0,
                    "Value".into(),
                ))
                .expect("read updated handle property")
                .value(),
                &Value::Num(9.0)
            );
            drop(target_root);
        });
    }

    #[test]
    fn indexed_object_callbacks_select_exact_zero_based_element() {
        runmat_gc::gc_test_context(|| {
            let make_handle = |value| {
                let mut target = ObjectInstance::new("FixtureHandle");
                target.replace_property_unbounded("Value".into(), Value::Num(value));
                let target_root = runmat_gc::gc_allocate_rooted(Value::Object(target))
                    .expect("root indexed handle target");
                (
                    HandleRef {
                        class_name: "FixtureHandle".into(),
                        target: target_root.handle(),
                        valid: true,
                    },
                    target_root,
                )
            };
            let (first, first_root) = make_handle(2.0);
            let (second, second_root) = make_handle(4.0);
            let array = runmat_value::ObjectArray::from_handles(
                "FixtureHandle",
                vec![first.clone(), second.clone()],
                vec![1, 2],
            )
            .expect("valid handle array");
            let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
            register_fixture_handle_class(&runtime);
            let services = RuntimeMexHostServices::new(runtime);

            assert_eq!(
                pollster::block_on(services.get_object_property_rooted(
                    Value::ObjectArray(array.clone()),
                    1,
                    "Value".into(),
                ))
                .expect("read indexed property")
                .value(),
                &Value::Num(4.0)
            );
            let updated = pollster::block_on(services.set_object_property_rooted(
                Value::ObjectArray(array),
                1,
                "Value".into(),
                Value::Num(9.0),
            ))
            .expect("write indexed property");
            let Value::ObjectArray(updated) = updated.value() else {
                panic!("indexed property write must preserve the object array");
            };
            assert_eq!(updated.shape(), &[1, 2]);
            runmat_gc::gc_with_value(&first.target, |value| {
                let Value::Object(value) = value else {
                    panic!("first indexed target must remain an object");
                };
                assert_eq!(value.properties().get("Value"), Some(&Value::Num(2.0)));
            })
            .expect("inspect first indexed target");
            runmat_gc::gc_with_value(&second.target, |value| {
                let Value::Object(value) = value else {
                    panic!("second indexed target must remain an object");
                };
                assert_eq!(value.properties().get("Value"), Some(&Value::Num(9.0)));
            })
            .expect("inspect second indexed target");
            drop((first_root, second_root));
        });
    }

    #[test]
    fn compiled_cpp_engine_properties_preserve_runtime_handle_identity() {
        runmat_gc::gc_test_context(|| {
            let directory = tempfile::tempdir().expect("temporary C++ MEX directory");
            let source = directory.path().join("handle_properties.cpp");
            fs::write(
                &source,
                r#"
#include "mex.hpp"
#include "mexAdapter.hpp"

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        matlab::data::Array object = inputs[0];
        if (object.getType() != matlab::data::ArrayType::HANDLE_OBJECT_REF) {
            throw matlab::Exception("input is not a handle object");
        }
        auto engine = getEngine();
        matlab::data::TypedArray<double> before =
            engine->getProperty(object, u"Value");
        if (static_cast<double>(before[0]) != 4.0) {
            throw matlab::Exception("handle property was not retained");
        }
        matlab::data::ArrayFactory factory;
        engine->setProperty(object, u"Value", factory.createScalar<double>(9.0));
        outputs[0] = object;
    }
};
"#,
            )
            .expect("write C++ MEX fixture");

            let mut target = ObjectInstance::new("FixtureHandle");
            target.replace_property_unbounded("Value".into(), Value::Num(4.0));
            let target_root =
                runmat_gc::gc_allocate_rooted(Value::Object(target)).expect("root handle target");
            let target = target_root.handle();
            let handle = HandleRef {
                class_name: "FixtureHandle".into(),
                target,
                valid: true,
            };
            let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
            register_fixture_handle_class(&runtime);
            let services = Rc::new(RuntimeMexHostServices::new(runtime));
            let artifact = runmat_mex::MexBuild::new(&source, directory.path())
                .compile()
                .expect("compile C++ MEX fixture");
            let module =
                runmat_mex::MexModule::load(&artifact.module).expect("load C++ MEX fixture");
            let invocation = module
                .invoke_with_services(
                    &[Value::HandleObject(handle.clone())],
                    1,
                    module.api_mode(),
                    services,
                )
                .expect("invoke C++ MEX fixture");
            let [Value::HandleObject(output)] = invocation.outputs.as_slice() else {
                panic!("C++ MEX output must retain handle-object identity");
            };
            assert_eq!(output, &handle);
            runmat_gc::gc_with_value(&handle.target, |value| {
                let Value::Object(value) = value else {
                    panic!("handle target must remain an object");
                };
                assert_eq!(value.properties().get("Value"), Some(&Value::Num(9.0)));
            })
            .expect("inspect updated handle target");
            drop(target_root);
        });
    }
}
