use super::*;

impl NativeFfiAdapter {
    pub(super) fn call(
        &self,
        context: RuntimeContext,
        arguments: &(impl InvocationArguments + ?Sized),
        requested_outputs: usize,
    ) -> Result<runmat_value::ValueSequence, RuntimeError> {
        if arguments.len() < 2 {
            return Err(invalid_call(
                "native FFI call expects a library alias, symbol, and its arguments",
            ));
        }
        let alias = string_argument(arguments, 0, "call")?;
        let symbol_name = string_argument(arguments, 1, "call")?;
        let entry = self.library(&alias)?;
        let prototype = entry.metadata.libraries[0]
            .symbols
            .iter()
            .find(|prototype| {
                prototype.name == symbol_name || prototype.exported_name == symbol_name
            })
            .cloned()
            .ok_or_else(|| invalid_call(format!("unknown native symbol `{symbol_name}`")))?;
        let argument_count = arguments.len() - 2;
        if argument_count != prototype.parameters.len() {
            return Err(invalid_call(format!(
                "native symbol `{symbol_name}` expects {} arguments, received {}",
                prototype.parameters.len(),
                argument_count
            )));
        }
        let declared_outputs = prototype.public_output_count();
        if requested_outputs > declared_outputs {
            return Err(invalid_call(format!(
                "native call provides {declared_outputs} outputs, but {requested_outputs} were requested"
            )));
        }
        let call_arguments = ArgumentWindow {
            source: arguments,
            offset: 2,
        };

        let mut pointer_entries = Vec::new();
        let mut callback_entries = Vec::new();
        for (index, parameter) in prototype.parameters.iter().enumerate() {
            let value = call_arguments
                .get(index)
                .expect("validated native argument count");
            crate::foreign::reject_scoped_callback(value)?;
            match (&parameter.ty, value) {
                (NativeType::Pointer { .. }, Value::Foreign(reference)) => {
                    self.handles.resolve(reference, ForeignCapability::Invoke)?;
                    let pointer = self
                        .state
                        .borrow()
                        .pointers
                        .get(&reference.handle)
                        .cloned()
                        .ok_or_else(|| {
                            foreign_error(
                                ForeignErrorKind::StaleHandle,
                                "native pointer resource has been released",
                            )
                        })?;
                    pointer_entries.push((index, value.clone(), pointer));
                }
                (NativeType::Callback { .. }, value) if is_callable(value) => {
                    callback_entries.push((index, value));
                }
                _ => {}
            }
        }

        let callback_results = Rc::new(RefCell::new(Vec::<crate::execution::RootedValue>::new()));
        // The scalar C ABI cannot carry an exception. Retain its typed owner
        // until native invocation returns instead of reconstructing it from text.
        let callback_error = RefCell::new(None);
        let callback_dispatches = callback_entries
            .iter()
            .map(|(_, callable)| {
                let context = context.clone();
                let callable = *callable;
                let callback_router = self.callback_router.clone();
                let callback_results = Rc::clone(&callback_results);
                let callback_error = &callback_error;
                Box::new(move |callback_arguments: &[Value]| {
                    let rooted = if let (Some(router), Some(callback_id)) =
                        (callback_router.as_ref(), isolated_callback_id(callable))
                    {
                        router.dispatch(callback_id, callback_arguments)
                    } else {
                        pollster::block_on(crate::call_feval_async_with_outputs_in_context(
                            &context,
                            callable.clone(),
                            callback_arguments,
                            1,
                        ))
                    }
                    .map_err(|error| {
                        let message = error.to_string();
                        // Preserve the first originating failure if C invokes
                        // a callback again before returning to the runtime.
                        let mut retained = callback_error.borrow_mut();
                        if retained.is_none() {
                            *retained = Some(error);
                        }
                        message
                    })?;
                    let value = match rooted.value() {
                        value @ (Value::Num(_)
                        | Value::Int(_)
                        | Value::Bool(_)
                        | Value::Tensor(_)) => value.clone(),
                        _ => return Err("native callback requires a real numeric scalar".into()),
                    };
                    // The C callback ABI consumes the returned numeric Value
                    // before invocation ends; keep its producer lease alive.
                    callback_results.borrow_mut().push(rooted);
                    Ok(value)
                }) as Box<dyn runmat_native_ffi::CallbackDispatch + '_>
            })
            .collect::<Vec<_>>();
        let callback_bindings = callback_entries
            .iter()
            .zip(&callback_dispatches)
            .map(|((index, _), dispatch)| CallbackBinding {
                argument_index: *index,
                dispatch: dispatch.as_ref(),
            })
            .collect::<Vec<_>>();
        let pointer_bindings = pointer_entries
            .iter()
            .map(|(index, _, pointer)| match pointer {
                PointerEntry::CallerOwned { pointer, .. } => {
                    PointerBinding::resource(*index, pointer)
                }
                PointerEntry::Opaque { pointer, .. } => PointerBinding::opaque(*index, pointer),
            })
            .collect::<Vec<_>>();

        let result = invoke_symbol_with_bindings(
            &entry.library,
            &prototype,
            &call_arguments,
            &entry.metadata,
            &callback_bindings,
            &pointer_bindings,
        );
        if let Some(error) = callback_error.borrow_mut().take() {
            return Err(error);
        }
        let result = result.map_err(|error| {
            foreign_error_with_source(ForeignErrorKind::InvocationFailed, error)
        })?;
        let mut outputs = Vec::new();
        if let Some(return_value) = result.return_value {
            outputs.push(match return_value {
                InvocationValue::Value(value) => value,
                InvocationValue::Pointer(pointer) => {
                    self.register_opaque_pointer(pointer, &entry)?
                }
            });
        }
        let copied_outputs = result
            .output_parameters
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        for (index, parameter) in prototype.parameters.iter().enumerate() {
            if matches!(
                parameter.direction,
                runmat_native_ffi::ParameterDirection::Output
                    | runmat_native_ffi::ParameterDirection::InputOutput
            ) {
                if let Some(value) = copied_outputs.get(&index) {
                    outputs.push(value.clone());
                } else if let Some((_, reference, _)) =
                    pointer_entries.iter().find(|(bound, _, _)| *bound == index)
                {
                    outputs.push(reference.clone());
                }
            }
        }
        requested_output_sequence(outputs, requested_outputs)
    }
}
pub(super) fn isolated_callback_id(value: &Value) -> Option<u64> {
    let Value::FunctionHandle(name) = value else {
        return None;
    };
    name.strip_prefix("__runmat_native_ffi_isolated_callback_")?
        .parse()
        .ok()
}

pub(super) fn requested_output_sequence(
    outputs: Vec<Value>,
    requested_outputs: usize,
) -> Result<runmat_value::ValueSequence, RuntimeError> {
    if requested_outputs > outputs.len() {
        return Err(invalid_call(format!(
            "native call provides {} outputs, but {requested_outputs} were requested",
            outputs.len()
        )));
    }
    let mut selected = outputs
        .into_iter()
        .take(requested_outputs)
        .collect::<Vec<_>>();
    match requested_outputs {
        0 => Ok(runmat_value::ValueSequence::empty()),
        1 => super::super::super::single_output_sequence(
            selected.pop().expect("validated native output count"),
        ),
        _ => runmat_value::ValueSequence::comma_separated(selected)
            .map_err(crate::sequence::sequence_error_to_runtime),
    }
}
pub(crate) fn is_callable(value: &Value) -> bool {
    matches!(
        value,
        Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
    )
}

pub(crate) fn isolated_callback_value(callback_id: u64) -> Value {
    Value::function_handle(format!(
        "__runmat_native_ffi_isolated_callback_{callback_id}"
    ))
}
