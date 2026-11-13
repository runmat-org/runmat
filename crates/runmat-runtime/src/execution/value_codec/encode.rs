use super::path::{field_path, index_path};
use super::ValueCodecError;
use runmat_execution::value::{
    CallableForm, CallableOwner, CallableValue, DenseValue, DetachedDiagnosticFrame, ElementType,
    ExceptionCallFrame, ExceptionFailureOrigin, ExceptionFrameSource, ExceptionGpuGatherRetry,
    ExceptionRuntimeMetadata, ExceptionValue, InlineValue, RegisteredData, RegisteredField,
    SparseValue, StructField, ValuePayload,
};
use runmat_value::{ComplexStorage, IntValue, IntegerStorage, SparseTensor, Tensor, Value};

#[path = "encode/aggregate.rs"]
mod aggregate;
#[cfg(test)]
pub(super) fn encode_factor_for_test(
    factor: &runmat_value::NumericFactorData,
    path: &str,
) -> Result<runmat_execution::value::DurableNumericFactor, ValueCodecError> {
    aggregate::encode_factor_for_test(factor, path)
}

pub(super) trait InlineSourceAuthority {
    fn session_lease(&self) -> std::sync::Weak<()>;
    fn frame_ordinal(
        &self,
        frame: &crate::CallFrame,
    ) -> Result<runmat_types::ProgramSourceId, &'static str>;
    fn render_call_stack(&self, error: &crate::RuntimeError) -> Vec<String>;
    fn allows_detached_diagnostics(&self) -> bool {
        false
    }
}

impl InlineSourceAuthority for crate::context::RuntimeContext {
    fn session_lease(&self) -> std::sync::Weak<()> {
        self.session_lease()
    }

    fn frame_ordinal(
        &self,
        frame: &crate::CallFrame,
    ) -> Result<runmat_types::ProgramSourceId, &'static str> {
        crate::execution::errors::portable_call_frame_source_in_context(frame, self)
    }

    fn render_call_stack(&self, error: &crate::RuntimeError) -> Vec<String> {
        crate::execution::render_source_call_stack_in_context(error, self)
    }
}

impl InlineSourceAuthority for super::graph::GraphSourceIssuer {
    fn session_lease(&self) -> std::sync::Weak<()> {
        self.session_lease().clone()
    }

    fn frame_ordinal(
        &self,
        frame: &crate::CallFrame,
    ) -> Result<runmat_types::ProgramSourceId, &'static str> {
        let source = frame
            .source_id
            .ok_or("caught exception frame has no source")?;
        self.source_ordinal(runmat_types::SourceId(source), frame.source_owner.as_ref())
            .ok_or("caught exception source is absent from selected graph units")
    }

    fn render_call_stack(&self, error: &crate::RuntimeError) -> Vec<String> {
        self.render_call_stack(error)
    }

    fn allows_detached_diagnostics(&self) -> bool {
        true
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CodecPurpose {
    ExecutionV2,
    DurablePreference,
    DurablePreferencePreparing,
}

impl CodecPurpose {
    fn durable(self) -> bool {
        self != Self::ExecutionV2
    }
}

pub fn encode_inline_value(value: &Value) -> Result<ValuePayload, ValueCodecError> {
    crate::builtins::ensure_builtin_value_classes_registered();
    encode(
        value,
        "$",
        CodecPurpose::ExecutionV2,
        None,
        Some(&super::ValidationWorkspaceOwner::ordinary()),
    )
}

pub fn encode_inline_value_in_context(
    value: &Value,
    runtime: &crate::context::RuntimeContext,
) -> Result<ValuePayload, ValueCodecError> {
    crate::builtins::ensure_builtin_value_classes_registered();
    encode(
        value,
        "$",
        CodecPurpose::ExecutionV2,
        Some(runtime),
        Some(&super::ValidationWorkspaceOwner::capture(runtime)),
    )
}

pub(super) fn encode_inline_value_with_authority(
    value: &Value,
    issuer: &dyn InlineSourceAuthority,
    workspace: Option<&super::ValidationWorkspaceOwner>,
) -> Result<ValuePayload, ValueCodecError> {
    crate::builtins::ensure_builtin_value_classes_registered();
    encode(
        value,
        "$",
        CodecPurpose::ExecutionV2,
        Some(issuer),
        workspace,
    )
}

pub(super) fn encode_durable_inline_value(value: &Value) -> Result<ValuePayload, ValueCodecError> {
    encode(
        value,
        "$",
        CodecPurpose::DurablePreference,
        None,
        Some(&super::ValidationWorkspaceOwner::ordinary()),
    )
}

/// Apply the same durable grammar before resident leaves are downloaded. The
/// placeholder is never persisted; the prepared host value is encoded again.
pub(super) fn encode_durable_preparation_value(
    value: &Value,
) -> Result<ValuePayload, ValueCodecError> {
    encode(
        value,
        "$",
        CodecPurpose::DurablePreferencePreparing,
        None,
        Some(&super::ValidationWorkspaceOwner::ordinary()),
    )
}

fn encode(
    value: &Value,
    path: &str,
    purpose: CodecPurpose,
    runtime: Option<&dyn InlineSourceAuthority>,
    workspace: Option<&super::ValidationWorkspaceOwner>,
) -> Result<ValuePayload, ValueCodecError> {
    let inline = match value {
        Value::Int(value) => encode_integer_scalar(value),
        Value::Num(value) => InlineValue::F64Bits(value.to_bits()),
        Value::Complex(real, imaginary) => InlineValue::ComplexF64Bits {
            real: real.to_bits(),
            imaginary: imaginary.to_bits(),
        },
        Value::Bool(value) => InlineValue::Logical(*value),
        Value::LogicalArray(value) => InlineValue::Dense(DenseValue {
            element_type: ElementType::Logical,
            shape: shape(&value.shape, path)?,
            little_endian_data: value.data.to_vec(),
        }),
        Value::String(value) => InlineValue::String(runmat_execution::value::StringElement(
            value.as_str().map(str::to_owned),
        )),
        Value::StringArray(value) => InlineValue::StringArray {
            shape: shape(&value.shape, path)?,
            values: value
                .data()
                .iter()
                .map(|text| {
                    runmat_execution::value::StringElement(text.as_str().map(str::to_owned))
                })
                .collect(),
        },
        Value::CharArray(value) => InlineValue::Char {
            encoding: runmat_execution::value::CharacterEncoding::Utf16CodeUnitsV1,
            shape: shape(&value.shape, path)?,
            code_units: value.data().iter().map(|value| value.raw()).collect(),
        },
        Value::Tensor(value) => InlineValue::Dense(encode_tensor(value, path)?),
        Value::SparseTensor(value) => InlineValue::Sparse(encode_sparse(value, path)?),
        Value::ComplexTensor(value) => {
            let (element_type, data) = encode_complex_storage(value.complex_storage());
            InlineValue::Dense(DenseValue {
                element_type,
                shape: shape(value.shape(), path)?,
                little_endian_data: data,
            })
        }
        Value::Symbolic(_) | Value::SymbolicArray(_) => {
            return Err(ValueCodecError::unsupported(
                path,
                "a registered symbolic codec is required",
            ))
        }
        Value::Cell(value) => InlineValue::Cell {
            shape: shape(&value.shape, path)?,
            values: value
                .data()
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    encode(value, &index_path(path, index), purpose, runtime, workspace)
                })
                .collect::<Result<_, _>>()?,
        },
        Value::Struct(value) => {
            let mut fields = value
                .fields
                .iter()
                .map(|(name, value)| {
                    Ok(StructField {
                        name: name.to_owned(),
                        value: encode(value, &field_path(path, name), purpose, runtime, workspace)?,
                    })
                })
                .collect::<Result<Vec<_>, ValueCodecError>>()?;
            if purpose.durable() {
                InlineValue::DurableOrderedStruct(fields)
            } else {
                fields.sort_by(|left, right| left.name.cmp(&right.name));
                InlineValue::Struct(fields)
            }
        }
        Value::StructArray(value) if purpose.durable() => InlineValue::DurableStructArray(
            aggregate::struct_array(value, path, purpose, runtime, workspace)?,
        ),
        Value::StructArray(_) => {
            return Err(ValueCodecError::unsupported(
                path,
                "structure arrays require the versioned aggregate-value transport",
            ))
        }
        Value::GpuTensor(_) if purpose == CodecPurpose::DurablePreferencePreparing => {
            InlineValue::Logical(false)
        }
        Value::GpuTensor(_) => {
            return Err(ValueCodecError::unsupported(
                path,
                "GPU buffers require an object reference fenced to a worker and device",
            ))
        }
        Value::Object(value) if purpose.durable() => InlineValue::DurablePlainObject(
            aggregate::plain_object(value, path, purpose, runtime, workspace)?,
        ),
        Value::ObjectArray(value) if purpose.durable() => InlineValue::DurableObjectArray(
            aggregate::object_array(value, path, purpose, runtime, workspace)?,
        ),
        Value::Object(value)
            if crate::parallel::codistributor::is_supported_class(value.class_name()) =>
        {
            crate::parallel::codistributor::validate_definition(value)
                .map_err(|error| ValueCodecError::invalid(path, error.to_string()))?;
            let mut fields = value
                .properties()
                .iter()
                .map(|(name, value)| {
                    Ok(RegisteredField {
                        name: name.to_owned(),
                        value: encode(value, &field_path(path, name), purpose, runtime, workspace)?,
                    })
                })
                .collect::<Result<Vec<_>, ValueCodecError>>()?;
            fields.sort_by(|left, right| left.name.cmp(&right.name));
            InlineValue::ImmutableValueClass(RegisteredData {
                type_identity: value.class_name().to_string(),
                schema_version: 1,
                fields,
            })
        }
        Value::Object(value)
            if super::immutable_class::schema_for_class(value.class_name().display_name())?
                .is_some() =>
        {
            encode_immutable_class(value, path, purpose, runtime, workspace)?
        }
        Value::ObjectArray(_) => {
            return Err(ValueCodecError::unsupported(
                path,
                "value-class arrays require a shape-preserving registered array codec",
            ))
        }
        Value::Object(_) => {
            return Err(ValueCodecError::unsupported(
                path,
                "value classes and object arrays require an explicitly registered codec",
            ))
        }
        Value::HandleObject(_) => {
            return Err(ValueCodecError::unsupported(
                path,
                "mutable handle identity cannot cross an execution boundary",
            ))
        }
        Value::Listener(_) => {
            return Err(ValueCodecError::unsupported(
                path,
                "event listeners are session-bound and cannot be transferred",
            ))
        }
        Value::OutputList(_) => return Err(ValueCodecError::transient_not_portable(path)),
        Value::ScopedFunctionHandle(_) => {
            return Err(ValueCodecError::unsupported(
                path,
                "invocation-scoped callable is not portable",
            ))
        }
        Value::FunctionHandle(name) => callable(
            CallableOwner::Workspace,
            name,
            CallableForm::Handle,
            runmat_types::FunctionSourceRole::Unknown,
            None,
            Vec::new(),
        ),
        Value::ExternalFunctionHandle(name) => callable(
            CallableOwner::External,
            name,
            CallableForm::Handle,
            runmat_types::FunctionSourceRole::Unknown,
            None,
            Vec::new(),
        ),
        Value::MethodFunctionHandle(name) => callable(
            CallableOwner::Method,
            name,
            CallableForm::Handle,
            runmat_types::FunctionSourceRole::Method,
            None,
            Vec::new(),
        ),
        Value::BoundFunctionHandle(handle) => callable(
            CallableOwner::Bound(
                handle
                    .authority
                    .export_origin(&selected_session_lease(runtime))
                    .ok_or_else(|| {
                        ValueCodecError::unsupported(path, "bound callable has no portable origin")
                    })?,
            ),
            &handle.name,
            CallableForm::Handle,
            handle.source_role,
            handle.source_file.as_deref(),
            Vec::new(),
        ),
        Value::Closure(closure) => {
            if closure.lexical_environment.is_some() {
                return Err(ValueCodecError::unsupported(
                    path,
                    "shared lexical environment cannot cross a value-codec boundary",
                ));
            }
            let captures = closure
                .captures
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    encode(
                        value,
                        &format!("{path}.capture[{index}]"),
                        purpose,
                        runtime,
                        workspace,
                    )
                })
                .collect::<Result<_, _>>()?;
            let owner = match &closure.binding {
                Some(binding) => CallableOwner::Bound(
                    binding
                        .authority
                        .export_origin(&selected_session_lease(runtime))
                        .ok_or_else(|| {
                            ValueCodecError::unsupported(path, "closure has no portable origin")
                        })?,
                ),
                None => CallableOwner::Workspace,
            };
            callable(
                owner,
                &closure.function_name,
                CallableForm::Closure,
                closure.source_role,
                closure.source_file.as_deref(),
                captures,
            )
        }
        Value::ClassRef(_) => {
            return Err(ValueCodecError::unsupported(
                path,
                "class metadata references are session-bound",
            ))
        }
        Value::MException(value) => {
            let runtime = if let Some(original) = value.original() {
                if has_opaque_exception_source(original) {
                    return Err(ValueCodecError::unsupported(
                        path,
                        "caught exception has a live opaque cause and cannot be transferred or stored",
                    ));
                }
                Some(exception_runtime_metadata(original, path, runtime)?)
            } else {
                None
            };
            InlineValue::Exception(ExceptionValue {
                identifier: value.identifier.as_str().to_owned(),
                message: value.message.as_str().to_owned(),
                stack: value
                    .stack
                    .iter()
                    .map(|frame| frame.as_str().to_owned())
                    .collect(),
                causes: Vec::new(),
                runtime,
            })
        }
        Value::Future(_) | Value::Task(_) | Value::Pool(_) | Value::Job(_) => {
            return Err(ValueCodecError::unsupported(
                path,
                "execution handles are capabilities, not transferable values",
            ))
        }
        Value::Distributed(handle) => {
            return Ok(ValuePayload::Distributed(Box::new(handle.as_ref().clone())))
        }
        Value::Composite(handle) => {
            return Ok(ValuePayload::Composite(Box::new(handle.as_ref().clone())))
        }
        Value::Foreign(_) => return Err(ValueCodecError::unsupported(
            path,
            "foreign references require an interop-manifest adapter and cannot be encoded inline",
        )),
    };
    Ok(ValuePayload::Inline(Box::new(inline)))
}

fn encode_immutable_class(
    object: &runmat_value::ObjectInstance,
    path: &str,
    purpose: CodecPurpose,
    runtime: Option<&dyn InlineSourceAuthority>,
    workspace: Option<&super::ValidationWorkspaceOwner>,
) -> Result<InlineValue, ValueCodecError> {
    let schema = super::immutable_class::schema_for_class(object.class_name().display_name())?
        .ok_or_else(|| {
            ValueCodecError::unsupported(
                path,
                "value classes require an explicitly registered immutable class codec",
            )
        })?;
    schema.validate_object(object, path, workspace)?;
    let mut fields = object.properties().iter().collect::<Vec<_>>();
    fields.sort_by(|left, right| left.0.cmp(right.0));
    let fields = fields
        .into_iter()
        .map(|(name, value)| {
            Ok(RegisteredField {
                name: name.to_owned(),
                value: encode(value, &field_path(path, name), purpose, runtime, workspace)?,
            })
        })
        .collect::<Result<Vec<_>, ValueCodecError>>()?;
    Ok(InlineValue::ImmutableValueClass(RegisteredData {
        type_identity: schema.type_identity().to_owned(),
        schema_version: schema.schema_version(),
        fields,
    }))
}

fn selected_session_lease(runtime: Option<&dyn InlineSourceAuthority>) -> std::sync::Weak<()> {
    runtime.map_or_else(crate::user_functions::current_session_lease, |runtime| {
        runtime.session_lease()
    })
}

/// A shared rethrow owner is still portable when its actual original is
/// source-less. Do not mistake the owning wrapper for an opaque host cause.
fn has_opaque_exception_source(error: &crate::RuntimeError) -> bool {
    let mut current = error;
    loop {
        match current.source.as_deref() {
            None => return false,
            Some(source) => {
                let Some(shared) = source.downcast_ref::<runmat_async::SharedRuntimeErrorSource>()
                else {
                    return true;
                };
                current = shared.original();
            }
        }
    }
}

fn encode_integer_scalar(value: &IntValue) -> InlineValue {
    match value {
        IntValue::I8(value) => InlineValue::I8(*value),
        IntValue::I16(value) => InlineValue::I16(*value),
        IntValue::I32(value) => InlineValue::I32(*value),
        IntValue::I64(value) => InlineValue::I64(*value),
        IntValue::U8(value) => InlineValue::U8(*value),
        IntValue::U16(value) => InlineValue::U16(*value),
        IntValue::U32(value) => InlineValue::U32(*value),
        IntValue::U64(value) => InlineValue::U64(*value),
    }
}

fn encode_tensor(value: &Tensor, path: &str) -> Result<DenseValue, ValueCodecError> {
    let (element_type, little_endian_data) = if let Some(storage) = value.integer_storage() {
        encode_integer_storage(storage)
    } else if let Some(values) = value.as_f64_slice() {
        (
            ElementType::F64,
            values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect(),
        )
    } else if let Some(values) = value.as_f32_slice() {
        (
            ElementType::F32,
            values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect(),
        )
    } else {
        return Err(ValueCodecError::invalid(path, "unknown tensor storage"));
    };
    Ok(DenseValue {
        element_type,
        shape: shape(value.shape(), path)?,
        little_endian_data,
    })
}

fn encode_sparse(value: &SparseTensor, path: &str) -> Result<SparseValue, ValueCodecError> {
    let (element_type, little_endian_data) = if let Some(storage) = value.integer_storage() {
        encode_integer_storage(storage)
    } else if let Some(values) = value.as_f64_slice() {
        (
            ElementType::F64,
            values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect(),
        )
    } else if let Some(values) = value.as_f32_slice() {
        (
            ElementType::F32,
            values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect(),
        )
    } else if let Some(values) = value.as_complex_f64_slice() {
        (
            ElementType::ComplexF64,
            values
                .iter()
                .flat_map(|value| {
                    let (real, imaginary) = (*value).into();
                    real.to_bits()
                        .to_le_bytes()
                        .into_iter()
                        .chain(imaginary.to_bits().to_le_bytes())
                })
                .collect(),
        )
    } else if let Some(values) = value.as_complex_f32_slice() {
        (
            ElementType::ComplexF32,
            values
                .iter()
                .flat_map(|value| {
                    let (real, imaginary) = (*value).into();
                    real.to_bits()
                        .to_le_bytes()
                        .into_iter()
                        .chain(imaginary.to_bits().to_le_bytes())
                })
                .collect(),
        )
    } else if value.is_logical() {
        (ElementType::Logical, vec![1; value.nnz()])
    } else {
        return Err(ValueCodecError::invalid(path, "unknown sparse storage"));
    };
    Ok(SparseValue {
        element_type,
        rows: value.rows as u64,
        columns: value.cols as u64,
        column_offsets: indices(&value.col_ptrs, path)?,
        row_indices: indices(&value.row_indices, path)?,
        little_endian_data,
    })
}

fn encode_complex_storage(storage: &ComplexStorage) -> (ElementType, Vec<u8>) {
    match storage {
        ComplexStorage::F64(values) => (
            ElementType::ComplexF64,
            values
                .iter()
                .flat_map(|value| {
                    let (real, imag) = (*value).into();
                    real.to_bits()
                        .to_le_bytes()
                        .into_iter()
                        .chain(imag.to_bits().to_le_bytes())
                })
                .collect(),
        ),
        ComplexStorage::F32(values) => (
            ElementType::ComplexF32,
            values
                .iter()
                .flat_map(|value| {
                    let (real, imag) = (*value).into();
                    real.to_bits()
                        .to_le_bytes()
                        .into_iter()
                        .chain(imag.to_bits().to_le_bytes())
                })
                .collect(),
        ),
        ComplexStorage::Integer(storage) => {
            let element_type = match storage.real() {
                IntegerStorage::I8(_) => ElementType::ComplexI8,
                IntegerStorage::I16(_) => ElementType::ComplexI16,
                IntegerStorage::I32(_) => ElementType::ComplexI32,
                IntegerStorage::I64(_) => ElementType::ComplexI64,
                IntegerStorage::U8(_) => ElementType::ComplexU8,
                IntegerStorage::U16(_) => ElementType::ComplexU16,
                IntegerStorage::U32(_) => ElementType::ComplexU32,
                IntegerStorage::U64(_) => ElementType::ComplexU64,
            };
            let (_, real) = encode_integer_storage(storage.real());
            let (_, imag) = encode_integer_storage(storage.imag());
            let width = real.len().checked_div(storage.len().max(1)).unwrap_or(0);
            let mut interleaved = Vec::with_capacity(real.len() + imag.len());
            for index in 0..storage.len() {
                interleaved.extend_from_slice(&real[index * width..(index + 1) * width]);
                interleaved.extend_from_slice(&imag[index * width..(index + 1) * width]);
            }
            (element_type, interleaved)
        }
    }
}

fn encode_integer_storage(storage: &IntegerStorage) -> (ElementType, Vec<u8>) {
    match storage {
        IntegerStorage::I8(values) => (ElementType::I8, values.iter().map(|v| *v as u8).collect()),
        IntegerStorage::I16(values) => (ElementType::I16, le_bytes(values, i16::to_le_bytes)),
        IntegerStorage::I32(values) => (ElementType::I32, le_bytes(values, i32::to_le_bytes)),
        IntegerStorage::I64(values) => (ElementType::I64, le_bytes(values, i64::to_le_bytes)),
        IntegerStorage::U8(values) => (ElementType::U8, values.clone()),
        IntegerStorage::U16(values) => (ElementType::U16, le_bytes(values, u16::to_le_bytes)),
        IntegerStorage::U32(values) => (ElementType::U32, le_bytes(values, u32::to_le_bytes)),
        IntegerStorage::U64(values) => (ElementType::U64, le_bytes(values, u64::to_le_bytes)),
    }
}

fn le_bytes<T: Copy, const N: usize>(values: &[T], encode: fn(T) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|value| encode(*value)).collect()
}

fn callable(
    owner: CallableOwner,
    qualified_name: &str,
    form: CallableForm,
    source_role: runmat_types::FunctionSourceRole,
    source_file: Option<&str>,
    captures: Vec<ValuePayload>,
) -> InlineValue {
    InlineValue::Callable(CallableValue {
        owner,
        qualified_name: qualified_name.into(),
        callable_digest: CallableValue::identity_digest(
            owner,
            qualified_name,
            form,
            source_role,
            source_file,
        ),
        form,
        source_role,
        source_file: source_file.map(ToString::to_string),
        captures,
    })
}

fn exception_runtime_metadata(
    error: &crate::RuntimeError,
    path: &str,
    runtime: Option<&dyn InlineSourceAuthority>,
) -> Result<ExceptionRuntimeMetadata, ValueCodecError> {
    fn portable(value: usize, path: &str) -> Result<u64, ValueCodecError> {
        u64::try_from(value).map_err(|_| {
            ValueCodecError::invalid(path, "exception metadata exceeds portable limits")
        })
    }
    fn span(
        value: Option<(usize, usize)>,
        path: &str,
    ) -> Result<Option<(u64, u64)>, ValueCodecError> {
        value
            .map(|(start, length)| Ok((portable(start, path)?, portable(length, path)?)))
            .transpose()
    }
    let call_frames = error
        .context
        .call_frames
        .iter()
        .map(|frame| {
            let graph = runtime.is_some_and(|runtime| runtime.allows_detached_diagnostics());
            let detached = frame
                .diagnostic_source
                .as_ref()
                .is_some_and(crate::CallFrameDiagnosticSource::is_detached);
            if detached && !graph {
                return Err(ValueCodecError::unsupported(
                    path,
                    "Direct execution transport cannot encode detached diagnostic frames",
                ));
            }
            let selected = if detached {
                None
            } else {
                frame.source_id.map(|_| {
                    let runtime = runtime.ok_or(
                        "caught exception source requires a selected portable program binding",
                    )?;
                    runtime.frame_ordinal(frame)
                })
            };
            let detached_presentation = || {
                frame
                    .diagnostic_source
                    .as_ref()
                    .map(|diagnostic| {
                        let (line, column) = diagnostic
                            .line_column(frame.span)
                            .map_or((None, None), |(line, column)| (Some(line), Some(column)));
                        ExceptionFrameSource::Detached(DetachedDiagnosticFrame {
                            display_name: diagnostic.display_name().to_owned(),
                            line,
                            column,
                        })
                    })
                    .unwrap_or(ExceptionFrameSource::Unknown)
            };
            let source = match selected {
                Some(Ok(ordinal)) => ExceptionFrameSource::Admitted(ordinal),
                Some(Err(error)) => {
                    if !graph || frame.diagnostic_source.is_none() {
                        return Err(ValueCodecError::unsupported(path, error));
                    }
                    detached_presentation()
                }
                None => {
                    if graph {
                        detached_presentation()
                    } else {
                        ExceptionFrameSource::Unknown
                    }
                }
            };
            Ok(ExceptionCallFrame {
                function: frame.function.clone(),
                source,
                span: span(frame.span, path)?,
            })
        })
        .collect::<Result<Vec<_>, ValueCodecError>>()?;
    Ok(ExceptionRuntimeMetadata {
        span: span(
            error
                .span
                .as_ref()
                .map(|value| (value.offset(), value.len())),
            path,
        )?,
        builtin: error.context.builtin.clone(),
        task_id: error.context.task_id.clone(),
        call_frames,
        call_frames_elided: portable(error.context.call_frames_elided, path)?,
        call_stack: if error.context.call_stack.is_empty() {
            runtime.map_or_else(Vec::new, |runtime| runtime.render_call_stack(error))
        } else {
            error.context.call_stack.clone()
        },
        phase: error.context.phase.clone(),
        gpu_gather_retry: match error.gpu_gather_retry {
            crate::GpuGatherRetry::Legacy => ExceptionGpuGatherRetry::Legacy,
            crate::GpuGatherRetry::Never => ExceptionGpuGatherRetry::Never,
            crate::GpuGatherRetry::Requested => ExceptionGpuGatherRetry::Requested,
        },
        failure_origin: match error.failure_origin {
            runmat_async::RuntimeFailureOrigin::Unclassified => {
                ExceptionFailureOrigin::Unclassified
            }
            runmat_async::RuntimeFailureOrigin::LanguageEvaluation => {
                ExceptionFailureOrigin::LanguageEvaluation
            }
            runmat_async::RuntimeFailureOrigin::ExecutionControlCancellation => {
                ExceptionFailureOrigin::ExecutionControlCancellation
            }
        },
    })
}

fn shape(shape: &[usize], path: &str) -> Result<Vec<u64>, ValueCodecError> {
    shape
        .iter()
        .map(|dimension| {
            u64::try_from(*dimension)
                .map_err(|_| ValueCodecError::invalid(path, "shape exceeds portable limits"))
        })
        .collect()
}

fn indices(values: &[usize], path: &str) -> Result<Vec<u64>, ValueCodecError> {
    values
        .iter()
        .map(|value| {
            u64::try_from(*value)
                .map_err(|_| ValueCodecError::invalid(path, "index exceeds portable limits"))
        })
        .collect()
}
