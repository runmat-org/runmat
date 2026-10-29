use super::pointer_roles::{PointerMember, PointerMethod};
use super::*;

impl NativeFfiAdapter {
    pub(super) fn pointer_value(
        &self,
        arguments: &(impl InvocationArguments + ?Sized),
    ) -> Result<Value, RuntimeError> {
        if arguments.len() != 1 {
            return Err(invalid_call(
                "native FFI pointer_value expects one pointer resource",
            ));
        }
        let Some(Value::Foreign(reference)) = arguments.get(0) else {
            return Err(invalid_call("pointer_value requires a foreign pointer"));
        };
        self.handles.resolve(reference, ForeignCapability::Read)?;
        let entry = self
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
        match entry {
            PointerEntry::CallerOwned {
                pointer, metadata, ..
            } => pointer
                .value(&metadata)
                .map_err(|error| foreign_error_with_source(ForeignErrorKind::InvalidCall, error)),
            PointerEntry::Opaque { .. } => Err(invalid_call(
                "opaque library pointer values cannot be read without an explicit copy contract",
            )),
        }
    }

    pub(super) fn get_member(
        &self,
        arguments: &(impl InvocationArguments + ?Sized),
    ) -> Result<Value, RuntimeError> {
        if arguments.len() != 2 {
            return Err(invalid_call(
                "native FFI get_member expects a foreign resource and member name",
            ));
        }
        let member = PointerMember::parse(string_argument(arguments, 1, "get_member")?);
        let Some(Value::Foreign(reference)) = arguments.get(0) else {
            return Err(invalid_call(
                "native FFI member access requires a foreign resource",
            ));
        };
        self.handles.resolve(reference, ForeignCapability::Read)?;
        let entry = self
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
        match (member, entry) {
            (
                PointerMember::Value,
                PointerEntry::CallerOwned {
                    pointer, metadata, ..
                },
            ) => pointer
                .value(&metadata)
                .map_err(|error| foreign_error_with_source(ForeignErrorKind::InvalidCall, error)),
            (PointerMember::DataType, PointerEntry::CallerOwned { type_name, .. }) => {
                Ok(Value::string(type_name))
            }
            (
                PointerMember::Value,
                PointerEntry::Opaque {
                    pointer,
                    view,
                    metadata,
                    ..
                },
            ) => {
                let view = view.borrow().clone().ok_or_else(|| {
                    invalid_call(
                        "opaque library pointer values require setdatatype before they can be copied",
                    )
                })?;
                runmat_native_ffi::copy_pointer_value(
                    &pointer,
                    &view.pointee,
                    &view.shape,
                    &metadata,
                )
                .map_err(|error| foreign_error_with_source(ForeignErrorKind::InvalidCall, error))
            }
            (PointerMember::DataType, PointerEntry::Opaque { pointer, view, .. }) => {
                Ok(Value::string(
                    view.borrow()
                        .as_ref()
                        .map(|view| view.type_name.clone())
                        .unwrap_or_else(|| format!("{:?}Ptr", pointer.pointee)),
                ))
            }
            (
                PointerMember::Field(member),
                PointerEntry::CallerOwned {
                    pointer, metadata, ..
                },
            ) => {
                let value = pointer.value(&metadata).map_err(|error| {
                    foreign_error_with_source(ForeignErrorKind::InvalidCall, error)
                })?;
                let Value::Struct(structure) = value else {
                    return Err(invalid_call(format!(
                        "native pointer has no member `{member}`"
                    )));
                };
                structure.fields.get(&member).cloned().ok_or_else(|| {
                    invalid_call(format!("native structure has no field `{member}`"))
                })
            }
            (PointerMember::Field(member), _) => Err(invalid_call(format!(
                "native pointer has no member `{member}`"
            ))),
        }
    }

    pub(super) fn set_member(
        &self,
        context: &RuntimeContext,
        arguments: &(impl InvocationArguments + ?Sized),
    ) -> Result<Value, RuntimeError> {
        if arguments.len() != 3 {
            return Err(invalid_call(
                "native FFI set_member expects a foreign resource, member name, and value",
            ));
        }
        let member = PointerMember::parse(string_argument(arguments, 1, "set_member")?);
        let Some(Value::Foreign(reference)) = arguments.get(0) else {
            return Err(invalid_call(
                "native FFI member assignment requires a foreign resource",
            ));
        };
        self.handles.resolve(reference, ForeignCapability::Write)?;
        let entry = self
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
        match entry {
            PointerEntry::CallerOwned { pointer, metadata, .. } => {
                if matches!(&member, PointerMember::Value) {
                    pointer
                        .set_value(arguments.get(2).expect("validated member value"), &metadata)
                        .map_err(|error| foreign_error_with_source(ForeignErrorKind::InvalidCall, error))?;
                    return Ok(Value::Foreign(reference.clone()));
                }
                match member {
                    PointerMember::Value => unreachable!("handled by borrowed pointer assignment"),
                    PointerMember::DataType => Err(invalid_call(
                        "native pointer member `DataType` is read-only",
                    )),
                    PointerMember::Field(member) => {
                        let current = pointer.value(&metadata).map_err(|error| {
                            foreign_error_with_source(ForeignErrorKind::InvalidCall, error)
                        })?;
                        let current = crate::execution::RootedValue::in_context(current, context)?;
                        let Value::Struct(structure) = current.value() else {
                            return Err(invalid_call(format!(
                                "native pointer member `{member}` is unavailable"
                            )));
                        };
                        if !structure.fields.contains_key(&member) {
                            return Err(invalid_call(format!(
                                "native structure has no field `{member}`"
                            )));
                        }
                        let replacement = arguments.get(2).expect("validated member value");
                        let replacement_roots = crate::execution::structure::replacement::lease(replacement, Some(context))?;
                        let updated = crate::execution::construct_struct_from_fields_in_context(
                            context,
                            structure.fields.iter().map(|(name, value)| (name, if name == member.as_str() { replacement } else { value })),
                        )?;
                        pointer.set_value(updated.value(), &metadata).map_err(|error| {
                            foreign_error_with_source(ForeignErrorKind::InvalidCall, error)
                        })?;
                        drop(replacement_roots);
                        Ok(Value::Foreign(reference.clone()))
                    }
                }
            }
            PointerEntry::Opaque { .. } => Err(invalid_call(
                "opaque library pointer values cannot be assigned without an explicit copy contract",
            )),
        }
    }

    pub(super) fn invoke_member(
        &self,
        arguments: &(impl InvocationArguments + ?Sized),
    ) -> Result<Value, RuntimeError> {
        if arguments.len() < 2 {
            return Err(invalid_call(
                "native FFI invoke_member expects a foreign resource and method name",
            ));
        }
        let Some(Value::Foreign(reference)) = arguments.get(0) else {
            return Err(invalid_call(
                "native FFI method invocation requires a foreign resource",
            ));
        };
        let method = string_argument(arguments, 1, "invoke_member")?;
        self.handles.resolve(reference, ForeignCapability::Write)?;
        let role = PointerMethod::parse(&method)
            .ok_or_else(|| invalid_call(format!("native pointer has no method `{method}`")))?;
        match role {
            PointerMethod::SetDataType => self.set_pointer_datatype(
                reference,
                &ArgumentWindow {
                    source: arguments,
                    offset: 2,
                },
            ),
            PointerMethod::IsNull => self.pointer_is_null(
                reference,
                &ArgumentWindow {
                    source: arguments,
                    offset: 2,
                },
            ),
            PointerMethod::Reshape => self.reshape_pointer(
                reference,
                &ArgumentWindow {
                    source: arguments,
                    offset: 2,
                },
            ),
        }
    }

    pub(super) fn set_pointer_datatype(
        &self,
        reference: &runmat_value::ForeignRef,
        arguments: &(impl InvocationArguments + ?Sized),
    ) -> Result<Value, RuntimeError> {
        if arguments.len() < 2 {
            return Err(invalid_call(
                "setdatatype expects a pointer type and one or more dimensions",
            ));
        }
        let type_name = string_argument(arguments, 0, "setdatatype")?;
        let shape = (1..arguments.len())
            .map(|index| pointer_dimension(arguments.get(index).expect("validated dimension")))
            .collect::<Result<Vec<_>, _>>()?;
        shape.iter().try_fold(1usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or_else(|| invalid_call("setdatatype dimensions overflow usize"))
        })?;

        let mut state = self.state.borrow_mut();
        let entry = state.pointers.get_mut(&reference.handle).ok_or_else(|| {
            foreign_error(
                ForeignErrorKind::StaleHandle,
                "native pointer resource has been released",
            )
        })?;
        let PointerEntry::Opaque { view, metadata, .. } = entry else {
            return Err(invalid_call(
                "setdatatype applies to opaque pointers returned by a native library",
            ));
        };
        let pointee = pointer_view_type(&type_name, metadata)?;
        *view.borrow_mut() = Some(OpaquePointerView {
            type_name,
            pointee,
            shape,
        });
        Ok(Value::Foreign(reference.clone()))
    }

    pub(super) fn pointer_is_null(
        &self,
        reference: &runmat_value::ForeignRef,
        arguments: &(impl InvocationArguments + ?Sized),
    ) -> Result<Value, RuntimeError> {
        if !arguments.is_empty() {
            return Err(invalid_call("isNull does not accept arguments"));
        }
        let state = self.state.borrow();
        let entry = state.pointers.get(&reference.handle).ok_or_else(|| {
            foreign_error(
                ForeignErrorKind::StaleHandle,
                "native pointer resource has been released",
            )
        })?;
        Ok(Value::Bool(match entry {
            PointerEntry::CallerOwned { pointer, .. } => pointer.is_null(),
            PointerEntry::Opaque { pointer, .. } => pointer.is_null(),
        }))
    }

    pub(super) fn reshape_pointer(
        &self,
        reference: &runmat_value::ForeignRef,
        arguments: &(impl InvocationArguments + ?Sized),
    ) -> Result<Value, RuntimeError> {
        if arguments.is_empty() {
            return Err(invalid_call("reshape expects one or more dimensions"));
        }
        let shape = (0..arguments.len())
            .map(|index| pointer_dimension(arguments.get(index).expect("validated dimension")))
            .collect::<Result<Vec<_>, _>>()?;
        let requested = shape.iter().try_fold(1usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or_else(|| invalid_call("reshape dimensions overflow usize"))
        })?;
        let mut state = self.state.borrow_mut();
        let entry = state.pointers.get_mut(&reference.handle).ok_or_else(|| {
            foreign_error(
                ForeignErrorKind::StaleHandle,
                "native pointer resource has been released",
            )
        })?;
        let PointerEntry::Opaque { view, .. } = entry else {
            return Err(invalid_call(
                "reshape on caller-owned pointers is not supported because their allocation shape is fixed",
            ));
        };
        let mut view = view.borrow_mut();
        let current = view.as_mut().ok_or_else(|| {
            invalid_call("reshape requires setdatatype to establish the pointer extent first")
        })?;
        let existing = current.shape.iter().product::<usize>();
        if requested != existing {
            return Err(invalid_call(format!(
                "reshape cannot change a pointer view from {existing} to {requested} elements"
            )));
        }
        current.shape = shape;
        Ok(Value::Foreign(reference.clone()))
    }
}

pub(super) fn pointer_metadata() -> NativeLibraryMetadata {
    NativeLibraryMetadata {
        schema_version: NATIVE_FFI_METADATA_SCHEMA_VERSION,
        target_triple: target_lexicon::HOST.to_string(),
        source_digest: "session-owned-pointer".into(),
        libraries: Vec::new(),
        structures: Vec::new(),
        enumerations: Vec::new(),
        aliases: Vec::new(),
    }
}

pub(super) fn declared_pointer_type(
    name: &str,
    metadata: &NativeLibraryMetadata,
) -> Option<NativeType> {
    metadata
        .aliases
        .iter()
        .find(|alias| alias.name == name)
        .map(|alias| alias.target.clone())
        .or_else(|| {
            metadata
                .structures
                .iter()
                .any(|definition| definition.name == name)
                .then(|| NativeType::Structure { name: name.into() })
        })
        .or_else(|| {
            metadata
                .enumerations
                .iter()
                .find(|definition| definition.name == name)
                .map(|definition| NativeType::Enumeration {
                    name: name.into(),
                    storage: definition.storage,
                })
        })
}

pub(super) fn pointer_view_type(
    type_name: &str,
    metadata: &NativeLibraryMetadata,
) -> Result<NativeType, RuntimeError> {
    if let Ok(pointee) = legacy_pointer_type(type_name) {
        return Ok(pointee);
    }
    let declared = type_name
        .strip_suffix("Ptr")
        .filter(|name| !name.is_empty())
        .unwrap_or(type_name);
    declared_pointer_type(declared, metadata).ok_or_else(|| {
        invalid_call(format!(
            "setdatatype type `{type_name}` is not a numeric type or a declaration from the pointer's library"
        ))
    })
}

pub(super) fn pointer_dimension(value: &Value) -> Result<usize, RuntimeError> {
    let dimension = match value {
        Value::Int(value) => value.try_to_usize(),
        Value::Num(value) if value.is_finite() && *value >= 1.0 && value.fract() == 0.0 => {
            usize::try_from(*value as u128).ok()
        }
        Value::Tensor(tensor) if tensor.len() == 1 => {
            tensor.numeric_value_at(0).and_then(|value| {
                value.into_int_value().map_or_else(
                    || {
                        let value = value.materialize_f64();
                        (value.is_finite() && value >= 1.0 && value.fract() == 0.0)
                            .then(|| usize::try_from(value as u128).ok())
                            .flatten()
                    },
                    |value| value.try_to_usize(),
                )
            })
        }
        _ => None,
    };
    dimension
        .filter(|dimension| *dimension > 0)
        .ok_or_else(|| invalid_call("pointer dimensions must be positive integer scalars"))
}

pub(super) fn default_structure_value(
    name: &str,
    metadata: &NativeLibraryMetadata,
) -> Result<runmat_value::StructValue, RuntimeError> {
    let definition = metadata
        .structures
        .iter()
        .find(|definition| definition.name == name)
        .ok_or_else(|| invalid_call(format!("missing native structure definition `{name}`")))?;
    let mut value = runmat_value::UnboundedStructValueBuilder::new();
    for field in &definition.fields {
        let field_value = match &field.ty {
            NativeType::Scalar { .. } | NativeType::Enumeration { .. } => Value::Num(0.0),
            NativeType::Structure { name } => {
                Value::Struct(default_structure_value(name, metadata)?)
            }
            unsupported => {
                return Err(invalid_call(format!(
                    "native structure field `{}` has unsupported default type {unsupported:?}",
                    field.name
                )))
            }
        };
        value.insert(field.name.clone(), field_value);
    }
    Ok(value.finish())
}

pub(super) fn pointer_type_identity(nominal_type: String) -> ForeignTypeIdentity {
    ForeignTypeIdentity {
        family: NATIVE_FFI_ADAPTER_ID.into(),
        name: nominal_type,
        version: super::pointer_roles::POINTER_TYPE_VERSION,
    }
}

pub(super) fn foreign_ownership(ownership: PointerOwnership) -> ForeignOwnership {
    match ownership {
        PointerOwnership::Borrowed => ForeignOwnership::Borrowed,
        PointerOwnership::CallerOwned | PointerOwnership::LibraryOwned => ForeignOwnership::Owned,
        PointerOwnership::Shared => ForeignOwnership::Shared,
    }
}
