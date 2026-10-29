use super::*;

impl NativeFfiAdapter {
    pub(super) fn pointer(
        &self,
        arguments: &(impl InvocationArguments + ?Sized),
    ) -> Result<Value, RuntimeError> {
        if arguments.len() > 2 {
            return Err(invalid_call(
                "native FFI pointer expects an optional pointer type and initial value",
            ));
        }
        let type_name = if arguments.is_empty() {
            "voidPtr".to_string()
        } else {
            string_argument(arguments, 0, "pointer")?
        };
        let resolved = self.resolve_pointer_type(&type_name)?;
        let pointer = match arguments.get(1) {
            Some(value) => {
                NativePointerResource::new(resolved.pointee.clone(), value, &resolved.metadata)
                    .map_err(|error| {
                        foreign_error_with_source(ForeignErrorKind::InvalidCall, error)
                    })?
            }
            None => NativePointerResource::null(resolved.pointee),
        };
        self.register_caller_owned_pointer(
            pointer,
            resolved.metadata,
            type_name,
            super::pointer_roles::POINTER_TYPE_NAME.into(),
            resolved.library,
        )
    }

    pub(super) fn resolve_pointer_type(
        &self,
        type_name: &str,
    ) -> Result<ResolvedPointerType, RuntimeError> {
        if let Ok(pointee) = legacy_pointer_type(type_name) {
            return Ok(ResolvedPointerType {
                pointee,
                metadata: Rc::new(pointer_metadata()),
                library: None,
            });
        }
        let declared = type_name
            .strip_suffix("Ptr")
            .filter(|name| !name.is_empty())
            .unwrap_or(type_name);
        let mut matches = self
            .state
            .borrow()
            .libraries
            .values()
            .filter_map(|entry| {
                declared_pointer_type(declared, &entry.metadata).map(|pointee| {
                    (
                        pointee,
                        Rc::clone(&entry.metadata),
                        Rc::clone(&entry.library),
                    )
                })
            })
            .collect::<Vec<_>>();
        let Some((pointee, metadata, library)) = matches.pop() else {
            return Err(invalid_call(format!(
                "unsupported libpointer type `{type_name}`; use a numeric pointer type or a type declared by a loaded library"
            )));
        };
        if matches
            .iter()
            .any(|(candidate, _, _)| candidate != &pointee)
        {
            return Err(invalid_call(format!(
                "libpointer type `{type_name}` is ambiguous across loaded libraries"
            )));
        }
        Ok(ResolvedPointerType {
            pointee,
            metadata,
            library: Some(library),
        })
    }

    pub(super) fn register_caller_owned_pointer(
        &self,
        pointer: NativePointerResource,
        metadata: Rc<NativeLibraryMetadata>,
        type_name: String,
        nominal_type: String,
        library: Option<Rc<runmat_native_ffi::LoadedLibrary>>,
    ) -> Result<Value, RuntimeError> {
        let reference = self.handles.register_resource(
            &self.host_identity,
            ForeignResourceMetadata {
                type_identity: pointer_type_identity(nominal_type),
                ownership: ForeignOwnership::Owned,
                affinity: ForeignAffinity::OriginThread,
                lifetime: ForeignLifetime::Session,
            },
        )?;
        self.state.borrow_mut().pointers.insert(
            reference.handle,
            PointerEntry::CallerOwned {
                pointer: Rc::new(pointer),
                metadata,
                type_name,
                library,
            },
        );
        Ok(Value::Foreign(reference))
    }

    pub(super) fn structure(
        &self,
        arguments: &(impl InvocationArguments + ?Sized),
    ) -> Result<Value, RuntimeError> {
        if !(1..=2).contains(&arguments.len()) {
            return Err(invalid_call(
                "native FFI structure expects a declared structure type and optional value",
            ));
        }
        let requested = string_argument(arguments, 0, "structure")?;
        let (entry, structure_name) = self.structure_library(&requested)?;
        let initial_value = match arguments.get(1) {
            Some(Value::Struct(value)) => Value::Struct(value.clone()),
            Some(_) => {
                return Err(invalid_call(
                    "native structure initialization requires a scalar structure value",
                ))
            }
            None => Value::Struct(default_structure_value(
                &structure_name,
                entry.metadata.as_ref(),
            )?),
        };
        let pointer = NativePointerResource::new(
            NativeType::Structure {
                name: structure_name.clone(),
            },
            &initial_value,
            &entry.metadata,
        )
        .map_err(|error| foreign_error_with_source(ForeignErrorKind::InvalidCall, error))?;
        self.register_caller_owned_pointer(
            pointer,
            Rc::clone(&entry.metadata),
            structure_name.clone(),
            format!("lib.{structure_name}"),
            Some(Rc::clone(&entry.library)),
        )
    }

    pub(super) fn structure_library(
        &self,
        requested: &str,
    ) -> Result<(LibraryEntry, String), RuntimeError> {
        let (requested_alias, structure_name) = requested
            .split_once('.')
            .map(|(alias, name)| (Some(alias), name))
            .unwrap_or((None, requested));
        if structure_name.trim().is_empty() {
            return Err(invalid_call("native structure type must not be empty"));
        }
        let state = self.state.borrow();
        let mut matches = state
            .libraries
            .iter()
            .filter_map(|(alias, entry)| {
                if requested_alias.is_some_and(|requested| requested != alias.as_str()) {
                    return None;
                }
                entry
                    .metadata
                    .structures
                    .iter()
                    .find(|definition| definition.name == structure_name)
                    .cloned()
                    .map(|definition| (entry.clone(), definition))
            })
            .collect::<Vec<_>>();
        let Some((entry, definition)) = matches.pop() else {
            return Err(invalid_call(format!(
                "native structure type `{requested}` is not declared by a loaded library"
            )));
        };
        if matches
            .iter()
            .any(|(_, candidate)| candidate != &definition)
        {
            return Err(invalid_call(format!(
                "native structure type `{structure_name}` is ambiguous; qualify it with the library alias"
            )));
        }
        Ok((entry, structure_name.to_string()))
    }
    pub(super) fn register_opaque_pointer(
        &self,
        pointer: runmat_native_ffi::NativePointer,
        library: &LibraryEntry,
    ) -> Result<Value, RuntimeError> {
        let ownership = foreign_ownership(pointer.ownership);
        let reference = self.handles.register_resource(
            &self.host_identity,
            ForeignResourceMetadata {
                type_identity: pointer_type_identity(
                    super::pointer_roles::POINTER_TYPE_NAME.into(),
                ),
                ownership,
                affinity: ForeignAffinity::OriginThread,
                lifetime: ForeignLifetime::Session,
            },
        )?;
        self.state.borrow_mut().pointers.insert(
            reference.handle,
            PointerEntry::Opaque {
                pointer: Rc::new(pointer),
                view: Rc::new(RefCell::new(None)),
                library: Rc::clone(&library.library),
                metadata: Rc::clone(&library.metadata),
            },
        );
        Ok(Value::Foreign(reference))
    }
}
