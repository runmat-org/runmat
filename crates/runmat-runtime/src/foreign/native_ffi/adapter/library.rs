use super::*;

#[derive(Debug, thiserror::Error)]
#[error("could not read native library {path}: {source}")]
pub(in crate::foreign::native_ffi) struct PreparedLibraryReadError {
    pub(in crate::foreign::native_ffi) path: PathBuf,
    #[source]
    pub(in crate::foreign::native_ffi) source: std::io::Error,
}

impl NativeFfiAdapter {
    pub(super) fn load(
        &self,
        arguments: &(impl InvocationArguments + ?Sized),
    ) -> Result<Value, RuntimeError> {
        self.load_legacy(arguments)
            .map(|report| Value::string(report.alias))
    }

    pub(super) fn load_report(
        &self,
        arguments: &(impl InvocationArguments + ?Sized),
    ) -> Result<runmat_value::ValueSequence, RuntimeError> {
        let report = self.load_legacy(arguments)?;
        let count = report.notfound.len();
        let notfound = CellArray::new(
            report.notfound.into_iter().map(Value::string).collect(),
            1,
            count,
        )
        .map_err(invalid_call)?;
        runmat_value::ValueSequence::comma_separated(vec![
            Value::Cell(notfound),
            Value::string(report.warnings),
        ])
        .map_err(crate::sequence::sequence_error_to_runtime)
    }

    pub(super) fn load_legacy(
        &self,
        arguments: &(impl InvocationArguments + ?Sized),
    ) -> Result<LegacyLoadReport, RuntimeError> {
        if arguments.len() < 2 {
            return Err(invalid_call(
                "native FFI load expects library path, header path, and an optional alias",
            ));
        }
        let library_path = string_argument(arguments, 0, "load")?;
        let header_path = string_argument(arguments, 1, "load")?;
        let alias = optional_string_argument(arguments, 2, "load")?
            .map(Ok)
            .unwrap_or_else(|| default_alias(&library_path))?;
        let (include_directories, additional_headers) = parse_load_trailer(arguments)?;
        self.validate_available_alias(&alias)?;
        let preparation = HeaderPreparation {
            header: PathBuf::from(header_path),
            library_name: alias.clone(),
            library_path: library_path.clone(),
            target_triple: target_lexicon::HOST.to_string(),
            clang: self.compiler_frontend.clone(),
            include_directories,
            definitions: Vec::new(),
        };
        let preparation =
            prepare_header_with_declarations_report(&preparation, &additional_headers)
                .map_err(|error| foreign_error_with_source(ForeignErrorKind::LoadFailed, error))?;
        let identity = artifact_identity(&preparation.metadata)
            .map_err(|error| foreign_error_with_source(ForeignErrorKind::LoadFailed, error))?;
        let notfound =
            self.insert_library(&alias, &library_path, preparation.metadata, identity, true)?;
        Ok(LegacyLoadReport {
            alias,
            notfound,
            warnings: preparation.warnings,
        })
    }

    pub(super) fn load_prepared(
        &self,
        arguments: &(impl InvocationArguments + ?Sized),
    ) -> Result<Value, RuntimeError> {
        if !(2..=3).contains(&arguments.len()) {
            return Err(invalid_call(
                "prepared native FFI load expects a library path, manifest path, and optional alias",
            ));
        }
        let library_path = PathBuf::from(string_argument(arguments, 0, "load_prepared")?);
        let manifest_path = PathBuf::from(string_argument(arguments, 1, "load_prepared")?);
        let manifest = NativeInterfaceArtifactManifest::read(&manifest_path)
            .map_err(|error| foreign_error_with_source(ForeignErrorKind::LoadFailed, error))?;
        let alias = optional_string_argument(arguments, 2, "load_prepared")?
            .unwrap_or_else(|| manifest.interface_name.clone());
        self.load_prepared_manifest(&alias, &library_path, manifest)
    }

    pub(super) fn load_prepared_manifest(
        &self,
        alias: &str,
        library_path: &std::path::Path,
        manifest: NativeInterfaceArtifactManifest,
    ) -> Result<Value, RuntimeError> {
        self.validate_available_alias(alias)?;
        let library_bytes = std::fs::read(library_path).map_err(|source| {
            foreign_error_with_source(
                ForeignErrorKind::LoadFailed,
                PreparedLibraryReadError {
                    path: library_path.to_path_buf(),
                    source,
                },
            )
        })?;
        manifest
            .validate_current_library(&library_bytes)
            .map_err(|error| foreign_error_with_source(ForeignErrorKind::LoadFailed, error))?;
        let metadata = manifest
            .materialized_metadata(library_path)
            .map_err(|error| foreign_error_with_source(ForeignErrorKind::LoadFailed, error))?;
        let identity = manifest.identity.to_string();
        self.insert_library(alias, library_path, metadata, identity, false)?;
        Ok(Value::String(alias.into()))
    }

    pub(super) fn validate_available_alias(&self, alias: &str) -> Result<(), RuntimeError> {
        if alias.trim().is_empty() {
            return Err(invalid_call("native library alias must not be empty"));
        }
        if self.state.borrow().libraries.contains_key(alias)
            || self
                .project_catalog
                .borrow()
                .loads
                .iter()
                .any(|load| load.alias == alias)
        {
            return Err(invalid_call(format!(
                "native library alias `{alias}` is already loaded"
            )));
        }
        Ok(())
    }

    pub(super) fn insert_library(
        &self,
        alias: &str,
        library_path: impl AsRef<std::path::Path>,
        metadata: NativeLibraryMetadata,
        artifact_identity: String,
        allow_missing_symbols: bool,
    ) -> Result<Vec<String>, RuntimeError> {
        let (entry, missing) = self.prepare_library_entry(
            library_path.as_ref(),
            metadata,
            artifact_identity,
            allow_missing_symbols,
            false,
        )?;
        self.state
            .borrow_mut()
            .libraries
            .insert(alias.into(), entry);
        Ok(missing)
    }

    pub(super) fn prepare_library_entry(
        &self,
        library_path: &std::path::Path,
        metadata: NativeLibraryMetadata,
        artifact_identity: String,
        allow_missing_symbols: bool,
        project_owned: bool,
    ) -> Result<(LibraryEntry, Vec<String>), RuntimeError> {
        let mut library = runmat_native_ffi::LoadedLibrary::open(library_path)
            .map_err(|error| foreign_error_with_source(ForeignErrorKind::LoadFailed, error))?;
        if let Some(lease) = self.private_byte_lease.borrow().as_ref() {
            if library_path.starts_with(lease.path()) {
                library
                    .attach_private_byte_lease(lease.clone())
                    .map_err(|error| {
                        foreign_error_with_source(ForeignErrorKind::LoadFailed, error)
                    })?;
            }
        }
        let mut metadata = metadata;
        let mut missing = Vec::new();
        for declared_library in &mut metadata.libraries {
            let mut available = Vec::with_capacity(declared_library.symbols.len());
            for symbol in std::mem::take(&mut declared_library.symbols) {
                if library.symbol(&symbol.exported_name).is_ok() {
                    available.push(symbol);
                } else {
                    missing.push(symbol.name.clone());
                    if !allow_missing_symbols {
                        continue;
                    }
                }
            }
            declared_library.symbols = available;
        }
        if !allow_missing_symbols && !missing.is_empty() {
            return Err(foreign_error(
                ForeignErrorKind::LoadFailed,
                format!(
                    "prepared native interface is missing declared symbols: {}",
                    missing.join(", ")
                ),
            ));
        }
        Ok((
            LibraryEntry {
                library: Rc::new(library),
                metadata: Rc::new(metadata),
                artifact_identity,
                project_owned,
            },
            missing,
        ))
    }

    pub(super) fn unload(
        &self,
        arguments: &(impl InvocationArguments + ?Sized),
    ) -> Result<Value, RuntimeError> {
        if arguments.len() != 1 {
            return Err(invalid_call("native FFI unload expects one library alias"));
        }
        let alias = string_argument(arguments, 0, "unload")?;
        let removed = self.state.borrow_mut().libraries.remove(&alias).is_some();
        if !removed {
            return Err(invalid_call(format!(
                "native library alias `{alias}` is not loaded"
            )));
        }
        Ok(Value::Bool(true))
    }

    pub(super) fn is_loaded(
        &self,
        arguments: &(impl InvocationArguments + ?Sized),
    ) -> Result<Value, RuntimeError> {
        if arguments.len() != 1 {
            return Err(invalid_call(
                "native FFI is_loaded expects one library alias",
            ));
        }
        let alias = string_argument(arguments, 0, "is_loaded")?;
        Ok(Value::Bool(
            self.state.borrow().libraries.contains_key(&alias),
        ))
    }

    pub(super) fn functions(
        &self,
        arguments: &(impl InvocationArguments + ?Sized),
    ) -> Result<Value, RuntimeError> {
        if !(1..=2).contains(&arguments.len()) {
            return Err(invalid_call(
                "native FFI functions expects a library alias and optional -full flag",
            ));
        }
        let alias = string_argument(arguments, 0, "functions")?;
        let full = match optional_string_argument(arguments, 1, "functions")? {
            None => false,
            Some(option) if option.eq_ignore_ascii_case("-full") => true,
            Some(option) => {
                return Err(invalid_call(format!(
                    "unsupported libfunctions option `{option}`; expected -full"
                )))
            }
        };
        let entry = self.library(&alias)?;
        let values = entry.metadata.libraries[0]
            .symbols
            .iter()
            .map(|symbol| {
                Value::string(if full {
                    legacy_signature(symbol)
                } else {
                    symbol.name.clone()
                })
            })
            .collect::<Vec<_>>();
        CellArray::new(values.clone(), 1, values.len())
            .map(Value::Cell)
            .map_err(invalid_call)
    }
    pub(super) fn library(&self, alias: &str) -> Result<LibraryEntry, RuntimeError> {
        self.state
            .borrow()
            .libraries
            .get(alias)
            .cloned()
            .ok_or_else(|| invalid_call(format!("native library alias `{alias}` is not loaded")))
    }

    pub(super) fn build_interface(
        &self,
        arguments: &(impl InvocationArguments + ?Sized),
    ) -> Result<Value, RuntimeError> {
        if arguments.is_empty() || arguments.len().is_multiple_of(2) {
            return Err(invalid_call(
                "clibgen.buildInterface expects a header followed by name-value pairs",
            ));
        }
        let header = string_argument(arguments, 0, "build_interface")?;
        let mut library = None;
        let mut alias = None;
        let mut includes = Vec::new();
        for index in (1..arguments.len()).step_by(2) {
            let name = String::try_from(arguments.get(index).expect("validated option pair"))
                .map_err(|_| invalid_call("buildInterface option names must be text"))?;
            match runmat_builtins::NativeBuildOption::parse(&name) {
                Some(runmat_builtins::NativeBuildOption::Libraries) => {
                    library = Some(String::try_from(arguments.get(index + 1).expect("validated option pair")).map_err(|_| {
                        invalid_call("buildInterface Libraries must name one shared library")
                    })?);
                }
                Some(runmat_builtins::NativeBuildOption::InterfaceName) => {
                    alias = Some(String::try_from(arguments.get(index + 1).expect("validated option pair")).map_err(|_| {
                        invalid_call("buildInterface InterfaceName must be text")
                    })?);
                }
                Some(runmat_builtins::NativeBuildOption::IncludePath) => {
                    includes.push(PathBuf::from(String::try_from(arguments.get(index + 1).expect("validated option pair")).map_err(|_| {
                        invalid_call("buildInterface include path must be text")
                    })?));
                }
                None => {
                    return Err(invalid_call(format!(
                        "buildInterface option `{name}` is not supported by the C ABI interface builder"
                    )))
                }
            }
        }
        let library = library.ok_or_else(|| {
            invalid_call("buildInterface requires the Libraries option for a native interface")
        })?;
        let alias = alias.map(Ok).unwrap_or_else(|| default_alias(&header))?;
        self.validate_available_alias(&alias)?;
        let library_path = PathBuf::from(&library);
        let prepared = prepare_native_interface(&NativeInterfacePreparation {
            interface_name: alias.clone(),
            library_name: alias.clone(),
            library_path: library_path.clone(),
            primary_header: PathBuf::from(header),
            additional_headers: Vec::new(),
            include_directories: includes,
            definitions: Vec::new(),
            compiler_frontend: self.compiler_frontend.clone(),
        })
        .map_err(|error| foreign_error_with_source(ForeignErrorKind::LoadFailed, error))?;
        let manifest = prepared.manifest;
        let manifest_path = NativeInterfaceArtifactManifest::path_for_library(&library_path);
        manifest
            .publish(&manifest_path)
            .map_err(|error| foreign_error_with_source(ForeignErrorKind::LoadFailed, error))?;
        self.load_prepared_manifest(&alias, &library_path, manifest)
    }
}
/// The legacy adapter also accepts positional include directories. A trailer
/// is treated as named options only when every name is an admitted preparation
/// option; unknown or odd trailers retain the positional interpretation.
fn parse_load_trailer(
    arguments: &(impl InvocationArguments + ?Sized),
) -> Result<(Vec<PathBuf>, Vec<PathBuf>), RuntimeError> {
    // The adapter admits a library and header without an explicit alias.
    let count = arguments.len().saturating_sub(3);
    let named = (count != 0 && count.is_multiple_of(2))
        .then(|| {
            (3..arguments.len())
                .step_by(2)
                .map(|index| {
                    let name = String::try_from(arguments.get(index)?).ok()?;
                    let kind = NativeLoadOption::parse(&name)?;
                    if matches!(
                        kind,
                        NativeLoadOption::IncludePath | NativeLoadOption::AddHeader
                    ) {
                        Some((kind, name, arguments.get(index + 1)?))
                    } else {
                        None
                    }
                })
                .collect::<Option<Vec<_>>>()
        })
        .flatten();
    if let Some(named) = named {
        let mut includes = Vec::new();
        let mut headers = Vec::new();
        for (kind, name, source) in named {
            let path = PathBuf::from(String::try_from(source).map_err(|_| {
                invalid_call(format!("native FFI load option `{name}` must be text"))
            })?);
            match kind {
                NativeLoadOption::IncludePath => includes.push(path),
                NativeLoadOption::AddHeader => headers.push(path),
                NativeLoadOption::Alias => {
                    return Err(invalid_call("native FFI load Alias is positional"))
                }
            }
        }
        Ok((includes, headers))
    } else {
        let includes = (3..arguments.len())
            .map(|index| string_argument(arguments, index, "load").map(PathBuf::from))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((includes, Vec::new()))
    }
}
pub(super) fn legacy_signature(symbol: &SymbolPrototype) -> String {
    let outputs = std::iter::once((&symbol.return_type, None))
        .filter(|(ty, _)| !matches!(ty, NativeType::Void))
        .chain(symbol.parameters.iter().filter_map(|parameter| {
            matches!(
                parameter.direction,
                ParameterDirection::Output | ParameterDirection::InputOutput
            )
            .then_some((&parameter.ty, Some(parameter.name.as_str())))
        }))
        .map(|(ty, name)| match name {
            Some(name) => format!("{} {name}", legacy_type_name(ty)),
            None => legacy_type_name(ty),
        })
        .collect::<Vec<_>>();
    let inputs = symbol
        .parameters
        .iter()
        .map(|parameter| legacy_type_name(&parameter.ty))
        .collect::<Vec<_>>()
        .join(", ");
    match outputs.as_slice() {
        [] => format!("{}({inputs})", symbol.name),
        [output] => format!("{output} {}({inputs})", symbol.name),
        outputs => format!("[{}] {}({inputs})", outputs.join(", "), symbol.name),
    }
}

pub(super) fn legacy_type_name(ty: &NativeType) -> String {
    match ty {
        NativeType::Void => "void".into(),
        NativeType::Scalar { scalar } => match scalar {
            NativeScalar::Bool => "logical",
            NativeScalar::Char | NativeScalar::SignedChar => "int8",
            NativeScalar::UnsignedChar => "uint8",
            NativeScalar::Short => "int16",
            NativeScalar::UnsignedShort => "uint16",
            NativeScalar::Int => "int32",
            NativeScalar::UnsignedInt => "uint32",
            NativeScalar::Long | NativeScalar::LongLong => "int64",
            NativeScalar::UnsignedLong | NativeScalar::UnsignedLongLong => "uint64",
            NativeScalar::I8 => "int8",
            NativeScalar::U8 => "uint8",
            NativeScalar::I16 => "int16",
            NativeScalar::U16 => "uint16",
            NativeScalar::I32 => "int32",
            NativeScalar::U32 => "uint32",
            NativeScalar::I64 | NativeScalar::Isize => "int64",
            NativeScalar::U64 | NativeScalar::Usize => "uint64",
            NativeScalar::F32 => "single",
            NativeScalar::F64 => "double",
        }
        .into(),
        NativeType::Pointer { pointee, .. } => format!("{}Ptr", legacy_type_name(pointee)),
        NativeType::Array { element, .. } => format!("{}Ptr", legacy_type_name(element)),
        NativeType::Structure { name } | NativeType::Enumeration { name, .. } => name.clone(),
        NativeType::Callback { .. } => "FcnPtr".into(),
    }
}

#[cfg(test)]
mod load_trailer_tests {
    use super::*;

    fn prefix() -> Vec<Value> {
        vec![
            Value::String("library".into()),
            Value::String("header".into()),
            Value::String("alias".into()),
        ]
    }

    #[test]
    fn omitted_alias_has_no_load_trailer() {
        let arguments = vec![Value::from("library"), Value::from("header")];
        let (includes, headers) = parse_load_trailer(&arguments).unwrap();
        assert!(includes.is_empty());
        assert!(headers.is_empty());
    }

    #[test]
    fn named_options_decode_once_and_positional_include_paths_remain_admitted() {
        let mut named = prefix();
        named.extend([
            Value::String("includePATH".into()),
            Value::String("inc".into()),
            Value::String("AddHeader".into()),
            Value::String("extra.h".into()),
        ]);
        let (includes, headers) = parse_load_trailer(&named).unwrap();
        assert_eq!(includes, vec![PathBuf::from("inc")]);
        assert_eq!(headers, vec![PathBuf::from("extra.h")]);

        let mut positional = prefix();
        positional.extend([
            Value::String("IncludePath".into()),
            Value::String("inc".into()),
            Value::String("other".into()),
        ]);
        let (includes, headers) = parse_load_trailer(&positional).unwrap();
        assert_eq!(
            includes,
            vec![
                PathBuf::from("IncludePath"),
                PathBuf::from("inc"),
                PathBuf::from("other")
            ]
        );
        assert!(headers.is_empty());
    }

    #[test]
    fn admitted_named_option_with_invalid_value_keeps_invalid_call() {
        let mut named = prefix();
        named.extend([
            Value::String("AddHeader".into()),
            Value::output_list(Vec::new()),
        ]);
        let error = parse_load_trailer(&named).expect_err("invalid named value");
        assert_eq!(error.identifier(), Some("RunMat:Foreign:InvalidCall"));
    }
}
