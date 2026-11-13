use serde::Serialize;

use super::ExecutableUnit;

#[path = "portable/parallel.rs"]
mod parallel;

pub(super) const SOURCE_MAP_SCHEMA_VERSION: u16 = 2;

#[derive(Serialize)]
struct PortableSourceMap<'a> {
    entries: Vec<PortableSourceEntry<'a>>,
}

#[derive(Serialize)]
struct PortableSourceEntry<'a> {
    source_id: runmat_types::ProgramSourceId,
    owner_identity: &'a str,
    relative_path: &'a str,
    content_revision: &'a runmat_execution::Digest,
    display_name: &'a str,
    text: &'a str,
}

impl ExecutableUnit {
    /// Publish the compiler's typed primary-source role consistently across
    /// interactive, precompiled and deferred graph products.
    pub(crate) fn portable_primary_source_envelope(
        &self,
        current: &std::collections::HashSet<runmat_hir::FunctionId>,
        primary: Option<runmat_types::ProgramSourceId>,
        interop: runmat_types::InteropManifest,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        super::SourceProductSelection::classify(self, current, primary)?
            .envelope(self, current, primary, interop)
    }

    /// Static imports are owned by this exact executable slice, not copied
    /// wholesale from an imported compiler product's optional contracts.
    pub(crate) fn portable_primary_source_envelope_with_static_imports(
        &self,
        current: &std::collections::HashSet<runmat_hir::FunctionId>,
        primary: Option<runmat_types::ProgramSourceId>,
        interop: runmat_types::InteropManifest,
        imports: Vec<runmat_execution::executable::ExecutableStaticImport>,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        let mut envelope = self.portable_primary_source_envelope(current, primary, interop)?;
        if !imports.is_empty() {
            envelope.manifest.optional_sections.push(
                runmat_execution::executable::static_import_section(imports)
                    .map_err(|error| error.to_string())?,
            );
            envelope
                .manifest
                .optional_sections
                .sort_by(|left, right| left.name.cmp(&right.name));
        }
        Ok(envelope)
    }

    /// Build the complete physical-path-free executable product for package,
    /// test, browser, remote, native-codegen, and cache consumers.
    pub fn portable_envelope(&self) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        self.portable_envelope_for(None)
    }

    /// Build the complete product with an optional explicitly selected function.
    /// A missing explicit primary-source function is an error. Implicit
    /// selection uses the compiler's typed primary-source product role.
    pub fn portable_envelope_for(
        &self,
        preferred_function: Option<&str>,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        self.portable_envelope_for_with_interop(
            preferred_function,
            runmat_types::InteropManifest::empty(),
        )
    }

    /// Resolve the canonical compiled entrypoint without requiring portable
    /// serialization. Local invocation must not acquire a transport requirement.
    pub fn entrypoint_identity(
        &self,
        preferred_function: Option<&str>,
    ) -> Result<
        (
            runmat_execution::ExecutableEntrypointKind,
            Option<runmat_types::ProgramFunctionId>,
        ),
        String,
    > {
        match super::portable_entrypoint::resolve(self, preferred_function)? {
            super::portable_entrypoint::PortableEntrypoint::Callable { kind, function, .. } => {
                Ok((kind, Some(function)))
            }
            super::portable_entrypoint::PortableEntrypoint::DeclarationOnly(_) => Ok((
                runmat_execution::ExecutableEntrypointKind::DeclarationOnly,
                None,
            )),
        }
    }

    /// Publish a dynamically selected class source through its unique typed
    /// primary constructor body, not through an unqualified filename alias.
    pub(crate) fn portable_envelope_for_constructor_source(
        &self,
        class: &runmat_types::ClassIdentity,
        primary_source: runmat_types::ProgramSourceId,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        self.portable_envelope_with_entrypoint(
            runmat_types::InteropManifest::empty(),
            None,
            |unit| {
                super::portable_entrypoint::resolve_constructor_source(unit, class, primary_source)
            },
        )
    }

    pub(crate) fn portable_envelope_for_declaration_source(
        &self,
        class: &runmat_types::ClassIdentity,
        primary_source: runmat_types::ProgramSourceId,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        self.portable_envelope_with_entrypoint(
            runmat_types::InteropManifest::empty(),
            None,
            |unit| {
                super::portable_entrypoint::resolve_declaration_source(unit, class, primary_source)
            },
        )
    }

    /// Build a complete product with an explicit foreign-interoperability
    /// contract. Package and adapter composition use this entrypoint once
    /// foreign requirements have been resolved; ordinary source compilation
    /// records the deterministic empty contract.
    pub fn portable_envelope_for_with_interop(
        &self,
        preferred_function: Option<&str>,
        interop: runmat_types::InteropManifest,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        self.portable_envelope_with_entrypoint(interop, None, |unit| {
            super::portable_entrypoint::resolve(unit, preferred_function)
        })
    }

    pub(crate) fn portable_envelope_for_current_definition_with_interop(
        &self,
        current: &std::collections::HashSet<runmat_hir::FunctionId>,
        primary_source: runmat_types::ProgramSourceId,
        interop: runmat_types::InteropManifest,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        self.portable_envelope_with_entrypoint(interop, Some(current), |unit| {
            super::portable_entrypoint::resolve_current_definition(unit, current, primary_source)
        })
    }

    /// Admit one compiler-proved closure body without making the surrounding
    /// eval script or its sibling definitions executable in this product.
    /// MIR/layout remain compiler provenance; the bytecode component has no
    /// script instruction stream and the registry contains only this body.
    pub(crate) fn portable_envelope_for_pending_closure(
        &self,
        function: runmat_hir::FunctionId,
        primary_source: runmat_types::ProgramSourceId,
        interop: runmat_types::InteropManifest,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        let included = std::collections::HashSet::from([function]);
        self.portable_envelope_with_entrypoint_mode(interop, Some(&included), false, |unit| {
            super::portable_entrypoint::resolve_current_definition(unit, &included, primary_source)
        })
    }

    pub(crate) fn portable_script_envelope_with_current_definitions(
        &self,
        interop: runmat_types::InteropManifest,
        included: &std::collections::HashSet<runmat_hir::FunctionId>,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        if included
            .iter()
            .any(|function| self.functions().get_current(*function).is_none())
        {
            return Err(
                "selected script current definition is absent from its compiled unit".into(),
            );
        }
        self.portable_envelope_with_entrypoint(interop, Some(included), |unit| {
            super::portable_entrypoint::script_entrypoint(unit)
        })
    }

    pub(crate) fn portable_declaration_envelope_with_interop(
        &self,
        class: &runmat_types::ClassIdentity,
        primary_source: runmat_types::ProgramSourceId,
        interop: runmat_types::InteropManifest,
        included: &std::collections::HashSet<runmat_hir::FunctionId>,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        self.portable_envelope_with_entrypoint(interop, Some(included), |unit| {
            super::portable_entrypoint::resolve_declaration_source(unit, class, primary_source)
        })
    }

    pub(crate) fn portable_constructor_envelope_with_interop(
        &self,
        class: &runmat_types::ClassIdentity,
        primary_source: runmat_types::ProgramSourceId,
        interop: runmat_types::InteropManifest,
        included: &std::collections::HashSet<runmat_hir::FunctionId>,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        self.portable_envelope_with_entrypoint(interop, Some(included), |unit| {
            super::portable_entrypoint::resolve_constructor_source(unit, class, primary_source)
        })
    }

    /// Build the complete product with an exact semantic function selected as its entrypoint.
    pub fn portable_envelope_for_function(
        &self,
        function: runmat_types::ProgramFunctionId,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        let origin = runmat_execution::FunctionBindingOrigin::from_revision(
            &self.revision().program_revision,
            function,
        )
        .map_err(|e| e.to_string())?;
        let function = self
            .local_function_for_origin(origin)?
            .ok_or("executable does not own the selected portable callable")?;
        let name = self
            .functions()
            .get(function)
            .ok_or_else(|| format!("executable does not define function {}", function.0))?
            .display_name
            .clone();
        self.portable_envelope_for_selection(Some((name, function)))
    }

    /// Complete product including the flat original dependency inventory.
    /// `None` selects the script entrypoint.
    pub(super) fn portable_envelope_for_selection(
        &self,
        selected: Option<(String, runmat_hir::FunctionId)>,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        let local = self.portable_local_envelope_for_selection(selected)?;
        runmat_execution::ExecutableUnitEnvelope::with_dependencies(
            local.manifest,
            local.payloads,
            self.portable_dependency_products()?,
        )
        .map_err(|error| error.to_string())
    }

    /// This unit's own product, without its original dependency inventory.
    pub(super) fn portable_local_envelope_for_selection(
        &self,
        selected: Option<(String, runmat_hir::FunctionId)>,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        // A standalone Core product conserves the compiler's inherited
        // projections and carries their original products in its dependency
        // inventory. Selected Graph products use explicit body/import slices.
        let included = self.functions().functions.keys().copied().collect();
        let mut envelope = self.portable_local_envelope_with_entrypoint_mode(
            runmat_types::InteropManifest::empty(),
            Some(&included),
            true,
            |unit| match selected {
                Some((name, function)) => {
                    super::portable_entrypoint::resolve_compiled_function_with_label(
                        unit, function, &name,
                    )
                }
                None => super::portable_entrypoint::script_entrypoint(unit),
            },
        )?;
        // This complete compiler product deliberately includes inherited
        // projections. Its executable sites still belong only to this
        // product's script/current bodies, and keep the compiler's exact
        // selected imports even though body selection was explicit above.
        if !self.original_static_imports().is_empty() {
            envelope.manifest.optional_sections.push(
                runmat_execution::executable::static_import_section(
                    self.original_static_imports().to_vec(),
                )
                .map_err(|error| error.to_string())?,
            );
            envelope
                .manifest
                .optional_sections
                .sort_by(|left, right| left.name.cmp(&right.name));
        }
        Ok(envelope)
    }

    pub(super) fn portable_envelope_with_entrypoint(
        &self,
        interop: runmat_types::InteropManifest,
        included: Option<&std::collections::HashSet<runmat_hir::FunctionId>>,
        resolve: impl FnOnce(&Self) -> Result<super::portable_entrypoint::PortableEntrypoint, String>,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        self.portable_envelope_with_entrypoint_mode(interop, included, true, resolve)
    }

    fn portable_envelope_with_entrypoint_mode(
        &self,
        interop: runmat_types::InteropManifest,
        included: Option<&std::collections::HashSet<runmat_hir::FunctionId>>,
        include_script: bool,
        resolve: impl FnOnce(&Self) -> Result<super::portable_entrypoint::PortableEntrypoint, String>,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        let local = self.portable_local_envelope_with_entrypoint_mode(
            interop,
            included,
            include_script,
            resolve,
        )?;
        runmat_execution::ExecutableUnitEnvelope::with_dependencies(
            local.manifest,
            local.payloads,
            self.portable_dependency_products()?,
        )
        .map_err(|error| error.to_string())
    }

    fn portable_local_envelope_with_entrypoint_mode(
        &self,
        interop: runmat_types::InteropManifest,
        included: Option<&std::collections::HashSet<runmat_hir::FunctionId>>,
        include_script: bool,
        resolve: impl FnOnce(&Self) -> Result<super::portable_entrypoint::PortableEntrypoint, String>,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        // A session compilation may retain earlier bodies for analysis and
        // direct invocation. They belong to their own selected products, not
        // to this primary V3 envelope. Explicit graph callers provide their
        // own body selection and attach their own issued import section.
        let automatic_selection = included
            .is_none()
            .then(|| self.portable_selection())
            .flatten();
        let included = included.or_else(|| automatic_selection.map(|(current, _)| current));
        let selected_scope = if include_script {
            None
        } else {
            let included = included.ok_or("function-only product has no selected roots")?;
            let roots = included.iter().copied().collect();
            let names = runmat_mir::analysis::ReachabilityNames {
                sources: self
                    .source_map()
                    .entries()
                    .iter()
                    .map(|entry| (entry.source_id, entry.relative_path.clone()))
                    .collect(),
                bindings: self.binding_names(),
            };
            let (report, functions) =
                runmat_mir::analysis::selected_body_reachability(self.mir(), &names, &roots);
            if !included.iter().all(|function| functions.contains(function)) {
                return Err("function-only root is absent from selected MIR reachability".into());
            }
            Some((report, functions))
        };
        let mut selected_local_functions = selected_scope
            .as_ref()
            .map(|(_, functions)| functions.clone())
            .unwrap_or_default();
        let script_owners = if include_script {
            self.compiled_script_owner_ids()?
        } else {
            std::collections::BTreeSet::new()
        };
        if include_script {
            for function in &script_owners {
                selected_local_functions.insert(*function);
            }
            if let Some(included) = included {
                for function in included {
                    selected_local_functions.insert(*function);
                }
            }
        }
        let selected_report = if selected_scope.is_none() && included.is_some() {
            let roots = selected_local_functions.iter().copied().collect();
            let names = runmat_mir::analysis::ReachabilityNames {
                sources: self
                    .source_map()
                    .entries()
                    .iter()
                    .map(|entry| (entry.source_id, entry.relative_path.clone()))
                    .collect(),
                bindings: self.binding_names(),
            };
            Some(runmat_mir::analysis::selected_body_reachability(self.mir(), &names, &roots).0)
        } else {
            None
        };
        let selected_functions = selected_local_functions
            .into_iter()
            .filter(|function| script_owners.contains(function) || self.owns_callable(*function))
            .map(|function| self.portable_function_for_local(function))
            .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
        let inferred = match (&selected_scope, &selected_report) {
            (Some((report, _)), _) | (_, Some(report)) => {
                Self::inferred_interop_manifest_for_report(report)
            }
            _ => self.inferred_interop_manifest(),
        };
        let interop = runmat_types::InteropManifest::merge([inferred, interop])
            .map_err(|error| format!("{}: {}", error.path, error.message))?;
        let payloads = self.component_payloads(included, include_script)?;
        let revisions = self.component_revisions()?;
        let components = component_descriptors(&payloads, &revisions)?;
        let entrypoint = resolve(self)?;
        let (
            entrypoint_name,
            entrypoint_compiled,
            entrypoint_function,
            entrypoint_kind,
            declaration_class,
        ) = match entrypoint {
            super::portable_entrypoint::PortableEntrypoint::Callable {
                name,
                compiled_function,
                function,
                kind,
                ..
            } => (name, Some(compiled_function), Some(function), kind, None),
            super::portable_entrypoint::PortableEntrypoint::DeclarationOnly(class) => (
                class.display_name().to_string(),
                None,
                None,
                runmat_execution::ExecutableEntrypointKind::DeclarationOnly,
                Some(class),
            ),
        };
        if automatic_selection.is_some()
            && entrypoint_compiled.is_some_and(|function| {
                !script_owners.contains(&function)
                    && included.is_none_or(|selected| !selected.contains(&function))
            })
        {
            return Err("preferred entrypoint belongs to a retained definition product".into());
        }
        // Cold products retain their original interoperability contract. Select
        // it before deriving capabilities so adapter requirements participate
        // in the same canonical reconstruction as fresh products. Do not copy
        // capability claims from the incoming manifest.
        let interop = self
            .imported_contract
            .as_deref()
            .map_or(interop, |imported| imported.interop.clone());
        let mut capabilities = runmat_types::CapabilitySet(
            self.analysis()
                .functions
                .iter()
                .filter(|function| {
                    included.is_none() || selected_functions.contains(&function.function)
                })
                .flat_map(|function| function.capabilities.0.iter().copied())
                .collect(),
        );
        for adapter in &interop.adapters {
            capabilities
                .0
                .extend(adapter.capabilities.0.iter().copied());
        }
        if !interop.foreign_types.is_empty() || !interop.adapters.is_empty() {
            capabilities
                .0
                .insert(runmat_types::CapabilityRequirement::ForeignRuntime);
        }
        let selected_functions = included.map(|_| &selected_functions);
        let mut parallel = self.analysis().parallel.clone();
        if let Some(functions) = selected_functions {
            parallel
                .parfor_regions
                .retain(|region| functions.contains(&region.id.0.function));
            parallel
                .spmd_regions
                .retain(|region| functions.contains(&region.id.0.function));
            parallel
                .distributed_values
                .retain(|value| functions.contains(&value.id.function));
            parallel
                .collectives
                .retain(|collective| functions.contains(&collective.id.region.0.function));
        }
        parallel::validate(self, &parallel, selected_functions)?;
        let mut optional_sections = Vec::new();
        if let Some((_, imports)) = automatic_selection {
            if !imports.is_empty() {
                optional_sections.push(
                    runmat_execution::executable::static_import_section(imports.to_vec())
                        .map_err(|error| error.to_string())?,
                );
            }
        }
        // A restored product also keeps its original parallel and
        // optional-section contracts rather than re-deriving them.
        let (parallel, mut optional_sections) = match self.imported_contract.as_deref() {
            Some(imported) => (
                imported.parallel.clone(),
                imported
                    .optional_sections
                    .iter()
                    .filter(|section| {
                        section.name != runmat_execution::executable::STATIC_IMPORT_SECTION
                    })
                    .cloned()
                    .chain(optional_sections)
                    .collect(),
            ),
            None => (parallel, optional_sections),
        };
        optional_sections.sort_by(|left, right| left.name.cmp(&right.name));
        let manifest = runmat_execution::ExecutableUnitManifest {
            dependencies: Vec::new(),
            schema_version: runmat_execution::EXECUTABLE_UNIT_SCHEMA_VERSION,
            identity: runmat_execution::ExecutableIdentity {
                program: self.revision().program_revision.clone(),
                root_package: self.source().owner_identity.clone(),
                entrypoint: entrypoint_name,
                entrypoint_function,
                entrypoint_kind,
                declaration_class,
                source_digest: self
                    .revision()
                    .source_digest
                    .parse()
                    .map_err(|error: runmat_execution::ContractError| error.to_string())?,
            },
            revisions,
            components,
            capabilities,
            regions: self
                .analysis()
                .regions
                .iter()
                .filter(|region| {
                    selected_functions
                        .is_none_or(|functions| functions.contains(&region.contract.id.function))
                })
                .map(|region| region.contract.clone())
                .collect(),
            interop,
            parallel,
            optional_sections,
        };
        runmat_execution::ExecutableUnitEnvelope::new(manifest, payloads)
            .map_err(|error| error.to_string())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn native_product_digest(&self) -> Result<runmat_execution::Digest, String> {
        let revisions = self.component_revisions()?;
        let selection = self.portable_selection();
        let components = component_descriptors(
            &self.component_payloads(selection.map(|(current, _)| current), true)?,
            &revisions,
        )?;
        // VM layout, registry and bytecode also own the retained interpreter
        // capture, resume locations and coverage instrumentation.
        Ok(runmat_execution::Digest::sha256(canonical_json(&(
            "runmat-native-compilation-v1",
            revisions,
            components,
            selection.map(|(_, imports)| imports),
            self.binding_names(),
            self.inferred_interop_manifest(),
            &self.source().owner_identity,
            &self.source().relative_path,
        ))?))
    }

    /// Adapter families proven reachable from this immutable compiled unit.
    /// Artifact identities are attached by a frozen project issuer separately.
    pub fn inferred_interop_manifest(&self) -> runmat_types::InteropManifest {
        Self::inferred_interop_manifest_for_report(&self.reachability_report())
    }

    pub(crate) fn inferred_interop_manifest_for_report(
        report: &runmat_mir::analysis::ReachabilityReport,
    ) -> runmat_types::InteropManifest {
        let mut families = std::collections::BTreeSet::new();
        for builtin in report.nodes.iter().filter_map(|node| node.builtin_id()) {
            if let Some(entry) = runmat_builtins::builtin_catalog_entry_by_name(&builtin.0) {
                if let runmat_builtins::BuiltinLinkPolicy::ForeignRuntime(family) =
                    entry.link.policy
                {
                    families.insert(family);
                }
            }
        }
        let python = super::reachability_requires_python(report)
            || families.contains(&runmat_builtins::ForeignAdapterFamily::Python);
        if python {
            families.insert(runmat_builtins::ForeignAdapterFamily::Python);
        }
        if families.is_empty() {
            return runmat_types::InteropManifest::empty();
        }
        let mut adapters = families
            .into_iter()
            .map(|family| runmat_types::ForeignAdapterRequirement {
                adapter: runmat_types::ForeignAdapterId::new(family.adapter_id())
                    .expect("canonical builtin foreign adapter identity"),
                minimum_version: 1,
                capabilities: runmat_types::CapabilitySet(std::collections::BTreeSet::from([
                    runmat_types::CapabilityRequirement::ForeignRuntime,
                ])),
                execution_stack: runmat_types::ExecutionStackRequirement::Process,
                artifact_identities: Vec::new(),
            })
            .collect::<Vec<_>>();
        adapters.sort_by(|left, right| left.adapter.cmp(&right.adapter));
        runmat_types::InteropManifest {
            schema_version: runmat_types::INTEROP_MANIFEST_SCHEMA_VERSION,
            foreign_types: python
                .then(super::python_environment_requirement)
                .into_iter()
                .collect(),
            adapters,
            adapter_contracts: Vec::new(),
        }
    }

    fn component_revisions(
        &self,
    ) -> Result<runmat_execution::ExecutableComponentRevisions, String> {
        let analysis = &self.analysis().revision;
        Ok(runmat_execution::ExecutableComponentRevisions {
            catalog_schema: u16::try_from(analysis.catalog_schema)
                .map_err(|_| "builtin catalog schema exceeds the executable schema".to_string())?,
            catalog_fingerprint: runmat_execution::Digest::from_bytes(analysis.catalog_fingerprint),
            contract_schema: runmat_types::RUNMAT_TYPES_SCHEMA.major,
            contract_fingerprint: runmat_execution::Digest::sha256(format!(
                "runmat-types-contract-v{}.{}",
                runmat_types::RUNMAT_TYPES_SCHEMA.major,
                runmat_types::RUNMAT_TYPES_SCHEMA.minor
            )),
            analysis_schema: analysis.schema_version,
            mir_schema: runmat_mir::MIR_SCHEMA_VERSION,
            bytecode_schema: runmat_vm::BYTECODE_SCHEMA_VERSION,
            vm_layout_schema: runmat_vm::VM_LAYOUT_SCHEMA_VERSION,
            function_registry_schema: runmat_vm::FUNCTION_REGISTRY_SCHEMA_VERSION,
            source_map_schema: SOURCE_MAP_SCHEMA_VERSION,
            region_schema: runmat_types::REGION_CONTRACT_SCHEMA_VERSION,
            interop_schema: runmat_types::INTEROP_MANIFEST_SCHEMA_VERSION,
            parallel_schema: runmat_types::PARALLEL_MANIFEST_SCHEMA_VERSION,
        })
    }

    fn component_payloads(
        &self,
        included: Option<&std::collections::HashSet<runmat_hir::FunctionId>>,
        include_script: bool,
    ) -> Result<Vec<runmat_execution::ExecutableComponentPayload>, String> {
        use runmat_execution::{ExecutableComponentKind as Kind, ExecutableComponentPayload};

        self.source_map().validate_content_revisions()?;
        self.source_map()
            .validate_private_source_owners(self.functions())?;

        let source_map = PortableSourceMap {
            entries: self
                .source_map()
                .entries()
                .iter()
                .map(|entry| PortableSourceEntry {
                    source_id: entry.source_id,
                    owner_identity: &entry.owner_identity,
                    relative_path: &entry.relative_path,
                    content_revision: &entry.content_revision,
                    display_name: &entry.display_name,
                    text: &entry.text,
                })
                .collect(),
        };
        let mut bytecode = if include_script {
            self.bytecode().clone()
        } else {
            runmat_vm::Bytecode::with_instructions(Vec::new(), 0)
        };
        bytecode.bound_functions.clear();
        bytecode.function_registry = runmat_vm::FunctionRegistry::default();
        bytecode.layout = None;
        bytecode.validate_portable_source_origins()?;
        let mut functions = self.functions().clone();
        if let Some(included) = included {
            let omitted = functions
                .functions
                .keys()
                .filter(|function| !included.contains(function))
                .copied()
                .collect::<Vec<_>>();
            for function in omitted {
                functions.remove(function);
            }
        }
        functions.validate_portable_source_origins()?;
        for function in functions.functions.values_mut() {
            if function.binding_origin.is_none() {
                function.binding_origin = Some(
                    runmat_execution::FunctionBindingOrigin::from_revision(
                        &self.revision().program_revision,
                        function.portable_function,
                    )
                    .map_err(|error| error.to_string())?,
                );
            }
            function.source_file = function
                .source_id
                .and_then(runmat_vm::BytecodeSourceRef::program)
                .and_then(|ordinal| {
                    self.source_map()
                        .entries()
                        .get(usize::try_from(ordinal.0).ok()?)
                })
                .map(|entry| entry.relative_path.clone());
        }
        [
            (Kind::Mir, canonical_json(self.mir())?),
            (Kind::Analysis, self.analysis().canonical_bytes()?),
            (Kind::Bytecode, canonical_json(&bytecode)?),
            (Kind::VmLayout, canonical_json(self.vm_layout())?),
            (Kind::FunctionRegistry, canonical_json(&functions)?),
            (Kind::SourceMap, canonical_json(&source_map)?),
        ]
        .into_iter()
        .map(|(kind, bytes)| {
            ExecutableComponentPayload::new(kind, bytes).map_err(|error| error.to_string())
        })
        .collect()
    }
}

pub(super) fn canonical_json(value: &impl Serialize) -> Result<Vec<u8>, String> {
    let value = serde_json::to_value(value).map_err(|error| error.to_string())?;
    serde_json::to_vec(&value).map_err(|error| error.to_string())
}

fn component_descriptors(
    payloads: &[runmat_execution::ExecutableComponentPayload],
    revisions: &runmat_execution::ExecutableComponentRevisions,
) -> Result<Vec<runmat_execution::ExecutableComponentDescriptor>, String> {
    payloads
        .iter()
        .map(|payload| {
            runmat_execution::ExecutableComponentDescriptor::from_payload(
                payload.kind,
                payload.kind.schema_version(revisions),
                &payload.bytes,
            )
            .map_err(|error| error.to_string())
        })
        .collect()
}

trait ComponentSchema {
    fn schema_version(self, revisions: &runmat_execution::ExecutableComponentRevisions) -> u16;
}

impl ComponentSchema for runmat_execution::ExecutableComponentKind {
    fn schema_version(self, revisions: &runmat_execution::ExecutableComponentRevisions) -> u16 {
        match self {
            Self::Mir => revisions.mir_schema,
            Self::Analysis => revisions.analysis_schema,
            Self::Bytecode => revisions.bytecode_schema,
            Self::VmLayout => revisions.vm_layout_schema,
            Self::FunctionRegistry => revisions.function_registry_schema,
            Self::SourceMap => revisions.source_map_schema,
        }
    }
}
