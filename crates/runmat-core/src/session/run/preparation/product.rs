use crate::session::run::{selected_class_product, selected_source_owner};
use crate::session::*;

pub(in crate::session::run) struct PreparedInteractiveProduct {
    pub unit: Rc<crate::ExecutableUnit>,
    pub runtime: runmat_runtime::context::RuntimeContext,
    pub bytecode: runmat_vm::Bytecode,
    pub publication: crate::session::source_evaluation::FunctionRegistryPublication,
    pub durable_registry: runmat_vm::FunctionRegistry,
    pub next_semantic_function_id: usize,
    pub lowering: LoweringResult,
}

impl RunMatSession {
    #[tracing::instrument(
        name = "runtime.prepare_product", skip_all,
        fields(input_len = input.len(), inherited_functions = self.function_registry.functions.len())
    )]
    pub(in crate::session::run) fn prepare_interactive_product(
        &mut self,
        input: &str,
    ) -> std::result::Result<PreparedInteractiveProduct, RunError> {
        let source_span = tracing::info_span!("runtime.prepare_product.source").entered();
        let source = self.canonical_interactive_source(input)?;
        // This is the exact bound-function table supplied to HIR lowering.
        // A later merged/current registry can reuse the same numeric ID for
        // a different body and cannot select an External static target.
        let inherited_registry = self.function_registry.clone();
        drop(source_span);
        let mut prepared = self.compile_input(input)?;
        let identity_span = tracing::info_span!("runtime.prepare_product.identity").entered();
        let revision = crate::ExecutableRevision::derive(
            &source,
            self.runtime_context.program_revision().cloned(),
            crate::program_environment(self.compat_mode),
        );
        let source_map = prepared
            .portable_source_map
            .take()
            .expect("the canonical frontend always retains its program source table");
        prepared
            .attach_function_origins(&revision.program_revision, &source_map)
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error.to_string())
                        .with_identifier("RunMat:ExecutableFunctionIdentity")
                        .with_source(error)
                        .build(),
                )
            })?;
        let PreparedExecution {
            ast,
            lowering,
            mir,
            analysis,
            bytecode,
            portable_source_map: _,
            source_projection,
            mut function_registry_after_success,
            published_registry_for_reuse,
            private_companion_function_names,
            mut current_function_ids,
            mut next_semantic_function_id_after_success,
        } = prepared;
        drop(identity_span);
        let mut draft = crate::executable::ExecutableUnitDraft::prepare(
            source.clone(),
            revision,
            source_map,
            mir,
            analysis,
            bytecode,
        )
        .map_err(|error| {
            RunError::Runtime(
                build_runtime_error(error)
                    .with_identifier("RunMat:ExecutableProduct")
                    .build(),
            )
        })?;
        #[cfg(not(target_arch = "wasm32"))]
        let prepared_foreign = self
            .prepare_selected_project_foreign(&draft.inferred_interop_manifest())
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error)
                        .with_identifier("RunMat:SelectedForeignPreparation")
                        .build(),
                )
            })?;
        #[cfg(not(target_arch = "wasm32"))]
        let foreign_issuance_source = crate::foreign_artifacts::ForeignIssuanceSource::capture(
            self.project_handoff.as_ref(),
            self.selected_foreign_config.as_ref(),
            self.prepared_selected_foreign_input.as_ref(),
        )
        .map_err(|error| {
            RunError::Runtime(
                build_runtime_error(error.to_string())
                    .with_identifier("RunMat:SelectedForeignSource")
                    .build(),
            )
        })?;
        #[cfg(not(target_arch = "wasm32"))]
        let selected_interop = prepared_foreign
            .as_ref()
            .map(|prepared| prepared.interop().clone())
            .unwrap_or_else(runmat_types::InteropManifest::empty);
        #[cfg(target_arch = "wasm32")]
        let selected_interop = runmat_types::InteropManifest::empty();
        #[cfg(not(target_arch = "wasm32"))]
        let selected_foreign_basis = if selected_interop == runmat_types::InteropManifest::empty() {
            None
        } else {
            prepared_foreign
                .as_ref()
                .map(|prepared| prepared.selected_reuse_basis())
                .transpose()
                .map_err(|error| {
                    RunError::Runtime(
                        build_runtime_error(error.to_string())
                            .with_identifier("RunMat:SelectedForeignPreparation")
                            .build(),
                    )
                })?
        };
        #[cfg(target_arch = "wasm32")]
        let selected_foreign_basis = None;
        let reuse_span = tracing::info_span!("runtime.prepare_product.reuse").entered();
        let reuse = crate::session::semantic_definition::unchanged_product_remap(
            draft.bytecode(),
            &current_function_ids,
            &published_registry_for_reuse,
            &private_companion_function_names,
        )
        .filter(|remap| {
            !draft.bytecode().instructions.is_empty()
                && remap.values().all(|target| {
                    published_registry_for_reuse
                        .get_current(*target)
                        .is_some_and(|body| {
                            let Some(product) = body.selected_executable_product.as_ref() else {
                                return false;
                            };
                            body.selected_source_owner.as_ref().is_some_and(|owner| {
                                owner.is_live()
                                    && product.source_owner().same_issuance(owner)
                                    && product.foreign_basis() == selected_foreign_basis
                            })
                        })
                })
        });
        let mut excluded_reused_ids = HashSet::new();
        let mut provisional_reused_ids = HashSet::new();
        if let Some(remap) = reuse {
            excluded_reused_ids.extend(remap.values().copied());
            provisional_reused_ids.extend(remap.keys().copied());
            draft
                .rebind_current_definitions(&published_registry_for_reuse, &remap)
                .map_err(|error| {
                    RunError::Runtime(
                        build_runtime_error(error)
                            .with_identifier("RunMat:ExecutableFunctionIdentity")
                            .build(),
                    )
                })?;
            current_function_ids = current_function_ids
                .into_iter()
                .map(|id| remap.get(&id).copied().unwrap_or(id))
                .collect();
            // The provisional IDs never escaped the draft. Reuse retains the
            // published high-water mark rather than consuming one per repeat.
            next_semantic_function_id_after_success = self.next_semantic_function_id;
        }
        let selected_current_function_ids = current_function_ids
            .difference(&excluded_reused_ids)
            .copied()
            .collect::<HashSet<_>>();
        drop(reuse_span);
        let dependencies_span =
            tracing::info_span!("runtime.prepare_product.dependencies").entered();
        // Keep one immutable code owner while an accepted deferred task is
        // outstanding; canonical V3 bytes remain lazy until graph use.
        let interactive_unit = self
            .program_service
            .bind_original_dependencies(draft.seal())
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error)
                        .with_identifier("RunMat:ExecutableProduct")
                        .build(),
                )
            })?;
        let selected_static_imports = crate::session::source_evaluation::selected_static_imports(
            &interactive_unit,
            &inherited_registry,
            &selected_current_function_ids,
            &excluded_reused_ids,
        )
        .map_err(|error| {
            RunError::Runtime(
                build_runtime_error(error)
                    .with_identifier("RunMat:ExecutableStaticImport")
                    .build(),
            )
        })?;
        let interactive_unit = Rc::new(
            interactive_unit
                .with_portable_selection(
                    selected_current_function_ids.clone(),
                    selected_static_imports.imports.clone(),
                    selected_static_imports.targets.clone(),
                )
                .map_err(|error| {
                    RunError::Runtime(
                        build_runtime_error(error)
                            .with_identifier("RunMat:ExecutableProduct")
                            .build(),
                    )
                })?,
        );
        drop(dependencies_span);
        let binding_span = tracing::info_span!("runtime.prepare_product.bind_sources").entered();
        let execution_runtime = interactive_unit
            .bind_runtime_sources(
                self.runtime_context
                    .clone()
                    .with_cancellation_token(self.runtime_context.cancellation())
                    .with_program_revision(Some(
                        interactive_unit.revision().program_revision.clone(),
                    )),
            )
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error)
                        .with_identifier("RunMat:ExecutableSourceIdentity")
                        .build(),
                )
            })?;
        let mut bytecode = interactive_unit
            .runtime_bytecode_with_selected_prior(
                &execution_runtime,
                &inherited_registry,
                &selected_current_function_ids,
                &excluded_reused_ids,
                &source_projection,
            )
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error)
                        .with_identifier("RunMat:ExecutableSourceIdentity")
                        .build(),
                )
            })?;
        drop(binding_span);
        let registry_span = tracing::info_span!("runtime.prepare_product.registry").entered();
        let mut active_registry = bytecode.function_registry();
        let mut publication_projection = function_registry_after_success.clone();
        for function in &provisional_reused_ids {
            publication_projection.remove(*function);
        }
        let publishable_current = selected_current_function_ids
            .iter()
            .copied()
            .filter(|function| publication_projection.get_current(*function).is_some())
            .collect::<HashSet<_>>();
        function_registry_after_success =
            runmat_vm::FunctionRegistry::selected_execution_registry_from_prior(
                &self.function_registry,
                &publication_projection,
                &publishable_current,
                &excluded_reused_ids,
                |source| source_projection.runtime_ref_for_program_ref(source),
            )
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error)
                        .with_identifier("RunMat:ExecutableFunctionIdentity")
                        .build(),
                )
            })?;
        crate::session::semantic_definition::finalize_current_definitions(
            &mut function_registry_after_success,
            &selected_current_function_ids,
            &active_registry,
        )
        .map_err(|error| {
            RunError::Runtime(
                build_runtime_error(error)
                    .with_identifier("RunMat:ExecutableFunctionIdentity")
                    .build(),
            )
        })?;
        selected_source_owner::attach_published_aliases(
            &interactive_unit,
            &execution_runtime,
            &selected_current_function_ids,
            &active_registry,
            &mut function_registry_after_success,
        )?;
        let selected_declaration_products = self
            .source_evaluator
            .selected_declaration_products()
            .map_err(RunError::Runtime)?;
        selected_class_product::attach_published_constructors(
            &mut bytecode,
            &selected_current_function_ids,
            &mut active_registry,
            &mut function_registry_after_success,
            &selected_declaration_products,
        )?;
        drop(registry_span);
        let artifacts_span = tracing::info_span!("runtime.prepare_product.artifacts").entered();
        // Only this compilation's definitions need a new artifact. Inherited
        // session definitions already retain their own indexed envelopes.
        let has_current_artifact_origin = selected_current_function_ids.iter().any(|function| {
            active_registry
                .get(*function)
                .is_some_and(|body| body.binding_origin.is_some())
        });
        let source_selection = crate::executable::SourceProductSelection::classify(
            &interactive_unit,
            &selected_current_function_ids,
            lowering.assembly.primary_source,
        )
        .map_err(|error| {
            RunError::Runtime(
                build_runtime_error(error)
                    .with_identifier("RunMat:ExecutableClassProduct")
                    .build(),
            )
        })?;
        // A declaration-only class may still own method bodies. Its methods
        // retain their callable artifact while class metadata retains the
        // declaration product; neither role manufactures a script entrypoint.
        let declaration_class = source_selection.declaration().cloned();
        let callable_artifact = if has_current_artifact_origin {
            Some(Arc::<[u8]>::from(
                source_selection
                    .envelope(
                        &interactive_unit,
                        &selected_current_function_ids,
                        lowering.assembly.primary_source,
                        selected_interop.clone(),
                    )
                    .and_then(|mut envelope| {
                        if !selected_static_imports.imports.is_empty() {
                            let section = runmat_execution::executable::static_import_section(
                                selected_static_imports.imports.clone(),
                            )
                            .map_err(|error| error.to_string())?;
                            envelope.manifest.optional_sections.push(section);
                            envelope
                                .manifest
                                .optional_sections
                                .sort_by(|left, right| left.name.cmp(&right.name));
                        }
                        envelope
                            .canonical_bytes()
                            .map_err(|error| error.to_string())
                    })
                    .map_err(|error| {
                        RunError::Runtime(
                            build_runtime_error(error)
                                .with_identifier("RunMat:CallableArtifact:ExecutableProduct")
                                .build(),
                        )
                    })?,
            ))
        } else {
            None
        };
        let callable_receipt = if let Some(bytes) = callable_artifact.as_ref() {
            selected_source_owner::attach_selected_executable_product(
                selected_source_owner::SelectedProductPublication {
                    unit: &interactive_unit,
                    current: &selected_current_function_ids,
                    bytecode: &mut bytecode,
                    executing: &mut active_registry,
                    durable: &mut function_registry_after_success,
                },
                selected_source_owner::SelectedProductAdmission {
                    bytes: Arc::clone(bytes),
                    static_targets: &selected_static_imports.targets,
                    foreign_basis: selected_foreign_basis,
                    dynamic_port: {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            self.source_evaluator.dynamic_admission_port_for_invocation(
                                prepared_foreign.clone(),
                                foreign_issuance_source.clone(),
                            )
                        }
                        #[cfg(target_arch = "wasm32")]
                        {
                            self.source_evaluator.dynamic_admission_port()
                        }
                    },
                    #[cfg(not(target_arch = "wasm32"))]
                    prepared: prepared_foreign.as_ref(),
                },
            )?
        } else {
            None
        };
        let requires_foreign_receipt = selected_interop != runmat_types::InteropManifest::empty()
            || interactive_unit.requires_python_runtime();
        // Preserve the compiler's exact private-companion publication choice:
        // those selected bodies execute from this request but are not named in
        // the durable session registry.
        let execution_only = selected_current_function_ids
            .iter()
            .copied()
            .filter(|function| {
                active_registry.get_current(*function).is_some_and(|body| {
                    private_companion_function_names.contains(&body.display_name)
                })
            })
            .collect::<HashSet<_>>();
        let mut selected_static_targets = Some(selected_static_imports.targets);
        let graph_span = tracing::info_span!("runtime.prepare_product.admit_artifacts").entered();
        let selected_graph = if declaration_class.is_none()
            && (callable_artifact.is_some() || !interactive_unit.bytecode().instructions.is_empty())
        {
            let primary = crate::session::source_evaluation::SelectedGraphPrimary::new(
                crate::session::source_evaluation::SelectedGraphSource {
                    unit: Rc::clone(&interactive_unit),
                    primary_source: lowering.assembly.primary_source,
                    code: callable_artifact.clone()
                        .map(crate::session::source_evaluation::SelectedGraphCodeSource::PublishedBytes)
                        .unwrap_or(crate::session::source_evaluation::SelectedGraphCodeSource::Unpublished),
                },
                &execution_runtime,
                bytecode.function_registry.clone(),
                crate::session::source_evaluation::SelectedGraphPublication::new(
                    &function_registry_after_success,
                    &execution_only,
                ),
                crate::session::source_evaluation::SelectedGraphSelection {
                    current_functions: selected_current_function_ids.clone(),
                    interop: selected_interop.clone(),
                    static_targets: selected_static_targets.take().ok_or_else(|| {
                        RunError::Runtime(
                            build_runtime_error("selected static targets have no product owner")
                                .with_identifier("RunMat:ExecutableStaticImport")
                                .build(),
                        )
                    })?,
                },
            )
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error)
                        .with_identifier("RunMat:ExecutableSourceIdentity")
                        .build(),
                )
            })?;
            let interop_receipt = if callable_artifact.is_some() {
                callable_receipt
            } else if requires_foreign_receipt {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    crate::foreign_artifacts::selected_unit_interop_receipt(
                        primary.canonical_v3().map_err(|error| {
                            RunError::Runtime(build_runtime_error(error).build())
                        })?,
                        primary.source_owner(),
                        prepared_foreign.as_ref(),
                    )
                    .map_err(|error| RunError::Runtime(build_runtime_error(error).build()))?
                }
                #[cfg(target_arch = "wasm32")]
                {
                    None
                }
            } else {
                None
            };
            Some(Rc::new(
                primary
                    .with_selected_interop_receipt(interop_receipt)
                    .map_err(|error| RunError::Runtime(build_runtime_error(error).build()))?,
            ))
        } else {
            None
        };
        let selected_declaration = if let Some(class) = declaration_class {
            let primary_source = lowering.assembly.primary_source.ok_or_else(|| {
                RunError::Runtime(
                    build_runtime_error("declaration-only product has no primary source")
                        .with_identifier("RunMat:ExecutableClassProduct")
                        .build(),
                )
            })?;
            let bytes = if let Some(bytes) = callable_artifact.as_ref() {
                // A declaration with methods already issued these exact bytes
                // above. Both roles share the immutable artifact backing.
                Arc::clone(bytes)
            } else {
                let mut envelope = interactive_unit
                    .portable_declaration_envelope_with_interop(
                        &class,
                        primary_source,
                        selected_interop,
                        &selected_current_function_ids,
                    )
                    .map_err(|error| RunError::Runtime(build_runtime_error(error).build()))?;
                if !selected_static_imports.imports.is_empty() {
                    let section = runmat_execution::executable::static_import_section(
                        selected_static_imports.imports,
                    )
                    .map_err(|error| {
                        RunError::Runtime(build_runtime_error(error.to_string()).build())
                    })?;
                    envelope.manifest.optional_sections.push(section);
                    envelope
                        .manifest
                        .optional_sections
                        .sort_by(|left, right| left.name.cmp(&right.name));
                }
                Arc::<[u8]>::from(envelope.canonical_bytes().map_err(|error| {
                    RunError::Runtime(build_runtime_error(error.to_string()).build())
                })?)
            };
            let source_owner = interactive_unit
                .selected_source_owner(&execution_runtime)
                .map_err(|error| RunError::Runtime(build_runtime_error(error).build()))?;
            #[cfg(not(target_arch = "wasm32"))]
            let receipt = crate::foreign_artifacts::selected_unit_interop_receipt(
                Arc::clone(&bytes),
                &source_owner,
                prepared_foreign.as_ref(),
            )
            .map_err(|error| RunError::Runtime(build_runtime_error(error).build()))?;
            #[cfg(target_arch = "wasm32")]
            let receipt = None;
            let product =
                runmat_vm::SelectedDeclarationProduct::admit_local(bytes, source_owner, receipt)
                    .map_err(RunError::Runtime)?;
            Some(
                crate::session::source_evaluation::SelectedDeclarationAdmission::new(
                    product,
                    selected_static_targets.take().ok_or_else(|| {
                        RunError::Runtime(
                            build_runtime_error("selected static targets have no product owner")
                                .with_identifier("RunMat:ExecutableStaticImport")
                                .build(),
                        )
                    })?,
                )
                .map_err(RunError::Runtime)?,
            )
        } else {
            None
        };
        if selected_graph.is_none()
            && selected_declaration.is_none()
            && selected_static_targets
                .as_ref()
                .is_some_and(|targets| !targets.is_empty())
        {
            return Err(RunError::Runtime(
                build_runtime_error("static imports have no selected executable product")
                    .with_identifier("RunMat:ExecutableStaticImport")
                    .build(),
            ));
        }
        #[cfg(not(target_arch = "wasm32"))]
        let selected_foreign_scope = if let Some(primary) = selected_graph.as_ref() {
            primary
                .selected_interop_receipt()
                .map(|receipt| {
                    crate::foreign_artifacts::SelectedForeignPreparation::from_selected_receipt(
                        receipt,
                        &primary.canonical_v3().map_err(anyhow::Error::msg)?,
                        primary.source_owner(),
                    )
                })
                .transpose()
        } else if let Some(declaration) = selected_declaration.as_ref() {
            let product = declaration.product();
            product
                .interop_receipt()
                .map(|receipt| {
                    crate::foreign_artifacts::SelectedForeignPreparation::from_selected_receipt(
                        receipt,
                        product.canonical_bytes(),
                        product.source_owner(),
                    )
                })
                .transpose()
        } else {
            Ok(None)
        }
        .map_err(|error| RunError::Runtime(build_runtime_error(error.to_string()).build()))?;
        drop(graph_span);
        drop(artifacts_span);
        let _publication_span = tracing::info_span!("runtime.prepare_product.publish").entered();
        let source_registry_publication = self
            .source_evaluator
            .publish_for_execution(active_registry, callable_artifact)
            .map_err(RunError::Runtime)?;
        if let Some(declaration) = selected_declaration.as_ref() {
            source_registry_publication
                .attach_selected_declaration(Rc::clone(declaration))
                .map_err(RunError::Runtime)?;
        }
        // Adapter installation can invoke user callbacks. Publish this
        // request's immutable foreign source capability transactionally before
        // the first install effect; Drop restores the prior scope on failure.
        #[cfg(not(target_arch = "wasm32"))]
        let mut source_registry_publication = source_registry_publication;
        #[cfg(not(target_arch = "wasm32"))]
        source_registry_publication.set_foreign_preparation(selected_foreign_scope);
        #[cfg(not(target_arch = "wasm32"))]
        source_registry_publication.set_foreign_issuance_source(foreign_issuance_source);
        #[cfg(not(target_arch = "wasm32"))]
        let execution_runtime = if let Some(selected_graph) = selected_graph {
            self.source_evaluator
                .bind_selected_foreign(
                    execution_runtime,
                    Rc::clone(&selected_graph),
                    None,
                    selected_graph.selected_interop_receipt().cloned(),
                    Some(Rc::clone(&self.mex_runtime)),
                )
                .map_err(RunError::Runtime)?
        } else if let Some(declaration) = selected_declaration {
            self.source_evaluator
                .bind_selected_declaration_foreign(execution_runtime, declaration)
                .map_err(RunError::Runtime)?
        } else {
            let service_ports = execution_runtime.service_ports().clone().without_foreign();
            execution_runtime
                .without_selected_graph_program()
                .with_service_ports(service_ports)
        };
        #[cfg(target_arch = "wasm32")]
        let execution_runtime = if let Some(selected_graph) = selected_graph {
            execution_runtime
                .without_selected_graph_program()
                .with_selected_graph_program(
                    crate::session::source_evaluation::SelectedGraphSnapshot::program(
                        selected_graph,
                        &self.source_evaluator,
                    ),
                )
        } else {
            execution_runtime.without_selected_graph_program()
        };
        if self.verbose {
            debug!("AST: {ast:?}");
        }
        Ok(PreparedInteractiveProduct {
            unit: interactive_unit,
            runtime: execution_runtime,
            bytecode,
            publication: source_registry_publication,
            durable_registry: function_registry_after_success,
            next_semantic_function_id: next_semantic_function_id_after_success,
            lowering,
        })
    }
}
