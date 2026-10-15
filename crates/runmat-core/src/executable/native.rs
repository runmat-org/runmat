use std::collections::BTreeMap;

use super::ExecutableUnit;

#[path = "native/class_retention.rs"]
mod class_retention;

/// Owned canonical frontend product used by both background JIT compilation
/// and ahead-of-time object emission.
///
/// Keeping this preparation in Core prevents either native product from
/// rebuilding MIR, analysis facts, executable identity, or binding names.
pub struct NativeCompilationInput {
    mir: runmat_mir::MirAssembly,
    analysis: runmat_mir::analysis::AnalysisStore,
    manifest: runmat_execution::ExecutableUnitManifest,
    source_map: runmat_execution::ExecutableSourceMapV2,
    binding_names: BTreeMap<runmat_types::BindingId, String>,
    entrypoint: runmat_native_codegen::NativeEntrypointBinding,
    program_capture: Vec<u8>,
    interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
    coverage_sites: BTreeMap<runmat_types::ProgramPointId, Vec<u64>>,
}

pub struct AotObjectDataOptions {
    pub runtime_binding_mode: runmat_native_codegen::aot::AotRuntimeBindingMode,
    pub builtin_bindings: Vec<runmat_native_codegen::aot::AotBuiltinBinding>,
    pub foreign_policy: Option<runmat_execution_artifact::LogicalObject>,
    pub native_interfaces: Vec<u8>,
    pub mex_artifacts: Vec<u8>,
    pub java_artifacts: Vec<u8>,
    pub python_artifacts: Vec<u8>,
}

impl NativeCompilationInput {
    pub fn entrypoint(&self) -> runmat_types::ProgramFunctionId {
        self.entrypoint.compiled()
    }

    pub fn lower(
        &self,
        target: runmat_native_codegen::NativeTarget,
    ) -> Result<runmat_native_codegen::NativeAssembly, runmat_native_codegen::NativeCodegenError>
    {
        self.validate_entrypoint().map_err(|error| {
            runmat_native_codegen::NativeCodegenError::new(
                "native.lowering.entrypoint_binding",
                error.to_string(),
            )
        })?;
        runmat_native_codegen::lower_executable(runmat_native_codegen::NativeLoweringInput {
            mir: &self.mir,
            analysis: &self.analysis,
            manifest: &self.manifest,
            entrypoint: self.entrypoint.clone(),
            binding_names: Some(&self.binding_names),
            target,
        })
    }

    fn validate_entrypoint(&self) -> Result<(), runmat_runtime::RuntimeError> {
        if self.manifest.identity.entrypoint_kind
            == runmat_execution::ExecutableEntrypointKind::DeclarationOnly
        {
            return Err(native_product_error(
                "declaration-only executable has no native entrypoint".into(),
            ));
        }
        if self.manifest.identity.entrypoint_kind
            == runmat_execution::ExecutableEntrypointKind::Script
        {
            return Ok(());
        }
        let registry = runmat_vm::decode_interpreter_program_v2(&self.program_capture)
            .map_err(|error| native_product_error(error.to_string()))?;
        let function = runmat_hir::FunctionId(
            usize::try_from(self.entrypoint.compiled().0)
                .map_err(|_| native_product_error("native entrypoint exceeds this host".into()))?,
        );
        let body = registry.get(function).ok_or_else(|| {
            native_product_error("selected native function is absent from its capture".into())
        })?;
        let expected = runmat_execution::FunctionBindingOrigin::from_revision(
            &self.manifest.identity.program,
            self.entrypoint.portable(),
        )
        .map_err(|error| native_product_error(error.to_string()))?;
        if body.portable_function != self.entrypoint.portable()
            || body.binding_origin != Some(expected)
        {
            return Err(native_product_error(
                "selected native function differs from its admitted portable origin".into(),
            ));
        }
        Ok(())
    }

    pub fn program_capture(&self) -> &[u8] {
        &self.program_capture
    }

    pub fn interpreter_resume_points(&self) -> &BTreeMap<runmat_types::ProgramPointId, u64> {
        &self.interpreter_resume_points
    }

    pub fn coverage_sites(&self) -> &BTreeMap<runmat_types::ProgramPointId, Vec<u64>> {
        &self.coverage_sites
    }

    pub fn retain_functions(
        mut self,
        retained: &std::collections::BTreeSet<runmat_types::ProgramFunctionId>,
    ) -> Result<Self, runmat_runtime::RuntimeError> {
        if !retained.contains(&self.entrypoint.compiled()) {
            return Err(native_product_error(
                "native retention set omits the executable entrypoint".into(),
            ));
        }
        let retained = class_retention::close(&self, retained)?;
        let retained_local = retained
            .iter()
            .map(|function| {
                usize::try_from(function.0)
                    .map(runmat_hir::FunctionId)
                    .map_err(|_| native_product_error("function identity exceeds this host".into()))
            })
            .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
        self.mir
            .bodies
            .retain(|function, _| retained_local.contains(function));
        self.mir
            .functions
            .retain(|function, _| retained_local.contains(function));
        self.mir
            .entrypoints
            .retain(|function| retained_local.contains(function));
        self.analysis
            .functions
            .retain(|function| retained.contains(&function.function));
        self.analysis
            .program_points
            .retain(|point| retained.contains(&point.point.function));
        self.analysis
            .regions
            .retain(|region| retained.contains(&region.contract.id.function));
        for class in &mut self.analysis.classes {
            class.methods.retain(|function| retained.contains(function));
        }
        self.interpreter_resume_points
            .retain(|point, _| retained.contains(&point.function));
        self.coverage_sites
            .retain(|point, _| retained.contains(&point.function));
        self.manifest
            .regions
            .retain(|region| retained.contains(&region.id.function));
        self.manifest
            .parallel
            .parfor_regions
            .retain(|region| retained.contains(&region.id.0.function));
        self.manifest
            .parallel
            .spmd_regions
            .retain(|region| retained.contains(&region.id.0.function));
        self.manifest
            .parallel
            .distributed_values
            .retain(|value| retained.contains(&value.id.function));
        self.manifest
            .parallel
            .collectives
            .retain(|collective| retained.contains(&collective.id.region.0.function));
        let mut registry: runmat_vm::FunctionRegistry =
            runmat_vm::decode_interpreter_program_v2(&self.program_capture).map_err(|error| {
                native_product_error(format!("failed to decode native program capture: {error}"))
            })?;
        registry
            .functions
            .retain(|function, _| retained_local.contains(function));
        registry
            .names
            .retain(|_, function| retained_local.contains(function));
        registry.source_functions.retain(|_, functions| {
            functions.retain(|function| retained_local.contains(function));
            !functions.is_empty()
        });
        self.program_capture =
            runmat_vm::encode_interpreter_program_v2(&registry).map_err(|error| {
                native_product_error(format!("failed to encode retained native program: {error}"))
            })?;
        Ok(self)
    }

    pub fn aot_object_data(
        &self,
        assembly: &runmat_native_codegen::NativeAssembly,
        options: AotObjectDataOptions,
    ) -> Result<
        Vec<runmat_native_codegen::aot::NativeObjectData>,
        runmat_native_codegen::NativeCodegenError,
    > {
        self.aot_object_data_for_product(assembly, options, None)
    }

    pub fn aot_object_data_for_selected(
        &self,
        assembly: &runmat_native_codegen::NativeAssembly,
        options: AotObjectDataOptions,
        selected: &runmat_native_codegen::aot::SelectedNativeProduct<'_>,
    ) -> Result<
        Vec<runmat_native_codegen::aot::NativeObjectData>,
        runmat_native_codegen::NativeCodegenError,
    > {
        selected.validate_assembly(assembly)?;
        self.aot_object_data_for_product(assembly, options, Some(selected))
    }

    fn aot_object_data_for_product(
        &self,
        assembly: &runmat_native_codegen::NativeAssembly,
        options: AotObjectDataOptions,
        selected: Option<&runmat_native_codegen::aot::SelectedNativeProduct<'_>>,
    ) -> Result<
        Vec<runmat_native_codegen::aot::NativeObjectData>,
        runmat_native_codegen::NativeCodegenError,
    > {
        self.validate_entrypoint().map_err(|error| {
            runmat_native_codegen::NativeCodegenError::new(
                "native.object.entrypoint_binding",
                error.to_string(),
            )
        })?;
        if assembly.executable_identity != self.manifest.identity
            || assembly.selected_entrypoint != self.entrypoint.compiled()
            || assembly.selected_portable_entrypoint != self.entrypoint.portable()
        {
            return Err(runmat_native_codegen::NativeCodegenError::new(
                "native.object.executable_identity",
                "AOT embedded data does not belong to the selected native assembly",
            ));
        }
        runmat_native_codegen::verify_against_mir(assembly, &self.mir, Some(&self.binding_names))?;
        let native_ir = serde_json::to_vec(assembly).map_err(|error| {
            runmat_native_codegen::NativeCodegenError::new(
                "native.object.native_ir",
                format!("failed to encode embedded Native IR: {error}"),
            )
        })?;
        let mut program = runmat_native_codegen::aot::AotProgramManifest::from_assembly(
            assembly,
            runmat_execution::Digest::sha256(&native_ir),
            options.foreign_policy,
            self.source_map.clone(),
            options.runtime_binding_mode,
            options.builtin_bindings,
        )?;
        if let Some(selected) = selected {
            // Same verified selection as the object producer; no decoder or
            // absent-field fallback is allowed to infer this commitment.
            program.product = selected.product().clone();
        }
        let program = program.canonical_bytes()?;
        let ordered_resume_points = self
            .interpreter_resume_points
            .iter()
            .map(|(point, pc)| (*point, *pc))
            .collect::<Vec<_>>();
        let resume_points = serde_json::to_vec(&ordered_resume_points).map_err(|error| {
            runmat_native_codegen::NativeCodegenError::new(
                "native.object.resume_points",
                format!("failed to encode embedded resume points: {error}"),
            )
        })?;
        let mut data = Vec::with_capacity(7);
        for blob in [
            runmat_native_codegen::aot::embedded_blob(
                runmat_native_codegen::aot::AOT_NATIVE_IR_SYMBOL,
                native_ir,
                8,
            )?,
            runmat_native_codegen::aot::embedded_blob(
                runmat_native_codegen::aot::AOT_PROGRAM_SYMBOL,
                program,
                8,
            )?,
            runmat_native_codegen::aot::embedded_blob(
                runmat_native_codegen::aot::AOT_RESUME_POINTS_SYMBOL,
                resume_points,
                8,
            )?,
            runmat_native_codegen::aot::embedded_blob(
                runmat_native_codegen::aot::AOT_NATIVE_INTERFACES_SYMBOL,
                options.native_interfaces,
                8,
            )?,
            runmat_native_codegen::aot::embedded_blob(
                runmat_native_codegen::aot::AOT_MEX_ARTIFACTS_SYMBOL,
                options.mex_artifacts,
                8,
            )?,
            runmat_native_codegen::aot::embedded_blob(
                runmat_native_codegen::aot::AOT_JAVA_ARTIFACTS_SYMBOL,
                options.java_artifacts,
                8,
            )?,
            runmat_native_codegen::aot::embedded_blob(
                runmat_native_codegen::aot::AOT_PYTHON_ARTIFACTS_SYMBOL,
                options.python_artifacts,
                8,
            )?,
        ] {
            data.push(blob);
        }
        Ok(data)
    }
}

impl ExecutableUnit {
    pub fn prepare_native_compilation(
        &self,
    ) -> Result<NativeCompilationInput, runmat_runtime::RuntimeError> {
        self.prepare_native_compilation_for(None)
    }

    pub fn prepare_native_compilation_for(
        &self,
        preferred_function: Option<&str>,
    ) -> Result<NativeCompilationInput, runmat_runtime::RuntimeError> {
        self.prepare_native_compilation_for_with_interop(
            preferred_function,
            runmat_types::InteropManifest::empty(),
        )
    }

    pub fn prepare_native_compilation_for_with_interop(
        &self,
        preferred_function: Option<&str>,
        interop: runmat_types::InteropManifest,
    ) -> Result<NativeCompilationInput, runmat_runtime::RuntimeError> {
        let envelope = self
            .portable_envelope_for_with_interop(preferred_function, interop)
            .map_err(native_product_error)?;
        let selected = super::portable_entrypoint::resolve(self, preferred_function)
            .map_err(native_product_error)?;
        let (compiled_entrypoint, portable_function) = match selected {
            super::portable_entrypoint::PortableEntrypoint::Callable {
                compiled_function,
                function,
                kind,
                ..
            } => {
                if kind == runmat_execution::ExecutableEntrypointKind::Function {
                    let body = self.functions().get(compiled_function).ok_or_else(|| {
                        native_product_error(
                            "native entrypoint has no exact selected function body".into(),
                        )
                    })?;
                    let expected = runmat_execution::FunctionBindingOrigin::from_revision(
                        &envelope.manifest.identity.program,
                        function,
                    )
                    .map_err(|error| native_product_error(error.to_string()))?;
                    if body.portable_function != function
                        || body.binding_origin.is_some_and(|origin| origin != expected)
                    {
                        return Err(native_product_error(
                            "native entrypoint lies outside its portable executable origin".into(),
                        ));
                    }
                }
                (compiled_function, function)
            }
            super::portable_entrypoint::PortableEntrypoint::DeclarationOnly(_) => {
                return Err(native_product_error(
                    "declaration-only executable has no native callable entrypoint".into(),
                ));
            }
        };
        let entrypoint = runmat_native_codegen::NativeEntrypointBinding::new(
            self.mir(),
            compiled_entrypoint,
            portable_function,
        )
        .map_err(|error| native_product_error(error.to_string()))?;
        let mut captured_functions = self.functions().clone();
        for body in captured_functions.functions.values_mut() {
            if body.binding_origin.is_none() {
                body.binding_origin = Some(
                    runmat_execution::FunctionBindingOrigin::from_revision(
                        &envelope.manifest.identity.program,
                        body.portable_function,
                    )
                    .map_err(|error| native_product_error(error.to_string()))?,
                );
            }
        }
        let program_capture = runmat_vm::encode_interpreter_program_v2(&captured_functions)
            .map_err(|error| {
                native_product_error(format!("failed to capture native program: {error}"))
            })?;
        let mut interpreter_resume_points = self
            .vm_layout()
            .functions
            .values()
            .flat_map(|function| function.resume_points.iter())
            .map(|(point, pc)| {
                u64::try_from(*pc).map(|pc| (*point, pc)).map_err(|_| {
                    native_product_error("bytecode resume PC exceeds native ABI".into())
                })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        for function in self.functions().functions.values() {
            for (point, pc) in &function.resume_points {
                let pc = u64::try_from(*pc).map_err(|_| {
                    native_product_error("bytecode resume PC exceeds native ABI".into())
                })?;
                interpreter_resume_points.insert(*point, pc);
            }
        }
        let mut coverage_sites = BTreeMap::new();
        let mut assigned_coverage = std::collections::BTreeSet::new();
        for function in self.functions().functions.values() {
            for (point, pc) in &function.resume_points {
                if let Some(sites) = function
                    .coverage_sites
                    .get(*pc)
                    .filter(|sites| !sites.is_empty())
                {
                    let sites = sites
                        .iter()
                        .copied()
                        .filter(|site| assigned_coverage.insert(*site))
                        .collect::<Vec<_>>();
                    if !sites.is_empty() {
                        coverage_sites.insert(*point, sites);
                    }
                }
            }
        }
        if let Some(layout) = self
            .vm_layout()
            .functions
            .values()
            .find(|layout| layout.function == compiled_entrypoint)
        {
            for (point, pc) in &layout.resume_points {
                if let Some(sites) = self
                    .bytecode()
                    .coverage_sites
                    .get(*pc)
                    .filter(|sites| !sites.is_empty())
                {
                    let sites = sites
                        .iter()
                        .copied()
                        .filter(|site| assigned_coverage.insert(*site))
                        .collect::<Vec<_>>();
                    if !sites.is_empty() {
                        coverage_sites.insert(*point, sites);
                    }
                }
            }
        }
        Ok(NativeCompilationInput {
            mir: self.mir().clone(),
            analysis: self.analysis().clone(),
            entrypoint,
            manifest: envelope.manifest,
            source_map: self
                .source_map()
                .portable_v2()
                .map_err(native_product_error)?,
            binding_names: self.binding_names(),
            program_capture,
            interpreter_resume_points,
            coverage_sites,
        })
    }
}

fn native_product_error(message: String) -> runmat_runtime::RuntimeError {
    runmat_runtime::build_runtime_error(message)
        .with_identifier("RunMat:NativeProduct")
        .build()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use futures::executor::block_on;

    use super::AotObjectDataOptions;
    use crate::{execute_text_request_for_testing, ExecutableSource, RunMatSession};

    #[test]
    fn native_entrypoint_projects_compiler_identity_without_rewriting_portable_origin() {
        let mut session = RunMatSession::with_options(false, false).expect("session init");
        let seed = execute_text_request_for_testing(
            &mut session,
            "seed = 0;\nfunction y = nativeIdentitySeed(x)\ny = x;\nend",
        )
        .expect("publish preceding session definition");
        assert!(
            seed.error.is_none(),
            "preceding session definition must publish without a diagnostic error: {:?}",
            seed.error
        );
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(
                "native-identity@1",
                "nativeIdentityProbe.m",
                "function y = nativeIdentityProbe(x)\ny = x + 1;\nend\nfunction y = nativeIdentityDecoy(x)\ny = x - 1;\nend",
            ),
            None,
        ))
        .expect("compile remapped native fixture");
        let compiled = unit
            .functions()
            .resolve_name("nativeIdentityProbe")
            .expect("compiled function identity");
        let portable = unit
            .functions()
            .get(compiled)
            .expect("compiled function body")
            .portable_function;
        let compiled = runmat_types::ProgramFunctionId(
            u32::try_from(compiled.0).expect("compiled identity fits native schema"),
        );
        assert_ne!(compiled, portable, "fixture must exercise remapping");

        let input = unit
            .prepare_native_compilation_for(Some("nativeIdentityProbe"))
            .expect("prepare native input");
        assert_eq!(input.entrypoint(), compiled);
        assert_eq!(input.manifest.identity.entrypoint_function, Some(portable));
        let assembly = input
            .lower(runmat_native_codegen::NativeTarget::current())
            .expect("lower remapped native entrypoint");
        assert!(assembly.entrypoints.contains(&compiled));
        assert_eq!(assembly.selected_entrypoint, compiled);
        assert_eq!(assembly.selected_portable_entrypoint, portable);
        assert_eq!(
            assembly.executable_identity.entrypoint_function,
            Some(portable)
        );

        let program = runmat_native_codegen::aot::AotProgramManifest::from_assembly(
            &assembly,
            runmat_execution::Digest::sha256(b"native identity fixture"),
            None,
            unit.source_map()
                .portable_v2()
                .expect("portable source map"),
            runmat_native_codegen::aot::AotRuntimeBindingMode::Dynamic,
            Vec::new(),
        )
        .expect("project AOT program identity");
        assert_eq!(program.entrypoint, compiled);
        assert_eq!(program.executable.entrypoint_function, Some(portable));

        let decoy = unit
            .functions()
            .resolve_name("nativeIdentityDecoy")
            .expect("decoy compiled identity");
        let mut wrong = unit
            .prepare_native_compilation_for(Some("nativeIdentityProbe"))
            .expect("prepare adversarial native input");
        wrong.entrypoint =
            runmat_native_codegen::NativeEntrypointBinding::new(&wrong.mir, decoy, portable)
                .expect("construct mismatched retained selector");
        let error = wrong
            .lower(runmat_native_codegen::NativeTarget::current())
            .expect_err("wrong retained selector must fail before native lowering");
        assert_eq!(error.code, "native.lowering.entrypoint_binding");
    }

    #[test]
    fn native_capture_retention_preserves_nonfinite_literals_before_pruning() {
        let mut session = RunMatSession::with_options(false, false).expect("session init");
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(
                "native-nonfinite-capture@1",
                "nonfinite_retention.m",
                r#"
helper(1)
function output = helper(input)
  output = input + Inf;
  function output = unused_nested(input)
    output = input + NaN;
  end
end
"#,
            ),
            None,
        ))
        .expect("compile nonfinite capture fixture");
        let retained = unit
            .reachability_analysis()
            .1
            .into_iter()
            .map(|function| runmat_types::ProgramFunctionId(u32::try_from(function.0).unwrap()))
            .collect::<BTreeSet<_>>();
        let input = unit
            .prepare_native_compilation()
            .expect("capture all functions");
        let complete = runmat_vm::decode_interpreter_program_v2(&input.program_capture)
            .expect("decode before pruning unreachable nonfinite function");
        let constants = |registry: &runmat_vm::FunctionRegistry| {
            registry
                .functions
                .values()
                .flat_map(|function| &function.instructions)
                .filter_map(|instruction| match instruction {
                    runmat_vm::Instr::LoadConst(value) => Some(value.to_bits()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let complete_constants = constants(&complete);
        assert!(complete_constants.contains(&f64::INFINITY.to_bits()));
        assert!(complete_constants
            .iter()
            .any(|bits| f64::from_bits(*bits).is_nan()));
        let input = input
            .retain_functions(&retained)
            .expect("retain reachable functions");
        let registry = runmat_vm::decode_interpreter_program_v2(&input.program_capture)
            .expect("decode retained native capture");
        assert!(registry.names.contains_key("helper"));
        assert!(!registry.names.contains_key("unused_nested"));
        let retained_constants = constants(&registry);
        assert!(retained_constants.contains(&f64::INFINITY.to_bits()));
        assert!(!retained_constants
            .iter()
            .any(|bits| f64::from_bits(*bits).is_nan()));
        assert_eq!(
            runmat_vm::encode_interpreter_program_v2(&registry).unwrap(),
            input.program_capture
        );
    }

    #[test]
    fn native_retention_prunes_code_analysis_contracts_and_registry_together() {
        let mut session = RunMatSession::with_options(false, false).expect("session init");
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(
                "native-retention-test@1",
                "retention.m",
                r#"
helper(11)
function output = helper(input)
  output = abs(input) + 1;
  function output = unused_nested(input)
    output = input * 100;
  end
end
"#,
            ),
            None,
        ))
        .expect("compile retention fixture");
        let report = unit.reachability_report();
        assert!(report
            .nodes
            .iter()
            .all(|node| node.symbol != "unused_nested"));
        let retained = unit
            .reachability_analysis()
            .1
            .into_iter()
            .map(|function| {
                u32::try_from(function.0)
                    .map(runmat_types::ProgramFunctionId)
                    .expect("portable function identity")
            })
            .collect::<BTreeSet<_>>();
        let input = unit
            .prepare_native_compilation()
            .expect("prepare native input")
            .retain_functions(&retained)
            .expect("retain reachable functions");

        assert_eq!(input.mir.bodies.len(), 2);
        assert_eq!(input.mir.functions.len(), 2);
        assert!(input
            .analysis
            .functions
            .iter()
            .all(|function| retained.contains(&function.function)));
        assert!(input
            .analysis
            .program_points
            .iter()
            .all(|point| retained.contains(&point.point.function)));
        assert!(input
            .analysis
            .regions
            .iter()
            .all(|region| retained.contains(&region.contract.id.function)));
        assert!(input
            .manifest
            .regions
            .iter()
            .all(|region| retained.contains(&region.id.function)));
        let registry: runmat_vm::FunctionRegistry =
            runmat_vm::decode_interpreter_program_v2(&input.program_capture)
                .expect("decode retained registry");
        assert_eq!(registry.functions.len(), 1);
        assert!(registry.names.contains_key("helper"));
        assert!(!registry.names.contains_key("unused_nested"));
        let assembly = input
            .lower(runmat_native_codegen::NativeTarget::current())
            .expect("lower retained Native IR");
        assert_eq!(assembly.functions.len(), 2);
        assert!(assembly
            .requirements
            .regions
            .iter()
            .all(|region| retained.contains(&region.id.function)));
        let object_data = input
            .aot_object_data(
                &assembly,
                AotObjectDataOptions {
                    runtime_binding_mode:
                        runmat_native_codegen::aot::AotRuntimeBindingMode::Dynamic,
                    builtin_bindings: Vec::new(),
                    foreign_policy: None,
                    native_interfaces: br#"{"schema_version":1,"interfaces":[]}"#.to_vec(),
                    mex_artifacts: br#"{"schema_version":1,"artifacts":[]}"#.to_vec(),
                    java_artifacts: br#"{"schema_version":1,"environment":null,"artifacts":[]}"#
                        .to_vec(),
                    python_artifacts: br#"{"schema_version":1,"artifacts":[]}"#.to_vec(),
                },
            )
            .expect("build retained AOT object data");
        let program_bytes = &object_data
            .iter()
            .find(|data| data.symbol == runmat_native_codegen::aot::AOT_PROGRAM_SYMBOL)
            .expect("AOT program manifest")
            .bytes;
        let program =
            runmat_native_codegen::aot::AotProgramManifest::from_canonical_bytes(program_bytes)
                .expect("decode AOT program manifest");
        assert_eq!(program.functions.len(), 2);
        assert_eq!(
            program.source_map.entries.len(),
            unit.source_map().entries().len()
        );
        for (embedded, original) in program
            .source_map
            .entries
            .iter()
            .zip(unit.source_map().entries())
        {
            assert_eq!(embedded.source_id, original.source_id);
            assert_eq!(embedded.relative_path, original.relative_path);
            assert_eq!(embedded.text, original.text);
            assert_eq!(embedded.content_revision, original.content_revision);
        }
        let mut forged = program.clone();
        forged.source_map.entries[0]
            .text
            .push_str("\nforged = true;");
        assert!(forged.canonical_bytes().is_err());
        let mut missing_source = program.clone();
        missing_source.functions[0].source = runmat_types::ProgramSourceId(u32::MAX);
        assert!(missing_source.canonical_bytes().is_err());
        assert!(program
            .functions
            .iter()
            .all(|function| function.name != "unused_nested"));
        assert_eq!(
            program.native_ir_digest,
            runmat_execution::Digest::sha256(
                object_data
                    .iter()
                    .find(|data| data.symbol == runmat_native_codegen::aot::AOT_NATIVE_IR_SYMBOL)
                    .expect("AOT Native IR")
                    .bytes
                    .as_slice()
            )
        );
        assert!(!String::from_utf8_lossy(program_bytes).contains("instructions"));
    }
}
