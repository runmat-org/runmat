use std::collections::BTreeMap;

use super::ExecutableUnit;

/// Owned canonical frontend product used by both background JIT compilation
/// and ahead-of-time object emission.
///
/// Keeping this preparation in Core prevents either native product from
/// rebuilding MIR, analysis facts, executable identity, or binding names.
pub struct NativeCompilationInput {
    mir: runmat_mir::MirAssembly,
    analysis: runmat_mir::analysis::AnalysisStore,
    manifest: runmat_execution::ExecutableUnitManifest,
    binding_names: BTreeMap<runmat_types::BindingId, String>,
    entrypoint: runmat_types::ProgramFunctionId,
    program_capture: Vec<u8>,
    interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
}

impl NativeCompilationInput {
    pub fn entrypoint(&self) -> runmat_types::ProgramFunctionId {
        self.entrypoint
    }

    pub fn lower(
        &self,
        target: runmat_native_codegen::NativeTarget,
    ) -> Result<runmat_native_codegen::NativeAssembly, runmat_native_codegen::NativeCodegenError>
    {
        runmat_native_codegen::lower_executable(runmat_native_codegen::NativeLoweringInput {
            mir: &self.mir,
            analysis: &self.analysis,
            manifest: &self.manifest,
            binding_names: Some(&self.binding_names),
            target,
        })
    }

    pub fn program_capture(&self) -> &[u8] {
        &self.program_capture
    }

    pub fn interpreter_resume_points(&self) -> &BTreeMap<runmat_types::ProgramPointId, u64> {
        &self.interpreter_resume_points
    }

    pub fn retain_functions(
        mut self,
        retained: &std::collections::BTreeSet<runmat_types::ProgramFunctionId>,
    ) -> Result<Self, runmat_runtime::RuntimeError> {
        if !retained.contains(&self.entrypoint) {
            return Err(native_product_error(
                "native retention set omits the executable entrypoint".into(),
            ));
        }
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
            serde_json::from_slice(&self.program_capture).map_err(|error| {
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
        self.program_capture = serde_json::to_vec(&registry).map_err(|error| {
            native_product_error(format!("failed to encode retained native program: {error}"))
        })?;
        Ok(self)
    }

    pub fn aot_object_data(
        &self,
        assembly: &runmat_native_codegen::NativeAssembly,
    ) -> Result<
        Vec<runmat_native_codegen::aot::NativeObjectData>,
        runmat_native_codegen::NativeCodegenError,
    > {
        if assembly.executable_identity != self.manifest.identity {
            return Err(runmat_native_codegen::NativeCodegenError::new(
                "native.object.executable_identity",
                "AOT embedded data does not belong to the supplied native assembly",
            ));
        }
        let native_ir = serde_json::to_vec(assembly).map_err(|error| {
            runmat_native_codegen::NativeCodegenError::new(
                "native.object.native_ir",
                format!("failed to encode embedded Native IR: {error}"),
            )
        })?;
        let program = runmat_native_codegen::aot::AotProgramManifest::from_assembly(
            assembly,
            runmat_execution::Digest::sha256(&native_ir),
        )?
        .canonical_bytes()?;
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
        let mut data = Vec::with_capacity(3);
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
        let envelope = self
            .portable_envelope_for(preferred_function)
            .map_err(native_product_error)?;
        let program_capture = serde_json::to_vec(self.functions()).map_err(|error| {
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
        Ok(NativeCompilationInput {
            mir: self.mir().clone(),
            analysis: self.analysis().clone(),
            entrypoint: envelope.manifest.identity.entrypoint_function,
            manifest: envelope.manifest,
            binding_names: self.binding_names(),
            program_capture,
            interpreter_resume_points,
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

    use crate::{ExecutableSource, RunMatSession};

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
        let retained = report
            .retained_function_ids()
            .map(|function| {
                u32::try_from(function)
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
            serde_json::from_slice(&input.program_capture).expect("decode retained registry");
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
            .aot_object_data(&assembly)
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
