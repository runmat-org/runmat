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
        let resume_points =
            serde_json::to_vec(&self.interpreter_resume_points).map_err(|error| {
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
                self.program_capture.clone(),
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
