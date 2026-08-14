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
        Ok(NativeCompilationInput {
            mir: self.mir().clone(),
            analysis: self.analysis().clone(),
            entrypoint: envelope.manifest.identity.entrypoint_function,
            manifest: envelope.manifest,
            binding_names: self.binding_names(),
        })
    }
}

fn native_product_error(message: String) -> runmat_runtime::RuntimeError {
    runmat_runtime::build_runtime_error(message)
        .with_identifier("RunMat:NativeProduct")
        .build()
}
