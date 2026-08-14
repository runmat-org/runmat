use std::collections::BTreeMap;
use std::rc::Rc;

use runmat_types::ProgramFunctionId;

use crate::ExecutableUnit;

pub(super) struct CompiledGenericUnit {
    pub executor: Rc<runmat_jit::GenericExecutor>,
    pub entrypoint: ProgramFunctionId,
    #[cfg(test)]
    pub safepoints: BTreeMap<ProgramFunctionId, Vec<runmat_native_codegen::NativeSafepointId>>,
}

pub(super) fn compile(
    unit: &ExecutableUnit,
    preferred_function: Option<&str>,
) -> Result<CompiledGenericUnit, runmat_runtime::RuntimeError> {
    let envelope = unit
        .portable_envelope_for(preferred_function)
        .map_err(|error| super::error::stage("NativeProduct", error))?;
    let binding_names = unit.binding_names();
    let assembly =
        runmat_native_codegen::lower_executable(runmat_native_codegen::NativeLoweringInput {
            mir: unit.mir(),
            analysis: unit.analysis(),
            manifest: &envelope.manifest,
            binding_names: Some(&binding_names),
            target: runmat_native_codegen::NativeTarget::current(),
        })
        .map_err(|error| super::error::stage("NativeLowering", error))?;
    #[cfg(test)]
    let safepoints = assembly
        .functions
        .iter()
        .map(|function| {
            let points = function
                .blocks
                .iter()
                .flat_map(|block| {
                    block
                        .instructions
                        .iter()
                        .filter_map(|instruction| instruction.safepoint)
                        .chain(block.terminator.safepoint)
                })
                .collect();
            (function.id, points)
        })
        .collect();
    let program_capture = serde_json::to_vec(unit.functions()).map_err(|error| {
        super::error::stage(
            "NativeProduct",
            format!("failed to capture native async program: {error}"),
        )
    })?;
    let mut interpreter_resume_points = unit
        .vm_layout()
        .functions
        .values()
        .flat_map(|function| function.resume_points.iter())
        .map(|(point, pc)| {
            u64::try_from(*pc).map(|pc| (*point, pc)).map_err(|_| {
                super::error::stage("NativeProduct", "bytecode resume PC exceeds native ABI")
            })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    for function in unit.functions().functions.values() {
        for (point, pc) in &function.resume_points {
            let pc = u64::try_from(*pc).map_err(|_| {
                super::error::stage("NativeProduct", "bytecode resume PC exceeds native ABI")
            })?;
            interpreter_resume_points.insert(*point, pc);
        }
    }
    let executor = runmat_jit::GenericExecutor::compile_with_resume_points(
        assembly,
        Some(program_capture),
        interpreter_resume_points,
    )
    .map_err(super::error::from_jit_error)?;
    Ok(CompiledGenericUnit {
        executor: Rc::new(executor),
        entrypoint: envelope.manifest.identity.entrypoint_function,
        #[cfg(test)]
        safepoints,
    })
}
