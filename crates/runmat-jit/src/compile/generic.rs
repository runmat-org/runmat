use std::collections::BTreeMap;

use cranelift_codegen::settings::Configurable;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

use crate::{CompiledExecutable, JitError, JitResult};

pub struct GenericCompiler;

impl GenericCompiler {
    pub fn compile(
        assembly: &runmat_native_codegen::NativeAssembly,
    ) -> JitResult<CompiledExecutable> {
        assembly.verify()?;
        let mut flags = cranelift_codegen::settings::builder();
        flags
            .set("use_colocated_libcalls", "false")
            .map_err(module_error)?;
        flags.set("is_pic", "false").map_err(module_error)?;
        flags.set("enable_verifier", "true").map_err(module_error)?;
        let isa = cranelift_native::builder()
            .map_err(|error| JitError::Module(error.to_string()))?
            .finish(cranelift_codegen::settings::Flags::new(flags))
            .map_err(module_error)?;
        let builder = JITBuilder::with_isa(isa, default_libcall_names());
        let mut module = JITModule::new(builder);
        let mut definitions = Vec::with_capacity(assembly.functions.len());
        for function in &assembly.functions {
            let compiled =
                runmat_native_codegen::cranelift::lower_function(function, &assembly.target)?;
            let symbol = format!("runmat_native_f{}", function.id.0);
            let id = module
                .declare_function(&symbol, Linkage::Export, &compiled.ir.signature)
                .map_err(module_error)?;
            let mut context = module.make_context();
            context.func = compiled.ir;
            module
                .define_function(id, &mut context)
                .map_err(module_error)?;
            definitions.push((function.id, id));
        }
        module.finalize_definitions().map_err(module_error)?;
        let entrypoints = definitions
            .into_iter()
            .map(|(function, id)| (function, module.get_finalized_function(id)))
            .collect::<BTreeMap<_, _>>();
        Ok(CompiledExecutable {
            _module: module,
            entrypoints,
        })
    }
}

fn module_error(error: impl std::fmt::Display) -> JitError {
    JitError::Module(error.to_string())
}
