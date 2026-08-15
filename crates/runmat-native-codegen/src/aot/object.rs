use cranelift_codegen::settings::Configurable;
use cranelift_module::{default_libcall_names, DataDescription, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use runmat_types::ProgramFunctionId;

use crate::{NativeAssembly, NativeCodegenError, NativeCodegenResult};

use super::{
    NativeObjectData, NativeObjectDataDescriptor, NativeObjectFormat, NativeObjectFunction,
    NativeObjectManifest, NativeOptimization, RelocatableNativeObject,
    NATIVE_OBJECT_SCHEMA_VERSION,
};

pub fn emit_relocatable_object(
    assembly: &NativeAssembly,
    optimization: NativeOptimization,
) -> NativeCodegenResult<RelocatableNativeObject> {
    emit_relocatable_object_with_data(assembly, optimization, Vec::new())
}

pub fn emit_relocatable_object_with_data(
    assembly: &NativeAssembly,
    optimization: NativeOptimization,
    mut embedded_data: Vec<NativeObjectData>,
) -> NativeCodegenResult<RelocatableNativeObject> {
    assembly.verify()?;
    assembly.target.validate()?;
    let entrypoint = assembly.executable_identity.entrypoint_function;
    if !assembly
        .functions
        .iter()
        .any(|function| function.id == entrypoint)
    {
        return Err(NativeCodegenError::new(
            "native.object.entrypoint",
            "native object entrypoint is absent from the verified assembly",
        ));
    }

    let mut flags = cranelift_codegen::settings::builder();
    flags
        .set("use_colocated_libcalls", "false")
        .map_err(module_error)?;
    flags.set("is_pic", "true").map_err(module_error)?;
    flags.set("enable_verifier", "true").map_err(module_error)?;
    flags
        .set("opt_level", optimization.cranelift_name())
        .map_err(module_error)?;
    let isa = cranelift_native::builder()
        .map_err(module_error)?
        .finish(cranelift_codegen::settings::Flags::new(flags))
        .map_err(module_error)?;
    let builder = ObjectBuilder::new(isa, "runmat-user-program", default_libcall_names())
        .map_err(module_error)?;
    let mut module = ObjectModule::new(builder);
    let mut functions = Vec::with_capacity(assembly.functions.len());
    let mut linked_functions = Vec::with_capacity(assembly.functions.len());

    for function in &assembly.functions {
        let compiled = crate::cranelift::lower_function(function, &assembly.target)?;
        let symbol = function_symbol(function.id, entrypoint);
        let id = module
            .declare_function(&symbol, Linkage::Export, &compiled.ir.signature)
            .map_err(module_error)?;
        let mut context = module.make_context();
        context.func = compiled.ir;
        module
            .define_function(id, &mut context)
            .map_err(module_error)?;
        module.clear_context(&mut context);
        linked_functions.push((function.id, id));
        functions.push(NativeObjectFunction {
            function: function.id,
            symbol,
        });
    }

    embedded_data.sort_by(|left, right| left.symbol.cmp(&right.symbol));
    if embedded_data.len() > 16
        || embedded_data
            .iter()
            .try_fold(0_usize, |total, item| total.checked_add(item.bytes.len()))
            .is_none_or(|total| total > 512 * 1024 * 1024)
    {
        return Err(NativeCodegenError::new(
            "native.object.data",
            "native object data exceeds its count or aggregate size bound",
        ));
    }
    if embedded_data
        .windows(2)
        .any(|pair| pair[0].symbol == pair[1].symbol)
    {
        return Err(NativeCodegenError::new(
            "native.object.data_identity",
            "native object data symbols must be unique",
        ));
    }
    let mut data = Vec::with_capacity(embedded_data.len());
    let mut linked_data = std::collections::BTreeMap::new();
    for item in embedded_data {
        if !super::product::valid_data_symbol(&item.symbol)
            || item.bytes.is_empty()
            || !matches!(item.alignment, 1 | 2 | 4 | 8 | 16)
        {
            return Err(NativeCodegenError::new(
                "native.object.data",
                "native object data is invalid or exceeds its bound",
            ));
        }
        let id = module
            .declare_data(&item.symbol, Linkage::Export, false, false)
            .map_err(module_error)?;
        let mut description = DataDescription::new();
        description.define(item.bytes.clone().into_boxed_slice());
        description.set_align(item.alignment);
        module.define_data(id, &description).map_err(module_error)?;
        linked_data.insert(
            item.symbol.clone(),
            (
                id,
                u64::try_from(item.bytes.len()).map_err(|_| {
                    NativeCodegenError::new(
                        "native.object.data",
                        "native object data size exceeds the portable manifest limit",
                    )
                })?,
            ),
        );
        data.push(NativeObjectDataDescriptor {
            symbol: item.symbol,
            digest: runmat_execution::Digest::sha256(&item.bytes),
            bytes: u64::try_from(item.bytes.len()).map_err(|_| {
                NativeCodegenError::new(
                    "native.object.data",
                    "native object data size exceeds the portable manifest limit",
                )
            })?,
            alignment: item.alignment,
        });
    }
    if !linked_data.is_empty() {
        let resolver = super::launcher::define_resolver(&mut module, &linked_functions)?;
        super::launcher::define(&mut module, resolver, &linked_data)?;
    }

    let product = module.finish();
    let bytes = product.emit().map_err(module_error)?;
    let object_bytes = u64::try_from(bytes.len()).map_err(|_| {
        NativeCodegenError::new(
            "native.object.size",
            "native object size exceeds the portable manifest limit",
        )
    })?;
    let manifest = NativeObjectManifest {
        schema_version: NATIVE_OBJECT_SCHEMA_VERSION,
        target: assembly.target.clone(),
        object_format: NativeObjectFormat::for_target(&assembly.target)?,
        executable_cache_key: assembly.executable_cache_key,
        native_cache_key: assembly.native_cache_key,
        runtime_fingerprint: *assembly.program.runtime_fingerprint(),
        catalog_fingerprint: *assembly.program.catalog_fingerprint(),
        optimization,
        object_digest: runmat_execution::Digest::sha256(&bytes),
        object_bytes,
        entrypoint,
        functions,
        data,
    };
    let object = RelocatableNativeObject { manifest, bytes };
    object.validate()?;
    Ok(object)
}

pub(super) fn function_symbol(
    function: ProgramFunctionId,
    entrypoint: ProgramFunctionId,
) -> String {
    if function == entrypoint {
        super::AOT_ENTRY_SYMBOL.to_string()
    } else {
        format!("runmat_native_f{}", function.0)
    }
}

fn module_error(error: impl std::fmt::Display) -> NativeCodegenError {
    NativeCodegenError::new("native.object.emit", error.to_string())
}
