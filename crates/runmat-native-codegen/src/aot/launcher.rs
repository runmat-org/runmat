use std::collections::BTreeMap;

use cranelift::prelude::{AbiParam, FunctionBuilder, FunctionBuilderContext, InstBuilder};
use cranelift_codegen::ir::{types, UserFuncName};
use cranelift_module::{DataId, FuncId, Linkage, Module};
use cranelift_object::ObjectModule;

use crate::{NativeCodegenError, NativeCodegenResult};

pub(super) fn define(
    module: &mut ObjectModule,
    entrypoint: FuncId,
    data: &BTreeMap<String, (DataId, u64)>,
) -> NativeCodegenResult<()> {
    let native_ir = required_data(data, super::AOT_NATIVE_IR_SYMBOL)?;
    let program = required_data(data, super::AOT_PROGRAM_SYMBOL)?;
    let resume_points = required_data(data, super::AOT_RESUME_POINTS_SYMBOL)?;
    let pointer = module.isa().pointer_type();

    let mut runtime_signature = module.make_signature();
    runtime_signature.params.extend([
        AbiParam::new(types::I32),
        AbiParam::new(pointer),
        AbiParam::new(pointer),
        AbiParam::new(pointer),
        AbiParam::new(types::I64),
        AbiParam::new(pointer),
        AbiParam::new(types::I64),
        AbiParam::new(pointer),
        AbiParam::new(types::I64),
    ]);
    runtime_signature.returns.push(AbiParam::new(types::I32));
    let runtime = module
        .declare_function(
            super::AOT_RUNTIME_MAIN_SYMBOL,
            Linkage::Import,
            &runtime_signature,
        )
        .map_err(module_error)?;

    let mut main_signature = module.make_signature();
    main_signature
        .params
        .extend([AbiParam::new(types::I32), AbiParam::new(pointer)]);
    main_signature.returns.push(AbiParam::new(types::I32));
    let main = module
        .declare_function("main", Linkage::Export, &main_signature)
        .map_err(module_error)?;
    let mut context = module.make_context();
    context.func.signature = main_signature;
    context.func.name = UserFuncName::user(1, 0);
    let runtime = module.declare_func_in_func(runtime, &mut context.func);
    let native_entry = module.declare_func_in_func(entrypoint, &mut context.func);
    let native_ir_data = module.declare_data_in_func(native_ir.0, &mut context.func);
    let program_data = module.declare_data_in_func(program.0, &mut context.func);
    let resume_data = module.declare_data_in_func(resume_points.0, &mut context.func);

    let mut builder_context = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut context.func, &mut builder_context);
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    let argc = builder.block_params(entry)[0];
    let argv = builder.block_params(entry)[1];
    let native_entry = builder.ins().func_addr(pointer, native_entry);
    let native_ir_ptr = builder.ins().symbol_value(pointer, native_ir_data);
    let program_ptr = builder.ins().symbol_value(pointer, program_data);
    let resume_ptr = builder.ins().symbol_value(pointer, resume_data);
    let native_ir_len = builder.ins().iconst(types::I64, native_ir.1 as i64);
    let program_len = builder.ins().iconst(types::I64, program.1 as i64);
    let resume_len = builder.ins().iconst(types::I64, resume_points.1 as i64);
    let call = builder.ins().call(
        runtime,
        &[
            argc,
            argv,
            native_entry,
            native_ir_ptr,
            native_ir_len,
            program_ptr,
            program_len,
            resume_ptr,
            resume_len,
        ],
    );
    let status = builder.inst_results(call)[0];
    builder.ins().return_(&[status]);
    builder.seal_all_blocks();
    builder.finalize();
    module
        .define_function(main, &mut context)
        .map_err(module_error)?;
    module.clear_context(&mut context);
    Ok(())
}

fn required_data<'a>(
    data: &'a BTreeMap<String, (DataId, u64)>,
    symbol: &str,
) -> NativeCodegenResult<&'a (DataId, u64)> {
    data.get(symbol).ok_or_else(|| {
        NativeCodegenError::new(
            "native.object.launcher_data",
            format!("native object is missing required launcher data '{symbol}'"),
        )
    })
}

fn module_error(error: impl std::fmt::Display) -> NativeCodegenError {
    NativeCodegenError::new("native.object.launcher", error.to_string())
}
