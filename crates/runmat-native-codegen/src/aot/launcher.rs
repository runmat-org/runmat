use std::collections::BTreeMap;

use cranelift::prelude::{
    types, AbiParam, FunctionBuilder, FunctionBuilderContext, InstBuilder, IntCC,
};
use cranelift_codegen::ir::UserFuncName;
use cranelift_module::{DataId, FuncId, Linkage, Module};
use cranelift_object::ObjectModule;

use crate::{NativeCodegenError, NativeCodegenResult};

use super::AotBuiltinBinding;

pub(super) fn define_resolver(
    module: &mut ObjectModule,
    functions: &[(runmat_types::ProgramFunctionId, FuncId)],
) -> NativeCodegenResult<FuncId> {
    if functions.is_empty()
        || functions.len() > 65_536
        || functions.windows(2).any(|pair| pair[0].0 >= pair[1].0)
    {
        return Err(NativeCodegenError::new(
            "native.object.function_resolver",
            "launcher functions must be non-empty, bounded, sorted, and unique",
        ));
    }
    let pointer = module.isa().pointer_type();
    let mut signature = module.make_signature();
    signature.params.push(AbiParam::new(types::I32));
    signature.returns.push(AbiParam::new(pointer));
    let resolver = module
        .declare_function("runmat_aot_resolve_function", Linkage::Local, &signature)
        .map_err(module_error)?;
    let mut context = module.make_context();
    context.func.signature = signature;
    context.func.name = UserFuncName::user(1, 1);
    let linked = functions
        .iter()
        .map(|(function, linked)| {
            (
                *function,
                module.declare_func_in_func(*linked, &mut context.func),
            )
        })
        .collect::<Vec<_>>();
    let mut builder_context = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut context.func, &mut builder_context);
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    let requested = builder.block_params(entry)[0];
    for (function, linked) in linked {
        let found = builder.create_block();
        let next = builder.create_block();
        let matches = builder
            .ins()
            .icmp_imm(IntCC::Equal, requested, i64::from(function.0));
        builder.ins().brif(matches, found, &[], next, &[]);
        builder.switch_to_block(found);
        let address = builder.ins().func_addr(pointer, linked);
        builder.ins().return_(&[address]);
        builder.seal_block(found);
        builder.switch_to_block(next);
        builder.seal_block(next);
    }
    let missing = builder.ins().iconst(pointer, 0);
    builder.ins().return_(&[missing]);
    builder.seal_block(entry);
    builder.finalize();
    module
        .define_function(resolver, &mut context)
        .map_err(module_error)?;
    module.clear_context(&mut context);
    Ok(resolver)
}

pub(super) fn define_builtin_resolver(
    module: &mut ObjectModule,
    bindings: &[AotBuiltinBinding],
) -> NativeCodegenResult<FuncId> {
    if bindings.len() > 65_536
        || bindings
            .windows(2)
            .any(|pair| (&pair[0].name, &pair[0].variant) >= (&pair[1].name, &pair[1].variant))
    {
        return Err(NativeCodegenError::new(
            "native.object.builtin_resolver",
            "launcher builtin bindings must be bounded, sorted, and unique",
        ));
    }
    let pointer = module.isa().pointer_type();
    let mut signature = module.make_signature();
    signature.params.push(AbiParam::new(types::I32));
    signature.returns.push(AbiParam::new(pointer));
    let resolver = module
        .declare_function("runmat_aot_resolve_builtin", Linkage::Local, &signature)
        .map_err(module_error)?;
    let mut context = module.make_context();
    context.func.signature = signature;
    context.func.name = UserFuncName::user(1, 2);
    let linked = bindings
        .iter()
        .map(|binding| {
            module
                .declare_data(&binding.native_symbol, Linkage::Import, false, false)
                .map(|data| module.declare_data_in_func(data, &mut context.func))
                .map_err(module_error)
        })
        .collect::<NativeCodegenResult<Vec<_>>>()?;
    let mut builder_context = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut context.func, &mut builder_context);
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    let requested = builder.block_params(entry)[0];
    for (index, linked) in linked.into_iter().enumerate() {
        let found = builder.create_block();
        let next = builder.create_block();
        let index = i64::try_from(index).map_err(|_| {
            NativeCodegenError::new(
                "native.object.builtin_resolver",
                "builtin resolver index exceeds the native ABI",
            )
        })?;
        let matches = builder.ins().icmp_imm(IntCC::Equal, requested, index);
        builder.ins().brif(matches, found, &[], next, &[]);
        builder.switch_to_block(found);
        let address = builder.ins().symbol_value(pointer, linked);
        builder.ins().return_(&[address]);
        builder.seal_block(found);
        builder.switch_to_block(next);
        builder.seal_block(next);
    }
    let missing = builder.ins().iconst(pointer, 0);
    builder.ins().return_(&[missing]);
    builder.seal_block(entry);
    builder.finalize();
    module
        .define_function(resolver, &mut context)
        .map_err(module_error)?;
    module.clear_context(&mut context);
    Ok(resolver)
}

pub(super) fn define(
    module: &mut ObjectModule,
    resolver: FuncId,
    builtin_resolver: FuncId,
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
    let resolver = module.declare_func_in_func(resolver, &mut context.func);
    let builtin_resolver = module.declare_func_in_func(builtin_resolver, &mut context.func);
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
    let resolver = builder.ins().func_addr(pointer, resolver);
    let builtin_resolver = builder.ins().func_addr(pointer, builtin_resolver);
    let native_ir_ptr = builder.ins().symbol_value(pointer, native_ir_data);
    let program_ptr = builder.ins().symbol_value(pointer, program_data);
    let resume_ptr = builder.ins().symbol_value(pointer, resume_data);
    let native_ir_len = length_constant(&mut builder, native_ir.1)?;
    let program_len = length_constant(&mut builder, program.1)?;
    let resume_len = length_constant(&mut builder, resume_points.1)?;
    let call = builder.ins().call(
        runtime,
        &[
            argc,
            argv,
            resolver,
            builtin_resolver,
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

fn length_constant(
    builder: &mut FunctionBuilder<'_>,
    length: u64,
) -> NativeCodegenResult<cranelift::prelude::Value> {
    let length = i64::try_from(length).map_err(|_| {
        NativeCodegenError::new(
            "native.object.launcher_data",
            "launcher data length exceeds the native ABI",
        )
    })?;
    Ok(builder.ins().iconst(types::I64, length))
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
