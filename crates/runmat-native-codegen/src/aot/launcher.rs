use std::collections::BTreeMap;

use cranelift::prelude::{
    types, AbiParam, FunctionBuilder, FunctionBuilderContext, InstBuilder, IntCC,
};
use cranelift_codegen::ir::UserFuncName;
use cranelift_module::{DataId, FuncId, Linkage, Module};
use cranelift_object::ObjectModule;

use crate::{NativeCodegenError, NativeCodegenResult};
use runmat_execution_artifact::target_test::NativePairedProductRole;

use super::AotBuiltinBinding;

pub(super) struct LauncherEntryContract<'a> {
    pub resolvers: LauncherResolvers,
    pub services: LauncherServiceImports,
    pub data: &'a BTreeMap<String, (DataId, u64)>,
    pub paired_role: Option<NativePairedProductRole>,
}

pub(super) struct LauncherResolvers {
    pub functions: FuncId,
    pub builtins: FuncId,
}

pub(super) struct LauncherServiceImports {
    pub source_evaluation: bool,
    pub network: bool,
    pub user_interface: bool,
    pub host_callback: bool,
    pub hosted_dynamic: bool,
}

struct ExternalFactories<T> {
    user_interface: T,
    host_callback: T,
    hosted_dynamic: T,
}

pub(super) fn define_resolver(
    module: &mut ObjectModule,
    functions: &[(runmat_types::FunctionId, FuncId)],
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
        let function_index = u32::try_from(function.0).map_err(|_| {
            NativeCodegenError::new(
                "native.object.function_resolver",
                "compiled function coordinate exceeds the AOT resolver ABI",
            )
            .at_function(function)
        })?;
        let found = builder.create_block();
        let next = builder.create_block();
        let matches = builder
            .ins()
            .icmp_imm(IntCC::Equal, requested, i64::from(function_index));
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
    contract: LauncherEntryContract<'_>,
) -> NativeCodegenResult<()> {
    let LauncherEntryContract {
        resolvers,
        services,
        data,
        paired_role,
    } = contract;
    let LauncherResolvers {
        functions: resolver,
        builtins: builtin_resolver,
    } = resolvers;
    let LauncherServiceImports {
        source_evaluation: source_service,
        network: network_service,
        user_interface,
        host_callback,
        hosted_dynamic,
    } = services;
    let native_ir = required_data(data, super::AOT_NATIVE_IR_SYMBOL)?;
    let program = required_data(data, super::AOT_PROGRAM_SYMBOL)?;
    let resume_points = required_data(data, super::AOT_RESUME_POINTS_SYMBOL)?;
    let native_interfaces = required_data(data, super::AOT_NATIVE_INTERFACES_SYMBOL)?;
    let mex_artifacts = required_data(data, super::AOT_MEX_ARTIFACTS_SYMBOL)?;
    let java_artifacts = required_data(data, super::AOT_JAVA_ARTIFACTS_SYMBOL)?;
    let python_artifacts = required_data(data, super::AOT_PYTHON_ARTIFACTS_SYMBOL)?;
    let paired_header = if paired_role.is_some() {
        Some(required_data(data, super::AOT_PAIRED_PRODUCT_SYMBOL)?)
    } else {
        None
    };
    let pointer = module.isa().pointer_type();
    let source_service_factory = if source_service {
        let mut signature = module.make_signature();
        signature.params.push(AbiParam::new(pointer));
        signature.returns.push(AbiParam::new(pointer));
        Some(
            module
                .declare_function(
                    super::AOT_SOURCE_SERVICE_FACTORY_SYMBOL,
                    Linkage::Import,
                    &signature,
                )
                .map_err(module_error)?,
        )
    } else {
        None
    };
    let network_service_factory = if network_service {
        let mut signature = module.make_signature();
        signature.params.push(AbiParam::new(pointer));
        signature.returns.push(AbiParam::new(pointer));
        Some(
            module
                .declare_function(
                    super::AOT_NETWORK_SERVICE_FACTORY_SYMBOL,
                    Linkage::Import,
                    &signature,
                )
                .map_err(module_error)?,
        )
    } else {
        None
    };
    let external_factories = ExternalFactories {
        user_interface: declare_external_factory(
            module,
            pointer,
            super::AOT_UI_SERVICE_FACTORY_SYMBOL,
            user_interface,
        )?,
        host_callback: declare_external_factory(
            module,
            pointer,
            super::AOT_HOST_CALLBACK_FACTORY_SYMBOL,
            host_callback,
        )?,
        hosted_dynamic: declare_external_factory(
            module,
            pointer,
            super::AOT_HOSTED_DYNAMIC_FACTORY_SYMBOL,
            hosted_dynamic,
        )?,
    };

    let mut runtime_signature = module.make_signature();
    runtime_signature.params.extend([
        AbiParam::new(types::I32),
        AbiParam::new(pointer),
        AbiParam::new(pointer),
        AbiParam::new(pointer),
        AbiParam::new(pointer),
        AbiParam::new(pointer),
        AbiParam::new(pointer),
        AbiParam::new(pointer),
        AbiParam::new(pointer),
        AbiParam::new(pointer),
        AbiParam::new(types::I64),
        AbiParam::new(pointer),
        AbiParam::new(types::I64),
        AbiParam::new(pointer),
        AbiParam::new(types::I64),
        AbiParam::new(pointer),
        AbiParam::new(types::I64),
        AbiParam::new(pointer),
        AbiParam::new(types::I64),
        AbiParam::new(pointer),
        AbiParam::new(types::I64),
        AbiParam::new(pointer),
        AbiParam::new(types::I64),
    ]);
    if paired_role == Some(NativePairedProductRole::Test) {
        runtime_signature
            .params
            .extend([AbiParam::new(pointer), AbiParam::new(types::I64)]);
    }
    runtime_signature.returns.push(AbiParam::new(types::I32));
    let runtime = module
        .declare_function(
            if paired_role == Some(NativePairedProductRole::Test) {
                super::AOT_RUNTIME_TEST_MAIN_SYMBOL
            } else {
                super::AOT_RUNTIME_MAIN_SYMBOL
            },
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
    let source_service_factory = source_service_factory
        .map(|function| module.declare_func_in_func(function, &mut context.func));
    let network_service_factory = network_service_factory
        .map(|function| module.declare_func_in_func(function, &mut context.func));
    let external_factories = ExternalFactories {
        user_interface: external_factories
            .user_interface
            .map(|function| module.declare_func_in_func(function, &mut context.func)),
        host_callback: external_factories
            .host_callback
            .map(|function| module.declare_func_in_func(function, &mut context.func)),
        hosted_dynamic: external_factories
            .hosted_dynamic
            .map(|function| module.declare_func_in_func(function, &mut context.func)),
    };
    let native_ir_data = module.declare_data_in_func(native_ir.0, &mut context.func);
    let program_data = module.declare_data_in_func(program.0, &mut context.func);
    let resume_data = module.declare_data_in_func(resume_points.0, &mut context.func);
    let native_interfaces_data =
        module.declare_data_in_func(native_interfaces.0, &mut context.func);
    let mex_artifacts_data = module.declare_data_in_func(mex_artifacts.0, &mut context.func);
    let java_artifacts_data = module.declare_data_in_func(java_artifacts.0, &mut context.func);
    let python_artifacts_data = module.declare_data_in_func(python_artifacts.0, &mut context.func);
    let paired_header_data =
        paired_header.map(|header| module.declare_data_in_func(header.0, &mut context.func));

    let mut builder_context = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut context.func, &mut builder_context);
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    let argc = builder.block_params(entry)[0];
    let argv = builder.block_params(entry)[1];
    let resolver = builder.ins().func_addr(pointer, resolver);
    let builtin_resolver = builder.ins().func_addr(pointer, builtin_resolver);
    let source_service_factory = match source_service_factory {
        Some(factory) => builder.ins().func_addr(pointer, factory),
        None => builder.ins().iconst(pointer, 0),
    };
    let network_service_factory = match network_service_factory {
        Some(factory) => builder.ins().func_addr(pointer, factory),
        None => builder.ins().iconst(pointer, 0),
    };
    let external_factories = ExternalFactories {
        user_interface: optional_factory_address(
            &mut builder,
            pointer,
            external_factories.user_interface,
        ),
        host_callback: optional_factory_address(
            &mut builder,
            pointer,
            external_factories.host_callback,
        ),
        hosted_dynamic: optional_factory_address(
            &mut builder,
            pointer,
            external_factories.hosted_dynamic,
        ),
    };
    let native_ir_ptr = builder.ins().symbol_value(pointer, native_ir_data);
    let program_ptr = builder.ins().symbol_value(pointer, program_data);
    let resume_ptr = builder.ins().symbol_value(pointer, resume_data);
    let native_interfaces_ptr = builder.ins().symbol_value(pointer, native_interfaces_data);
    let mex_artifacts_ptr = builder.ins().symbol_value(pointer, mex_artifacts_data);
    let java_artifacts_ptr = builder.ins().symbol_value(pointer, java_artifacts_data);
    let python_artifacts_ptr = builder.ins().symbol_value(pointer, python_artifacts_data);
    let native_ir_len = length_constant(&mut builder, native_ir.1)?;
    let program_len = length_constant(&mut builder, program.1)?;
    let resume_len = length_constant(&mut builder, resume_points.1)?;
    let native_interfaces_len = length_constant(&mut builder, native_interfaces.1)?;
    let mex_artifacts_len = length_constant(&mut builder, mex_artifacts.1)?;
    let java_artifacts_len = length_constant(&mut builder, java_artifacts.1)?;
    let python_artifacts_len = length_constant(&mut builder, python_artifacts.1)?;
    let mut arguments = vec![
        argc,
        argv,
        resolver,
        builtin_resolver,
        source_service_factory,
        network_service_factory,
        external_factories.user_interface,
        external_factories.host_callback,
        external_factories.hosted_dynamic,
        native_ir_ptr,
        native_ir_len,
        program_ptr,
        program_len,
        resume_ptr,
        resume_len,
        native_interfaces_ptr,
        native_interfaces_len,
        mex_artifacts_ptr,
        mex_artifacts_len,
        java_artifacts_ptr,
        java_artifacts_len,
        python_artifacts_ptr,
        python_artifacts_len,
    ];
    if paired_role == Some(NativePairedProductRole::Test) {
        let header = paired_header.expect("test role has header");
        let header_ptr = builder
            .ins()
            .symbol_value(pointer, paired_header_data.expect("test role has data"));
        let header_len = length_constant(&mut builder, header.1)?;
        arguments.extend([header_ptr, header_len]);
    }
    let call = builder.ins().call(runtime, &arguments);
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

fn declare_external_factory(
    module: &mut ObjectModule,
    pointer: cranelift_codegen::ir::Type,
    symbol: &str,
    required: bool,
) -> NativeCodegenResult<Option<FuncId>> {
    if !required {
        return Ok(None);
    }
    let mut signature = module.make_signature();
    signature
        .params
        .extend((0..3).map(|_| AbiParam::new(pointer)));
    signature.returns.push(AbiParam::new(pointer));
    module
        .declare_function(symbol, Linkage::Import, &signature)
        .map(Some)
        .map_err(module_error)
}

fn optional_factory_address(
    builder: &mut FunctionBuilder<'_>,
    pointer: cranelift_codegen::ir::Type,
    factory: Option<cranelift_codegen::ir::FuncRef>,
) -> cranelift::prelude::Value {
    match factory {
        Some(factory) => builder.ins().func_addr(pointer, factory),
        None => builder.ins().iconst(pointer, 0),
    }
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
