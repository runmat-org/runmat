use std::collections::BTreeMap;

use cranelift::prelude::*;

use crate::{NativeCodegenError, NativeCodegenResult, NativeFunction, NativeTarget};
use runmat_runtime::native::NativeSiteOutcomeKind;

use super::abi::AbiTypes;
use super::site::{HostValues, SiteCallResult, SiteSlots};

pub struct CraneliftFunction {
    pub function: runmat_types::ProgramFunctionId,
    pub ir: cranelift_codegen::ir::Function,
}

pub fn lower_function(
    function: &NativeFunction,
    target: &NativeTarget,
) -> NativeCodegenResult<CraneliftFunction> {
    target.validate()?;
    if target.pointer_width != usize::BITS as u16 {
        return Err(NativeCodegenError::new(
            "native.cranelift.pointer_width",
            "Cranelift lowering requires the current target pointer width",
        ));
    }
    let abi = AbiTypes::current();
    let mut ir = cranelift_codegen::ir::Function::with_name_signature(
        cranelift_codegen::ir::UserFuncName::user(0, function.id.0),
        abi.entry_signature.clone(),
    );
    let mut builder_context = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ir, &mut builder_context);

    let entry = builder.create_block();
    let invalid = builder.create_block();
    let host_check = builder.create_block();
    let failure = builder.create_block();
    let return_ok = builder.create_block();
    builder.append_block_params_for_function_params(entry);

    let blocks = function
        .blocks
        .iter()
        .map(|block| (block.id, builder.create_block()))
        .collect::<BTreeMap<_, _>>();
    let _entry_target = *blocks.get(&function.entry).ok_or_else(|| {
        NativeCodegenError::new(
            "native.cranelift.entry",
            "Native IR entry block is unavailable during Cranelift lowering",
        )
    })?;

    builder.switch_to_block(entry);
    let call = builder.block_params(entry)[0];
    let exit = builder.block_params(entry)[1];
    let call_missing = builder.ins().icmp_imm(IntCC::Equal, call, 0);
    let exit_missing = builder.ins().icmp_imm(IntCC::Equal, exit, 0);
    let invalid_arguments = builder.ins().bor(call_missing, exit_missing);
    builder
        .ins()
        .brif(invalid_arguments, invalid, &[], host_check, &[]);

    builder.switch_to_block(invalid);
    return_status(
        &mut builder,
        runmat_runtime::native::NativeHostStatus::INVALID_ARGUMENT.0,
    );

    builder.switch_to_block(host_check);
    let host = builder.ins().load(
        abi.pointer,
        MemFlags::new(),
        call,
        super::abi::call_host_offset(),
    );
    let host_missing = builder.ins().icmp_imm(IntCC::Equal, host, 0);
    let host_ready = builder.create_block();
    builder
        .ins()
        .brif(host_missing, invalid, &[], host_ready, &[]);
    builder.switch_to_block(host_ready);
    let context = builder.ins().load(
        abi.pointer,
        MemFlags::new(),
        host,
        super::abi::host_context_offset(),
    );
    let execute_site = builder.ins().load(
        abi.pointer,
        MemFlags::new(),
        host,
        super::abi::host_execute_site_offset(),
    );
    let execute_missing = builder.ins().icmp_imm(IntCC::Equal, execute_site, 0);
    let resume_dispatch = builder.create_block();
    builder
        .ins()
        .brif(execute_missing, invalid, &[], resume_dispatch, &[]);
    builder.switch_to_block(resume_dispatch);
    dispatch_resume_block(&mut builder, &abi, call, &blocks, invalid);

    builder.switch_to_block(failure);
    return_status(
        &mut builder,
        runmat_runtime::native::NativeHostStatus::HOST_FAILURE.0,
    );
    builder.switch_to_block(return_ok);
    return_status(&mut builder, runmat_runtime::native::NativeHostStatus::OK.0);

    let host = HostValues {
        call,
        exit,
        context,
        execute_site,
    };
    let slots = SiteSlots::new(&mut builder);
    for block in &function.blocks {
        builder.switch_to_block(blocks[&block.id]);
        for instruction in &block.instructions {
            let result = slots.call(&mut builder, &abi, &host, function.id.0, &instruction.site);
            continue_or_exit(&mut builder, result, failure, return_ok);
        }
        let result = slots.call(
            &mut builder,
            &abi,
            &host,
            function.id.0,
            &block.terminator.site,
        );
        dispatch_terminator(
            &mut builder,
            result,
            &block.terminator.kind,
            &blocks,
            failure,
            return_ok,
        );
    }

    builder.seal_all_blocks();
    builder.finalize();
    Ok(CraneliftFunction {
        function: function.id,
        ir,
    })
}

fn dispatch_resume_block(
    builder: &mut FunctionBuilder<'_>,
    abi: &AbiTypes,
    call: Value,
    blocks: &BTreeMap<crate::NativeBlockId, Block>,
    invalid: Block,
) {
    let frame = builder.ins().load(
        abi.pointer,
        MemFlags::new(),
        call,
        super::abi::call_frame_offset(),
    );
    let frame_missing = builder.ins().icmp_imm(IntCC::Equal, frame, 0);
    let resume_ready = builder.create_block();
    builder
        .ins()
        .brif(frame_missing, invalid, &[], resume_ready, &[]);
    builder.switch_to_block(resume_ready);
    let resume = builder.ins().load(
        abi.pointer,
        MemFlags::new(),
        frame,
        super::abi::frame_resume_offset(),
    );
    let resume_missing = builder.ins().icmp_imm(IntCC::Equal, resume, 0);
    let block_ready = builder.create_block();
    builder
        .ins()
        .brif(resume_missing, invalid, &[], block_ready, &[]);
    builder.switch_to_block(block_ready);
    let requested = builder.ins().load(
        types::I32,
        MemFlags::new(),
        resume,
        super::abi::resume_block_offset(),
    );
    for (index, (block, target)) in blocks.iter().enumerate() {
        let matches = builder
            .ins()
            .icmp_imm(IntCC::Equal, requested, i64::from(block.0));
        if index + 1 == blocks.len() {
            builder.ins().brif(matches, *target, &[], invalid, &[]);
        } else {
            let next = builder.create_block();
            builder.ins().brif(matches, *target, &[], next, &[]);
            builder.switch_to_block(next);
        }
    }
}

fn continue_or_exit(
    builder: &mut FunctionBuilder<'_>,
    result: SiteCallResult,
    failure: Block,
    return_ok: Block,
) {
    let status_ok = builder.ins().icmp_imm(IntCC::Equal, result.status, 0);
    let outcome = builder.create_block();
    let status_return = builder.create_block();
    builder.append_block_param(status_return, types::I32);
    builder
        .ins()
        .brif(status_ok, outcome, &[], status_return, &[result.status]);
    builder.switch_to_block(status_return);
    let status = builder.block_params(status_return)[0];
    builder.ins().return_(&[status]);
    builder.switch_to_block(outcome);
    validate_outcome_header(builder, &result, failure);
    let is_continue = builder.ins().icmp_imm(
        IntCC::Equal,
        result.outcome_kind,
        i64::from(NativeSiteOutcomeKind::CONTINUE.0),
    );
    let continue_edge_is_zero = builder.ins().icmp_imm(IntCC::Equal, result.edge, 0);
    let valid_continue = builder.ins().band(is_continue, continue_edge_is_zero);
    let not_continue = builder.create_block();
    let continuation = builder.create_block();
    builder
        .ins()
        .brif(valid_continue, continuation, &[], not_continue, &[]);
    builder.switch_to_block(not_continue);
    let is_exit = builder.ins().icmp_imm(
        IntCC::Equal,
        result.outcome_kind,
        i64::from(NativeSiteOutcomeKind::EXIT.0),
    );
    let exit_edge_is_zero = builder.ins().icmp_imm(IntCC::Equal, result.edge, 0);
    let valid_exit = builder.ins().band(is_exit, exit_edge_is_zero);
    builder.ins().brif(valid_exit, return_ok, &[], failure, &[]);
    builder.switch_to_block(continuation);
}

fn dispatch_terminator(
    builder: &mut FunctionBuilder<'_>,
    result: SiteCallResult,
    terminator: &crate::NativeTerminatorKind,
    blocks: &BTreeMap<crate::NativeBlockId, Block>,
    failure: Block,
    return_ok: Block,
) {
    let status_ok = builder.ins().icmp_imm(IntCC::Equal, result.status, 0);
    let outcome = builder.create_block();
    let status_return = builder.create_block();
    builder.append_block_param(status_return, types::I32);
    builder
        .ins()
        .brif(status_ok, outcome, &[], status_return, &[result.status]);
    builder.switch_to_block(status_return);
    let status = builder.block_params(status_return)[0];
    builder.ins().return_(&[status]);
    builder.switch_to_block(outcome);
    validate_outcome_header(builder, &result, failure);
    let is_exit = builder.ins().icmp_imm(
        IntCC::Equal,
        result.outcome_kind,
        i64::from(NativeSiteOutcomeKind::EXIT.0),
    );
    let edge_check = builder.create_block();
    let exit_edge_is_zero = builder.ins().icmp_imm(IntCC::Equal, result.edge, 0);
    let valid_exit = builder.ins().band(is_exit, exit_edge_is_zero);
    builder
        .ins()
        .brif(valid_exit, return_ok, &[], edge_check, &[]);
    builder.switch_to_block(edge_check);
    let is_edge = builder.ins().icmp_imm(
        IntCC::Equal,
        result.outcome_kind,
        i64::from(NativeSiteOutcomeKind::EDGE.0),
    );
    let first_edge = builder.create_block();
    builder.ins().brif(is_edge, first_edge, &[], failure, &[]);
    builder.switch_to_block(first_edge);

    let edges = super::terminator::edges(terminator);
    if edges.is_empty() {
        builder.ins().jump(failure, &[]);
        return;
    }
    for (index, edge) in edges.iter().enumerate() {
        let matches = builder
            .ins()
            .icmp_imm(IntCC::Equal, result.edge, index as i64);
        let target = blocks[&edge.target];
        if index + 1 == edges.len() {
            builder.ins().brif(matches, target, &[], failure, &[]);
        } else {
            let next = builder.create_block();
            builder.ins().brif(matches, target, &[], next, &[]);
            builder.switch_to_block(next);
        }
    }
}

fn validate_outcome_header(
    builder: &mut FunctionBuilder<'_>,
    result: &SiteCallResult,
    failure: Block,
) {
    let reserved = builder.ins().bor(result.flags, result.reserved);
    let valid = builder.ins().icmp_imm(IntCC::Equal, reserved, 0);
    let valid_outcome = builder.create_block();
    builder.ins().brif(valid, valid_outcome, &[], failure, &[]);
    builder.switch_to_block(valid_outcome);
}

fn return_status(builder: &mut FunctionBuilder<'_>, status: u32) {
    let status = builder.ins().iconst(types::I32, i64::from(status));
    builder.ins().return_(&[status]);
}
