use cranelift::prelude::*;
use cranelift_codegen::ir::StackSlot;
use runmat_execution::host::{NativeAbiField, NativeAbiType};

use crate::{NativeMirSite, NativeSitePhase};

pub(super) struct SiteSlots {
    request: StackSlot,
    outcome: StackSlot,
}

pub(super) struct HostValues {
    pub call: Value,
    pub exit: Value,
    pub context: Value,
    pub execute_site: Value,
}

pub(super) struct SiteCallResult {
    pub status: Value,
    pub outcome_kind: Value,
    pub edge: Value,
    pub flags: Value,
    pub reserved: Value,
}

impl SiteSlots {
    pub fn new(builder: &mut FunctionBuilder<'_>, abi: &super::abi::AbiTypes<'_>) -> Self {
        let request_layout = abi.layout.layout(NativeAbiType::SiteRequest);
        let outcome_layout = abi.layout.layout(NativeAbiType::SiteOutcome);
        let request = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            request_layout.size,
            request_layout.alignment.ilog2() as u8,
        ));
        let outcome = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            outcome_layout.size,
            outcome_layout.alignment.ilog2() as u8,
        ));
        Self { request, outcome }
    }

    pub fn call(
        &self,
        builder: &mut FunctionBuilder<'_>,
        abi: &super::abi::AbiTypes<'_>,
        host: &HostValues,
        function: u32,
        site: &NativeMirSite,
    ) -> SiteCallResult {
        let request = builder.ins().stack_addr(abi.pointer, self.request, 0);
        let outcome = builder.ins().stack_addr(abi.pointer, self.outcome, 0);
        store_u32(
            builder,
            request,
            abi.offset(NativeAbiField::RequestFunction),
            function,
        );
        store_u32(
            builder,
            request,
            abi.offset(NativeAbiField::RequestBlock),
            site.point.block,
        );
        store_u32(
            builder,
            request,
            abi.offset(NativeAbiField::RequestPosition),
            site.point.position,
        );
        store_u32(
            builder,
            request,
            abi.offset(NativeAbiField::RequestPhase),
            phase(site.phase).0,
        );
        store_u32(
            builder,
            request,
            abi.offset(NativeAbiField::RequestOrdinal),
            site.ordinal,
        );
        store_u32(
            builder,
            request,
            abi.offset(NativeAbiField::RequestReserved),
            0,
        );
        for field in [
            NativeAbiField::OutcomeKind,
            NativeAbiField::OutcomeEdge,
            NativeAbiField::OutcomeFlags,
            NativeAbiField::OutcomeReserved,
        ] {
            store_u32(builder, outcome, abi.offset(field), 0);
        }
        let signature = builder.import_signature(abi.execute_site_signature.clone());
        let call = builder.ins().call_indirect(
            signature,
            host.execute_site,
            &[host.context, host.call, request, outcome, host.exit],
        );
        let status = builder.inst_results(call)[0];
        let outcome_kind = builder.ins().load(
            types::I32,
            MemFlags::new(),
            outcome,
            abi.offset(NativeAbiField::OutcomeKind),
        );
        let edge = builder.ins().load(
            types::I32,
            MemFlags::new(),
            outcome,
            abi.offset(NativeAbiField::OutcomeEdge),
        );
        let flags = builder.ins().load(
            types::I32,
            MemFlags::new(),
            outcome,
            abi.offset(NativeAbiField::OutcomeFlags),
        );
        let reserved = builder.ins().load(
            types::I32,
            MemFlags::new(),
            outcome,
            abi.offset(NativeAbiField::OutcomeReserved),
        );
        SiteCallResult {
            status,
            outcome_kind,
            edge,
            flags,
            reserved,
        }
    }
}

fn store_u32(builder: &mut FunctionBuilder<'_>, base: Value, offset: i32, value: u32) {
    let value = builder.ins().iconst(types::I32, i64::from(value));
    builder.ins().store(MemFlags::new(), value, base, offset);
}

fn phase(phase: NativeSitePhase) -> runmat_runtime::native::NativeSitePhase {
    match phase {
        NativeSitePhase::Rvalue => runmat_runtime::native::NativeSitePhase::RVALUE,
        NativeSitePhase::Statement => runmat_runtime::native::NativeSitePhase::STATEMENT,
        NativeSitePhase::TerminatorRvalue => {
            runmat_runtime::native::NativeSitePhase::TERMINATOR_RVALUE
        }
        NativeSitePhase::Terminator => runmat_runtime::native::NativeSitePhase::TERMINATOR,
    }
}
