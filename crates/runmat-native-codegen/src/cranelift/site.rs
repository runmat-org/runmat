use cranelift::prelude::*;
use cranelift_codegen::ir::StackSlot;

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
    pub fn new(builder: &mut FunctionBuilder<'_>) -> Self {
        let request = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            std::mem::size_of::<runmat_runtime::native::NativeSiteRequest>() as u32,
            2,
        ));
        let outcome = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            std::mem::size_of::<runmat_runtime::native::NativeSiteOutcome>() as u32,
            2,
        ));
        Self { request, outcome }
    }

    pub fn call(
        &self,
        builder: &mut FunctionBuilder<'_>,
        abi: &super::abi::AbiTypes,
        host: &HostValues,
        function: u32,
        site: &NativeMirSite,
    ) -> SiteCallResult {
        let request = builder.ins().stack_addr(abi.pointer, self.request, 0);
        let outcome = builder.ins().stack_addr(abi.pointer, self.outcome, 0);
        store_u32(builder, request, 0, function);
        store_u32(builder, request, 4, site.point.block);
        store_u32(builder, request, 8, site.point.position);
        store_u32(builder, request, 12, phase(site.phase).0);
        store_u32(builder, request, 16, site.ordinal);
        store_u32(builder, request, 20, 0);
        for offset in [0, 4, 8, 12] {
            store_u32(builder, outcome, offset, 0);
        }
        let signature = builder.import_signature(abi.execute_site_signature.clone());
        let call = builder.ins().call_indirect(
            signature,
            host.execute_site,
            &[host.context, host.call, request, outcome, host.exit],
        );
        let status = builder.inst_results(call)[0];
        let outcome_kind = builder.ins().load(types::I32, MemFlags::new(), outcome, 0);
        let edge = builder.ins().load(types::I32, MemFlags::new(), outcome, 4);
        let flags = builder.ins().load(types::I32, MemFlags::new(), outcome, 8);
        let reserved = builder.ins().load(types::I32, MemFlags::new(), outcome, 12);
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
