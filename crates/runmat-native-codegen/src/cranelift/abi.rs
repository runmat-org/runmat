use cranelift::prelude::*;

pub(super) struct AbiTypes {
    pub pointer: Type,
    pub entry_signature: Signature,
    pub execute_site_signature: Signature,
}

impl AbiTypes {
    pub fn current() -> Self {
        let pointer = if usize::BITS == 32 {
            types::I32
        } else {
            types::I64
        };
        let call_conv =
            cranelift_codegen::isa::CallConv::triple_default(&target_lexicon::Triple::host());
        let mut entry_signature = Signature::new(call_conv);
        entry_signature.params.push(AbiParam::new(pointer));
        entry_signature.params.push(AbiParam::new(pointer));
        entry_signature.returns.push(AbiParam::new(types::I32));

        let mut execute_site_signature = Signature::new(call_conv);
        execute_site_signature.params.extend([
            AbiParam::new(pointer),
            AbiParam::new(pointer),
            AbiParam::new(pointer),
            AbiParam::new(pointer),
            AbiParam::new(pointer),
        ]);
        execute_site_signature
            .returns
            .push(AbiParam::new(types::I32));
        Self {
            pointer,
            entry_signature,
            execute_site_signature,
        }
    }
}

pub(super) fn call_host_offset() -> i32 {
    std::mem::offset_of!(runmat_runtime::native::NativeCall, host) as i32
}

pub(super) fn host_context_offset() -> i32 {
    std::mem::offset_of!(runmat_runtime::native::NativeHostVTable, context) as i32
}

pub(super) fn host_execute_site_offset() -> i32 {
    std::mem::offset_of!(runmat_runtime::native::NativeHostVTable, execute_site) as i32
}
