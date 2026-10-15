use crate::{NativeCodegenError, NativeCodegenResult, NativeTarget};
use cranelift::prelude::*;
use runmat_execution::host::{NativeAbiField, NativeAbiProbe};

pub(super) struct AbiTypes<'a> {
    pub pointer: Type,
    pub entry_signature: Signature,
    pub execute_site_signature: Signature,
    pub layout: &'a NativeAbiProbe,
}

impl<'a> AbiTypes<'a> {
    pub fn for_target(target: &'a NativeTarget) -> NativeCodegenResult<Self> {
        let (selected, layout) = target.native_parts()?;
        let triple = selected
            .triple()
            .map_err(|error| NativeCodegenError::new("native.cranelift.target", error.to_string()))?
            .parse::<target_lexicon::Triple>()
            .map_err(|error| {
                NativeCodegenError::new("native.cranelift.target", error.to_string())
            })?;
        let pointer = types::I64; // Checked explicit probe/selected-target width.
        let call_conv = cranelift_codegen::isa::CallConv::triple_default(&triple);
        let mut entry_signature = Signature::new(call_conv);
        entry_signature
            .params
            .extend([AbiParam::new(pointer), AbiParam::new(pointer)]);
        entry_signature.returns.push(AbiParam::new(types::I32));
        let mut execute_site_signature = Signature::new(call_conv);
        execute_site_signature
            .params
            .extend([AbiParam::new(pointer); 5]);
        execute_site_signature
            .returns
            .push(AbiParam::new(types::I32));
        Ok(Self {
            pointer,
            entry_signature,
            execute_site_signature,
            layout,
        })
    }
    pub fn offset(&self, field: NativeAbiField) -> i32 {
        self.layout.offset(field) as i32
    }
}
