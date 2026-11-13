//! Portable objects use an explicit minimum baseline. Process-local JIT code
//! is host-tuned only after exact target/probe executable admission; it is not
//! a deployable baseline object or a persisted portable machine-code product.
use crate::{NativeCodegenError, NativeCodegenResult, NativeTarget};
use cranelift_codegen::{isa::OwnedTargetIsa, settings::Flags};
use runmat_execution::host::NativeCpuBaseline;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IsaCompilationPurpose {
    SelectedBaselineObject,
    ExactHostInProcessJit,
}

pub fn compilation_isa(
    target: &NativeTarget,
    purpose: IsaCompilationPurpose,
    flags: Flags,
) -> NativeCodegenResult<OwnedTargetIsa> {
    match purpose {
        IsaCompilationPurpose::SelectedBaselineObject => {}
        IsaCompilationPurpose::ExactHostInProcessJit => {
            // This check belongs here, before even querying host CPU features,
            // rather than relying only on a downstream executable-memory caller.
            target.require_host_executable()?;
            return cranelift_native::builder()
                .map_err(error)?
                .finish(flags)
                .map_err(error);
        }
    }
    let (selected, _) = target.native_parts()?;
    match selected.cpu_baseline {
        NativeCpuBaseline::X86_64V1 | NativeCpuBaseline::ArmV8A => {}
    }
    let triple = selected
        .triple()
        .map_err(error)?
        .parse::<target_lexicon::Triple>()
        .map_err(error)?;
    let isa = cranelift_codegen::isa::lookup(triple)
        .map_err(error)?
        .finish(flags)
        .map_err(error)?;
    // Pinned Cranelift defaults describe SSE2 / ARMv8-A, with every optional
    // ISA flag false. Fail closed if a backend revision changes that contract.
    if isa
        .isa_flags()
        .iter()
        .any(|flag| flag.as_bool() == Some(true))
    {
        return Err(error(
            "selected baseline unexpectedly enables optional ISA extensions",
        ));
    }
    Ok(isa)
}
fn error(error: impl std::fmt::Display) -> NativeCodegenError {
    NativeCodegenError::new("native.isa.selected", error.to_string())
}
