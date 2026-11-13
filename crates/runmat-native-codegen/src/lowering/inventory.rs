use crate::ir::NativeMirSite;
use crate::{NativeCodegenError, NativeCodegenResult};
use runmat_types::ProgramFunctionId;

/// Independently inventories canonical MIR sites before lowering. The Native IR
/// verifier compares this list with emitted sites so adding an unsupported MIR
/// arm cannot silently disappear behind the lowering implementation.
pub(super) fn expected_sites(
    function: ProgramFunctionId,
    body: &runmat_mir::MirBody,
) -> NativeCodegenResult<Vec<NativeMirSite>> {
    runmat_mir::operation_sites(function, body).ok_or_else(|| {
        NativeCodegenError::new(
            "native.lowering.site_inventory",
            "canonical MIR site identity exceeds the Native IR schema or is duplicated",
        )
        .at_function(body.function)
    })
}
