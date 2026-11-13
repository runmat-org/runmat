use crate::{NativeAssembly, NativeCodegenError, NativeCodegenResult};
use runmat_execution_artifact::target_test::NativePairedProductHeader;

use super::AotProgramManifest;

/// Compare role-specific callable roots with the actual admitted Native IR and
/// AOT program. The physical resolver may also contain retained helpers and
/// the application entrypoint; those are not test-callable through this header.
/// Body digests identify canonical MIR pairs, not generated machine code. The
/// caller must separately authenticate the exact embedded Native IR bytes
/// against `program.native_ir_digest` and the linked object, then select tests
/// only through the header's binding.
pub fn validate_paired_product_header(
    header: &NativePairedProductHeader,
    assembly: &NativeAssembly,
    program: &AotProgramManifest,
) -> NativeCodegenResult<()> {
    header.validate().map_err(paired_error)?;
    assembly.verify()?;
    program.validate()?;
    super::validate_native_product(
        &program.product,
        &assembly.target,
        *assembly.program.runtime_fingerprint(),
        *assembly.program.catalog_fingerprint(),
        header.binding.selected_product.optimization,
    )?;
    if program.executable != assembly.executable_identity
        || program.entrypoint != assembly.selected_entrypoint
        || program.product.selected() != Some(&header.binding.selected_product)
        || header.binding.program_revision != assembly.program
    {
        return Err(paired_error(
            "paired header, AOT program, and Native IR differ",
        ));
    }
    for entry in &header.linked_entries {
        if assembly
            .entrypoints
            .binary_search(&entry.compiled_function)
            .is_err()
            || assembly
                .functions
                .binary_search_by_key(&entry.compiled_function, |function| function.id)
                .ok()
                .and_then(|index| assembly.functions.get(index))
                .is_none_or(|function| {
                    function.portable_id != entry.function || function.mir_body_digest != entry.body
                })
        {
            return Err(paired_error(
                "paired linked entry differs from its admitted Native IR function",
            ));
        }
    }
    Ok(())
}

fn paired_error(error: impl std::fmt::Display) -> NativeCodegenError {
    NativeCodegenError::new("native.aot.paired_product", error.to_string())
}
