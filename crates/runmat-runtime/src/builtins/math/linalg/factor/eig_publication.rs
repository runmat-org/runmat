//! One protected, exact-owner publication of the four canonical eigen roles.
use super::*;
#[path = "eig_publication/bundle.rs"]
mod bundle;
#[path = "eig_publication/transfer.rs"]
mod transfer;

pub(super) async fn upload_eig_eval_to_owner_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    eval: EigEval,
    source: &GpuTensorHandle,
    vector_output: bool,
    operands: &[GpuTensorHandle],
) -> BuiltinResult<EigEval> {
    if operands.len() > 2 {
        return Err(eig_internal_error("eig: coefficient owner extent changed"));
    }
    // Retain an already-owned, explicit-dominant common-owner prototype once.
    // Upload then publishes the correct original intent directly, without a
    // second derived provenance view after publication.
    let output_source = gpu_helpers::select_resident_output_source(
        std::iter::once(source.clone()).chain(operands.iter().cloned()),
        BUILTIN_NAME,
    )?
    .expect("the original source is present");
    let automatic = std::iter::once(source).chain(operands).all(|handle| {
        runmat_accelerate_api::handle_provenance(handle)
            == Some(runmat_accelerate_api::GpuHandleProvenance::Automatic)
    });
    let mut outputs = bundle::PublishedFactors::new();
    use bundle::FactorRole;
    let roles = [
        FactorRole::Eigenvalues,
        FactorRole::Diagonal,
        FactorRole::Right,
        FactorRole::Left,
    ];
    for role in roles {
        let value = match role {
            FactorRole::Eigenvalues => Some(&eval.eigenvalues),
            FactorRole::Diagonal => Some(&eval.diagonal_matrix),
            FactorRole::Right => Some(&eval.right),
            FactorRole::Left => eval.left.as_ref(),
        };
        let Some(value) = value else { continue };
        match transfer::publish_factor(runtime, value, &output_source, source, operands, &outputs) {
            Ok(output) => outputs.insert(role, output),
            Err(error) => {
                // Only certified pre-effect capability absence may retain
                // the original complete host bundle. Never resource/control,
                // unproven-effect, or failed-retirement fallback.
                if automatic && transfer::is_pre_effect_unsupported(&error) {
                    return match outputs.retire() {
                        Ok(()) => Ok(eval),
                        Err(cleanup) => {
                            crate::runtime_error::complete_with_cleanup(Err(error), Err(cleanup))
                        }
                    };
                }
                return Err(outputs.reject(error));
            }
        }
    }
    Ok(outputs.finish(vector_output))
}
