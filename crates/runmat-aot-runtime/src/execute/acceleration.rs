//! Standalone provider bootstrap from verified program intent and exact
//! target-built backend inventory. A linked backend is not itself a grant.

use runmat_types::CapabilityRequirement;

use super::linked_selected::LinkedSelectedAdmission;

pub(super) fn initialize_for_program(
    program: &runmat_native_codegen::aot::AotProgramManifest,
    assembly: &runmat_native_codegen::NativeAssembly,
    selected: Option<&LinkedSelectedAdmission>,
) -> Result<(), String> {
    let hard_required = assembly
        .requirements
        .capabilities
        .0
        .contains(&CapabilityRequirement::Accelerator);
    let physical = crate::accelerator_profile::PROFILE;
    let selected_granted = selected.is_none_or(|selected| {
        let binding = selected.product.binding();
        binding
            .available_runtime_capabilities
            .0
            .contains(&CapabilityRequirement::Accelerator)
            && binding
                .authorized_runtime_capabilities
                .0
                .contains(&CapabilityRequirement::Accelerator)
    });
    // Clear inherited process state even when admission fails below. An
    // embedding caller's provider cannot satisfy or survive this selection.
    runmat_accelerate_api::clear_provider();
    if hard_required && (!physical.has_backend() || !selected_granted) {
        return Err(
            "standalone program requires an accelerator absent from its linked runtime or selected grant"
                .into(),
        );
    }

    if !(hard_required || program.provider_reachable)
        || !physical.has_backend()
        || !selected_granted
    {
        return Ok(());
    }
    runmat_accelerate::initialize_acceleration_provider_with(
        &runmat_accelerate::AccelerateInitOptions {
            allow_inprocess_fallback: false,
            ..runmat_accelerate::AccelerateInitOptions::default()
        },
    );
    if hard_required && runmat_accelerate_api::provider().is_none() {
        return Err(
            "standalone program requires an accelerator, but no physical provider is available"
                .into(),
        );
    }
    Ok(())
}
