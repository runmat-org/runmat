//! Validated native compute emission. Never ingest untrusted raw SPIR-V.

pub(crate) fn compile(
    source: &str,
    metadata: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
) -> anyhow::Result<crate::wgsl::host_buffers::ClaimedVec<u32>> {
    let module = naga::front::wgsl::parse_str(source).map_err(|error| {
        // Naga owns this diagnostic rendering. Keep its useful spans and
        // ORIGINAL typed cause; formatting it into anyhow! lost the latter.
        let diagnostic = error.emit_to_string(source);
        anyhow::Error::new(error).context(diagnostic)
    })?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::FLOAT64 | naga::valid::Capabilities::SHADER_INT64,
    )
    .validate(&module)?;
    anyhow::ensure!(
        !module.entry_points.is_empty()
            && module
                .entry_points
                .iter()
                .all(|entry| entry.stage == naga::ShaderStage::Compute),
        "controlled binary64 modules must contain compute entrypoints"
    );
    let options = naga::back::spv::Options {
        // Vulkan 1.2 guarantees SPIR-V 1.5, including core float controls.
        lang_version: (1, 5),
        flags: naga::back::spv::WriterFlags::empty(),
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            index: naga::proc::BoundsCheckPolicy::Restrict,
            buffer: naga::proc::BoundsCheckPolicy::Restrict,
            image_load: naga::proc::BoundsCheckPolicy::Restrict,
            binding_array: naga::proc::BoundsCheckPolicy::Restrict,
        },
        zero_initialize_workgroup_memory:
            naga::back::spv::ZeroInitializeWorkgroupMemoryMode::Polyfill,
        ..Default::default()
    };
    // Naga's parser/validator/write_vec allocations are the explicit library
    // boundary. Our subsequent control-analysis buffers and transformed copy
    // use the original captured admission and retain their own exact claims.
    let words = naga::back::spv::write_vec(&module, &info, &options, None)?;
    super::controls::apply(&words, metadata)
}
