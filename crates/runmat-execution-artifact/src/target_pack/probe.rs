use super::*;
use runmat_execution::host::NativeAbiProbe;
use std::collections::BTreeMap;

pub(super) fn verify(
    manifest: &TargetPackManifest,
    contents: &BTreeMap<ComponentId, Vec<u8>>,
) -> ArtifactResult<NativeAbiProbe> {
    let bytes = contents
        .get(&manifest.abi_probe_component)
        .ok_or_else(|| invalid("selected closure omits its ABI probe"))?;
    let probe = NativeAbiProbe::from_canonical_bytes(bytes).map_err(contract)?;
    probe
        .binding()
        .map_err(contract)?
        .require_exact(&manifest.abi)
        .map_err(contract)?;
    let target = &manifest.target;
    if probe.architecture != target.target.architecture
        || probe.operating_system != target.target.operating_system
        || probe.pointer_width != target.target.pointer_width
        || probe.endianness != target.endianness
        || probe.environment != target.environment
    {
        return Err(invalid(
            "authenticated ABI probe does not describe the selected target",
        ));
    }
    Ok(probe)
}
