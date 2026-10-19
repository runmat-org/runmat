//! One implementation closure for all native meshing cache owners.
use runmat_execution::Digest;
use runmat_meshing_core::MeshingStageKind;

pub(crate) fn implementation(stage: MeshingStageKind) -> Option<Digest> {
    let value = if matches!(
        stage,
        MeshingStageKind::Sizing | MeshingStageKind::CurveMesh | MeshingStageKind::SurfaceMesh
    ) {
        env!("RUNMAT_MESHING_UPSTREAM_IMPLEMENTATION")
    } else {
        env!("RUNMAT_MESHING_VOLUME_IMPLEMENTATION")
    };
    if value.len() != 64 {
        return None;
    }
    let mut bytes = [0_u8; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).ok()?;
    }
    let mut closure = bytes.to_vec();
    closure.extend_from_slice(runmat_meshing_execution::meshing_build_configuration(stage).bytes());
    for configuration in [
        runmat_geometry_io::BUILD_CONFIGURATION,
        runmat_geometry_ops::BUILD_CONFIGURATION,
        runmat_cpu_execution::BUILD_CONFIGURATION,
    ] {
        closure.extend_from_slice(configuration.as_bytes());
    }
    Some(Digest::sha256(closure))
}
