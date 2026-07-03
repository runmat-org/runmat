use runmat_geometry_core::GeometryAsset;
use runmat_meshing_core::{MeshTargetSize, VolumeMeshingOptions};

pub(super) fn target_curve_size_m(options: &VolumeMeshingOptions, geometry: &GeometryAsset) -> f64 {
    match options.target_size {
        MeshTargetSize::LengthM(length) if length.is_finite() && length > 0.0 => length,
        MeshTargetSize::Auto => geometry_span_m(geometry).unwrap_or(1.0) / 8.0,
        _ => 0.05,
    }
    .max(options.min_size_m.unwrap_or(f64::EPSILON))
    .min(options.max_size_m.unwrap_or(f64::INFINITY))
}

fn geometry_span_m(geometry: &GeometryAsset) -> Option<f64> {
    let vertices = geometry
        .surface_meshes
        .iter()
        .flat_map(|mesh| mesh.vertices.iter().copied());
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    let mut count = 0_usize;
    for vertex in vertices {
        count += 1;
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    (count > 0).then(|| {
        (0..3)
            .map(|axis| max[axis] - min[axis])
            .fold(0.0_f64, f64::max)
    })
}
