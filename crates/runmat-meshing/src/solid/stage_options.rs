use runmat_geometry_core::GeometryAsset;
use runmat_meshing_core::VolumeMeshingOptions;
use runmat_meshing_curve::CurveDiscretizationOptions;
use runmat_meshing_surface::SurfaceDiscretizationOptions;

use super::sizing::target_curve_size_m;

pub(super) fn curve_discretization_options(
    options: &VolumeMeshingOptions,
    geometry: &GeometryAsset,
) -> CurveDiscretizationOptions {
    CurveDiscretizationOptions {
        target_size_m: target_curve_size_m(options, geometry),
        ..CurveDiscretizationOptions::default()
    }
}

pub(super) fn surface_discretization_options() -> SurfaceDiscretizationOptions {
    SurfaceDiscretizationOptions::default()
}
