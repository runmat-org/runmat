//! One world-space camera for every saved view of an example, including deformation.
use runmat_plot::core::{camera::ProjectionType, BoundingBox, Camera};
use runmat_runtime::builtins::plotting::PlotCameraProjection;
use runmat_runtime::builtins::plotting::{PlotCameraState, PlotSurfaceCameraState};

pub(super) fn shared_camera(bounds: impl Iterator<Item = BoundingBox>) -> PlotSurfaceCameraState {
    let bounds = bounds
        .reduce(|left, right| BoundingBox {
            min: left.min.min(right.min),
            max: left.max.max(right.max),
        })
        .expect("snapshot contains three native scenes");
    let (min, max) = (bounds.min, bounds.max);
    let mut camera = Camera::new();
    camera.update_aspect_ratio(8.0 / 5.0);
    let center = (min + max) * 0.5;
    camera.target = center;
    let mut direction = camera.up;
    direction.x = 0.35;
    direction.y = -1.0;
    direction.z = 0.45;
    camera.position = center + direction.normalize();
    camera.fit_bounds(min, max);
    // The shared fit leaves generous orbiting room. Bring the teaching example
    // closer while keeping the same framing for original and deformed views.
    camera.zoom(3.0);
    let ProjectionType::Perspective { fov, near, far } = camera.projection else {
        unreachable!("3D camera must be perspective")
    };
    PlotSurfaceCameraState {
        active_axes: 0,
        axes: vec![PlotCameraState {
            position: camera.position.to_array(),
            target: camera.target.to_array(),
            up: camera.up.to_array(),
            zoom: camera.zoom,
            aspect_ratio: camera.aspect_ratio,
            projection: PlotCameraProjection::Perspective { fov, near, far },
        }],
    }
}
