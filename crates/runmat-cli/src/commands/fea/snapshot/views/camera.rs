//! One world-space camera for every saved view of an example, including deformation.
use runmat_plot::core::{camera::ProjectionType, BoundingBox, Camera};
use runmat_runtime::builtins::plotting::PlotCameraProjection;
use runmat_runtime::builtins::plotting::{PlotCameraState, PlotSurfaceCameraState};

/// The standard poster direction, from the target towards the camera: about MATLAB
/// `view(19, 23)`.
const POSTER_DIRECTION: [f32; 3] = [0.35, -1.0, 0.45];

/// The direction from the target towards the camera for MATLAB `view(az, el)` in degrees:
/// azimuth from the -y axis towards +x, elevation above the xy plane.
pub(super) fn view_direction(azimuth_deg: f32, elevation_deg: f32) -> [f32; 3] {
    let (az, el) = (azimuth_deg.to_radians(), elevation_deg.to_radians());
    [az.sin() * el.cos(), -az.cos() * el.cos(), el.sin()]
}

/// `direction` points from the target towards the camera (default: the standard poster
/// view); `aspect_ratio` is the poster's width over height.
pub(super) fn shared_camera(
    bounds: impl Iterator<Item = BoundingBox>,
    direction: Option<[f32; 3]>,
    aspect_ratio: f32,
) -> PlotSurfaceCameraState {
    let bounds = bounds
        .reduce(|left, right| BoundingBox {
            min: left.min.min(right.min),
            max: left.max.max(right.max),
        })
        .expect("snapshot contains at least one view");
    let (min, max) = (bounds.min, bounds.max);
    let mut camera = Camera::new();
    camera.update_aspect_ratio(aspect_ratio);
    let center = (min + max) * 0.5;
    camera.target = center;
    let [x, y, z] = direction.unwrap_or(POSTER_DIRECTION);
    let mut direction = camera.up;
    (direction.x, direction.y, direction.z) = (x, y, z);
    camera.position = center + direction.normalize();
    // Every corner of the union of all views' bounds (the whole part, undeformed and
    // deformed) projects inside the frame, so no view crops the part.
    camera.fit_bounds(min, max);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    fn unit(a: [f32; 3]) -> [f32; 3] {
        let length = dot(a, a).sqrt();
        a.map(|value| value / length)
    }

    /// A long part with a tab far from its centre (a busbar) and a small deformed copy: every
    /// corner of both views' bounds lies inside the poster frame.
    #[test]
    fn posters_frame_every_view_whole() {
        let views = [
            BoundingBox {
                min: [0.0, 0.0, 0.0].into(),
                max: [0.40, 0.03, 0.006].into(),
            },
            BoundingBox {
                min: [0.38, -0.02, 0.0].into(),
                max: [0.45, 0.03, 0.03].into(),
            },
        ];
        let state = shared_camera(views.iter().copied(), None, 8.0 / 5.0);
        let camera = &state.axes[0];
        let PlotCameraProjection::Perspective { fov, .. } = camera.projection else {
            panic!("snapshot cameras are perspective");
        };
        let forward = unit(sub(camera.target, camera.position));
        let right = unit(cross(forward, camera.up));
        let up = cross(right, forward);
        let tan_y = (fov * 0.5).tan();
        let tan_x = tan_y * camera.aspect_ratio;
        for view in views {
            for corner in 0..8 {
                let pick = |axis: usize| {
                    if corner & (1 << axis) == 0 {
                        view.min[axis]
                    } else {
                        view.max[axis]
                    }
                };
                let offset = sub([pick(0), pick(1), pick(2)], camera.position);
                let depth = dot(offset, forward);
                assert!(depth > 0.0, "corner behind the camera");
                assert!(dot(offset, right).abs() / depth <= tan_x * 1.0001);
                assert!(dot(offset, up).abs() / depth <= tan_y * 1.0001);
            }
        }
    }

    /// MATLAB's view convention: view(0, 0) looks along +y from -y, view(90, 0) along -x
    /// from +x, view(az, 90) straight down; a wide poster frames the bounds whole.
    #[test]
    fn view_angles_follow_matlab_and_frame_the_bounds() {
        let close = |a: [f32; 3], b: [f32; 3]| (0..3).all(|i| (a[i] - b[i]).abs() < 1e-6);
        assert!(close(view_direction(0.0, 0.0), [0.0, -1.0, 0.0]));
        assert!(close(view_direction(90.0, 0.0), [1.0, 0.0, 0.0]));
        assert!(close(
            view_direction(-37.5, 90.0).map(|v| (v * 1e5).round() / 1e5),
            [0.0, 0.0, 1.0]
        ));
        let pack = BoundingBox {
            min: [-1.2, -0.1, 0.0].into(),
            max: [0.1, 0.7, 0.2].into(),
        };
        let state = shared_camera(
            std::iter::once(pack),
            Some(view_direction(-30.0, 30.0)),
            1.6,
        );
        let camera = &state.axes[0];
        assert_eq!(camera.aspect_ratio, 1.6);
        let from_target = unit(sub(camera.position, camera.target));
        assert!(close(
            from_target.map(|v| (v * 1e4).round() / 1e4),
            view_direction(-30.0, 30.0).map(|v| (v * 1e4).round() / 1e4)
        ));
    }
}
