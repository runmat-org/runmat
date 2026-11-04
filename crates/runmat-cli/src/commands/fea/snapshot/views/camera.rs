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
        .expect("snapshot contains at least one view");
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
        let state = shared_camera(views.iter().copied());
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
}
