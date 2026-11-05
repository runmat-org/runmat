//! Camera tests.

#[test]
fn parallel_view_keeps_size_with_depth_and_frames_like_perspective_at_the_target() {
    for depth_mode in [DepthMode::Standard, DepthMode::ReversedZ] {
        let mut camera = Camera::new();
        camera.depth_mode = depth_mode;
        camera.aspect_ratio = 1.6;
        camera.position = Vec3::new(0.0, 0.0, 100.0);
        camera.target = Vec3::ZERO;
        camera.up = Vec3::Y;
        camera.projection = ProjectionType::Perspective {
            fov: 30.0_f32.to_radians(),
            near: 1.0,
            far: 500.0,
        };
        let ndc = |camera: &mut Camera, p: Vec3| {
            let clip = camera.view_proj_matrix() * p.extend(1.0);
            clip.truncate() / clip.w
        };
        // At the target the two projections agree.
        let edge = Vec3::new(0.0, 100.0 * 15.0_f32.to_radians().tan(), 0.0);
        camera.mark_dirty();
        let perspective = ndc(&mut camera, edge);
        camera.parallel = true;
        camera.mark_dirty();
        let parallel = ndc(&mut camera, edge);
        assert!((perspective.y - 1.0).abs() < 1e-4 && (parallel.y - 1.0).abs() < 1e-4);
        // Off the target only the parallel view keeps the size, and depth still orders.
        let near_point = ndc(&mut camera, Vec3::new(0.0, 10.0, 40.0));
        let far_point = ndc(&mut camera, Vec3::new(0.0, 10.0, -40.0));
        assert!((near_point.y - far_point.y).abs() < 1e-5);
        let nearer = match depth_mode {
            DepthMode::Standard => near_point.z < far_point.z,
            DepthMode::ReversedZ => near_point.z > far_point.z,
        };
        assert!(
            nearer,
            "{depth_mode:?}: the nearer point wins the depth test"
        );
        assert!((0.0..=1.0).contains(&near_point.z) && (0.0..=1.0).contains(&far_point.z));
        // A point behind the eye (a view zoomed into a part) still draws.
        let behind = ndc(&mut camera, Vec3::new(0.0, 0.0, 150.0));
        assert!((0.0..=1.0).contains(&behind.z));
    }
}

#[test]
fn parallel_fit_frames_the_box_by_its_extent_across_the_view() {
    let mut camera = Camera::new();
    camera.aspect_ratio = 1.0;
    camera.parallel = true;
    camera.up = Vec3::Y;
    camera.position = Vec3::new(0.0, 0.0, 10.0);
    camera.target = Vec3::ZERO;
    // A deep box: 20 across, 200 along the view.
    camera.fit_bounds(
        Vec3::new(-10.0, -10.0, -100.0),
        Vec3::new(10.0, 10.0, 100.0),
    );
    let corner = Vec3::new(10.0, 10.0, 100.0);
    let clip = camera.view_proj_matrix() * corner.extend(1.0);
    let y = clip.y / clip.w;
    // Fills the view to the fit margin (0.92) whatever the depth.
    assert!((y - 0.92).abs() < 0.01, "corner at {y}");
}
use super::*;

#[test]
fn scaled_camera_reset_zoom_and_orbit_preserve_their_screen_anchors() {
    let center = Vec3::new(100.0, 1.0, 25.0);
    let size = Vec2::new(800.0, 600.0);
    let cursor = Vec2::new(600.0, 220.0);
    for explicit_anchor in [false, true] {
        let mut camera = Camera::new();
        camera.axis_scale = Vec3::new(0.02, 1.0, 0.04);
        camera.target = center * camera.axis_scale;
        camera.position = camera.target + Vec3::new(4.0, -4.0, 4.0);
        let distance = camera.position.distance(camera.target);
        camera.reset_perspective_orientation(Some(center));
        assert!((camera.position.distance(camera.target) - distance).abs() < 1e-5);
        let projected = camera.view_proj_matrix().project_point3(center);
        assert!(projected.truncate().length() < 1e-5, "{projected:?}");
        assert!(projected.z > 0.0 && projected.z < 1.0);

        let near = camera.screen_to_render(cursor, size, 0.0);
        let far = camera.screen_to_render(cursor, size, 1.0);
        let direction = (far - near).normalize();
        let normal = (camera.target - camera.position).normalize();
        let distance = (camera.target - near).dot(normal) / direction.dot(normal);
        let anchor = (near + direction * distance) / camera.axis_scale;
        let screen_anchor = camera.view_proj_matrix().project_point3(anchor).truncate();
        let mut controller = CameraController::new();
        controller.mouse_wheel(
            Vec2::Y,
            cursor,
            (800, 600),
            Modifiers::default(),
            &mut camera,
        );
        let after_zoom = camera.view_proj_matrix().project_point3(anchor).truncate();
        assert!(
            after_zoom.distance(screen_anchor) < 1e-4,
            "{after_zoom:?} != {screen_anchor:?}"
        );

        controller
            .set_navigation_options(NavigationOptions {
                orbit_center: OrbitCenter::UnderCursor,
                ..Default::default()
            })
            .unwrap();
        let orbit_anchor = if explicit_anchor {
            controller.set_cursor_orbit_anchor(Some(anchor)).unwrap();
            anchor
        } else {
            // Dolly moves the target plane. Without a host-picked point,
            // orbit selects the cursor intersection with that new plane,
            // which need not be the pre-zoom anchor at its original depth.
            let near = camera.screen_to_render(cursor, size, 0.0);
            let far = camera.screen_to_render(cursor, size, 1.0);
            let direction = (far - near).normalize();
            let normal = (camera.target - camera.position).normalize();
            let distance = (camera.target - near).dot(normal) / direction.dot(normal);
            (near + direction * distance) / camera.axis_scale
        };
        let orbit_screen_anchor = camera
            .view_proj_matrix()
            .project_point3(orbit_anchor)
            .truncate();
        assert!(orbit_screen_anchor.distance(screen_anchor) < 1e-4);
        controller.mouse_press(cursor, MouseButton::Left, Modifiers::default());
        controller.mouse_move(
            cursor,
            Vec2::new(8.0, 5.0),
            (800, 600),
            Modifiers::default(),
            &mut camera,
        );
        let after_orbit = camera
            .view_proj_matrix()
            .project_point3(orbit_anchor)
            .truncate();
        assert!(
            after_orbit.distance(orbit_screen_anchor) < 1e-4,
            "explicit_anchor={explicit_anchor}: {after_orbit:?} != {orbit_screen_anchor:?}"
        );
    }
}

#[test]
fn navigation_options_preserve_defaults_and_apply_trackpad_zoom_and_pivots() {
    let mut controller = CameraController::new();
    assert_eq!(
        controller.navigation_options(),
        NavigationOptions::default()
    );
    for speed in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
        assert!(controller
            .set_navigation_options(NavigationOptions {
                zoom_speed: speed,
                ..Default::default()
            })
            .is_err());
    }
    for (reverse, expected_in) in [(false, true), (true, false)] {
        controller
            .set_navigation_options(NavigationOptions {
                reverse_zoom: reverse,
                zoom_to_cursor: false,
                ..Default::default()
            })
            .unwrap();
        let mut camera = Camera::new();
        let before = camera.position.distance(camera.target);
        // Wheel dolly translates target too; inspect movement toward the
        // original target rather than eye-to-target distance.
        let target = camera.target;
        controller.mouse_wheel(
            Vec2::Y,
            Vec2::ZERO,
            (800, 600),
            Modifiers::default(),
            &mut camera,
        );
        assert_eq!(camera.position.distance(target) < before, expected_in);
    }
    controller
        .set_navigation_options(NavigationOptions {
            trackpad_pan: true,
            ..Default::default()
        })
        .unwrap();
    let mut camera = Camera::new_2d((-1.0, 1.0, -1.0, 1.0));
    controller.trackpad_scroll(
        Vec2::new(3.0, 4.0),
        Vec2::ZERO,
        (800, 600),
        Modifiers::default(),
        &mut camera,
    );
    let ProjectionType::Orthographic {
        left,
        right,
        bottom,
        top,
        ..
    } = camera.projection
    else {
        unreachable!()
    };
    assert!((right - left - 2.0).abs() < 1e-6 && (top - bottom - 2.0).abs() < 1e-6);
    assert_ne!(left, -1.0);
    for center in [
        OrbitCenter::ViewCenter,
        OrbitCenter::Origin,
        OrbitCenter::UnderCursor,
    ] {
        controller
            .set_navigation_options(NavigationOptions {
                orbit_center: center,
                ..Default::default()
            })
            .unwrap();
        let mut camera = Camera::new();
        camera.target = Vec3::new(0.3, 0.1, 0.0);
        let pivot = match center {
            OrbitCenter::ViewCenter => camera.target,
            OrbitCenter::Origin => Vec3::ZERO,
            OrbitCenter::UnderCursor => Vec3::new(0.5, 0.2, 0.0),
        };
        if center == OrbitCenter::UnderCursor {
            controller.set_cursor_orbit_anchor(Some(pivot)).unwrap();
        }
        let distance = camera.position.distance(pivot);
        controller.mouse_press(
            Vec2::new(100.0, 100.0),
            MouseButton::Left,
            Modifiers::default(),
        );
        controller.mouse_move(
            Vec2::new(110.0, 105.0),
            Vec2::new(10.0, 5.0),
            (800, 600),
            Modifiers::default(),
            &mut camera,
        );
        assert!((camera.position.distance(pivot) - distance).abs() < 1e-5);
        assert!(camera.position.is_finite() && camera.target.is_finite());
    }
    for preset in [
        NavigationPreset::Legacy,
        NavigationPreset::Cad,
        NavigationPreset::Dcc,
    ] {
        controller
            .set_navigation_options(NavigationOptions {
                preset,
                ..Default::default()
            })
            .unwrap();
        let mut camera = Camera::new();
        let before = camera.position;
        let (button, modifiers) = match preset {
            NavigationPreset::Legacy => (MouseButton::Left, Modifiers::default()),
            NavigationPreset::Cad => (MouseButton::Middle, Modifiers::default()),
            NavigationPreset::Dcc => (
                MouseButton::Left,
                Modifiers {
                    alt: true,
                    ..Default::default()
                },
            ),
        };
        controller.mouse_press(Vec2::ZERO, button, modifiers);
        controller.mouse_move(
            Vec2::new(10.0, 5.0),
            Vec2::new(10.0, 5.0),
            (800, 600),
            modifiers,
            &mut camera,
        );
        assert_ne!(camera.position, before);
    }
}

#[test]
fn test_camera_creation() {
    let camera = Camera::new();
    assert_eq!(camera.position, Vec3::new(3.5, 3.5, 3.5));
    assert_eq!(camera.target, Vec3::ZERO);
    assert_eq!(camera.up, Vec3::Z);
}

#[test]
fn test_2d_camera() {
    let camera = Camera::new_2d((-10.0, 10.0, -10.0, 10.0));
    assert_eq!(camera.position, Vec3::new(0.0, 0.0, 1.0));
    assert_eq!(camera.target, Vec3::new(0.0, 0.0, 0.0));
    match camera.projection {
        ProjectionType::Orthographic {
            left,
            right,
            bottom,
            top,
            ..
        } => {
            assert_eq!(left, -10.0);
            assert_eq!(right, 10.0);
            assert_eq!(bottom, -10.0);
            assert_eq!(top, 10.0);
        }
        _ => panic!("Expected orthographic projection"),
    }
}

#[test]
fn test_camera_bounds_fitting() {
    let mut camera = Camera::new_2d((-1.0, 1.0, -1.0, 1.0));
    let min_bounds = Vec3::new(-5.0, -3.0, 0.0);
    let max_bounds = Vec3::new(5.0, 3.0, 0.0);

    camera.fit_bounds(min_bounds, max_bounds);

    // Check that the bounds were expanded appropriately
    match camera.projection {
        ProjectionType::Orthographic {
            left,
            right,
            bottom,
            top,
            ..
        } => {
            assert!(left <= -5.0);
            assert!(right >= 5.0);
            assert!(bottom <= -3.0);
            assert!(top >= 3.0);
        }
        _ => panic!("Expected orthographic projection"),
    }
}

#[test]
fn test_2d_camera_tracks_non_origin_bounds_center() {
    let camera = Camera::new_2d((10.0, 30.0, -2.0, 2.0));
    assert_eq!(camera.position, Vec3::new(20.0, 0.0, 1.0));
    assert_eq!(camera.target, Vec3::new(20.0, 0.0, 0.0));
}

#[test]
fn test_set_view_angles_preserves_distance() {
    let mut camera = Camera::new();
    camera.target = Vec3::new(1.0, 2.0, 3.0);
    camera.position = camera.target + Vec3::new(2.0, 0.0, 0.0);
    camera.set_view_angles_deg(90.0, 0.0);
    let offset = camera.position - camera.target;
    assert!((offset.length() - 2.0).abs() < 1e-5);
    assert!(offset.x.abs() < 1e-4);
    assert!((offset.y - 2.0).abs() < 1e-4);
}

#[test]
fn figure_fit_contains_all_display_corners_at_wide_and_narrow_aspects() {
    let min = Vec3::new(-8.0, 2.0, -0.5);
    let max = Vec3::new(12.0, 7.0, 2.5);
    for viewport in [(1200, 400), (200, 700)] {
        for angles in [(45.0, 30.0), (-120.0, 50.0)] {
            let mut camera = Camera::new();
            camera.set_view_angles_deg(angles.0, angles.1);
            let original_direction = (camera.position - camera.target).normalize();
            assert!(camera.fit_figure_perspective_bounds(min, max, viewport));
            assert!(
                ((camera.position - camera.target).normalize() - original_direction).length()
                    < 1.0e-5
            );
            let matrix = camera.view_proj_matrix();
            for x in [min.x, max.x] {
                for y in [min.y, max.y] {
                    for z in [min.z, max.z] {
                        let clip = matrix * Vec3::new(x, y, z).extend(1.0);
                        let ndc = clip.truncate() / clip.w;
                        assert!(
                            clip.w > 0.0 && ndc.x.abs() < 1.0 && ndc.y.abs() < 1.0,
                            "corner ({x},{y},{z}) outside {viewport:?}: {ndc:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn figure_fit_is_invariant_to_data_units_with_normalized_axes() {
    let min = Vec3::new(-8.0, 2.0, -0.5);
    let max = Vec3::new(12.0, 7.0, 2.5);
    for viewport in [(1200, 400), (200, 700)] {
        let mut reference: Option<Vec<Vec3>> = None;
        for units in [0.001_f32, 1.0, 1000.0] {
            let mut camera = Camera::new();
            camera.axis_scale = Vec3::new(1.0 / 20.0, 1.0 / 5.0, 1.0 / 3.0) / units;
            camera.set_view_angles_deg(45.0, 30.0);
            assert!(camera.fit_figure_perspective_bounds(min * units, max * units, viewport));
            let matrix = camera.view_proj_matrix();
            let mut corners = Vec::new();
            for x in [min.x, max.x] {
                for y in [min.y, max.y] {
                    for z in [min.z, max.z] {
                        let clip = matrix * (Vec3::new(x, y, z) * units).extend(1.0);
                        let ndc = clip.truncate() / clip.w;
                        assert!(clip.w > 0.0 && ndc.x.abs() < 1.0 && ndc.y.abs() < 1.0);
                        assert!((0.0..=1.0).contains(&ndc.z));
                        corners.push(ndc);
                    }
                }
            }
            assert!(corners.iter().any(|p| p.x.abs().max(p.y.abs()) > 0.7));
            if let Some(expected) = &reference {
                for (actual, expected) in corners.iter().zip(expected) {
                    assert!((*actual - *expected).length() < 1.0e-4);
                }
            } else {
                reference = Some(corners);
            }
        }
    }
}
