//! Camera system for 3D navigation and 2D plotting
//!
//! Provides both perspective and orthographic cameras with smooth
//! navigation controls for interactive plotting.

use crate::core::interaction::Modifiers;
use crate::core::{ClipPolicy, DepthMode};
use crate::core::{NavigationOptions, NavigationPreset, OrbitCenter};
use glam::{Mat4, Quat, Vec2, Vec3};

/// Camera projection type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProjectionType {
    Perspective {
        fov: f32,
        near: f32,
        far: f32,
    },
    Orthographic {
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraViewPreset {
    Perspective,
    Top,
    Bottom,
    Front,
    Back,
    Left,
    Right,
}

impl Default for ProjectionType {
    fn default() -> Self {
        Self::Perspective {
            fov: 45.0_f32.to_radians(),
            near: 0.1,
            far: 100.0,
        }
    }
}

/// Interactive camera for 3D plotting with smooth navigation
#[derive(Debug, Clone)]
pub struct Camera {
    // Position and orientation
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,

    // Projection parameters
    pub projection: ProjectionType,
    pub aspect_ratio: f32,
    /// Depth mapping mode used when building projection matrices.
    pub depth_mode: DepthMode,

    // Navigation state
    pub zoom: f32,
    pub rotation: Quat,

    // Interaction settings
    pub pan_sensitivity: f32,
    pub zoom_sensitivity: f32,
    pub rotate_sensitivity: f32,

    /// Per-axis scale from data units to the space the camera views. 3-D axes stretch each
    /// axis to fill a cube, as MATLAB's default plot box does; `ONE` keeps data units.
    pub axis_scale: Vec3,

    // Cached matrices
    view_matrix: Mat4,
    projection_matrix: Mat4,
    view_proj_dirty: bool,
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

impl Camera {
    /// Create a new camera with default 3D settings
    pub fn new() -> Self {
        let mut camera = Self {
            // CAD-like default: Z-up, with an isometric-ish starting view.
            position: Vec3::new(3.5, 3.5, 3.5),
            target: Vec3::ZERO,
            up: Vec3::Z,
            projection: ProjectionType::default(),
            aspect_ratio: 16.0 / 9.0,
            depth_mode: DepthMode::default(),
            zoom: 1.0,
            rotation: Quat::IDENTITY,
            pan_sensitivity: 0.01,
            zoom_sensitivity: 0.1,
            rotate_sensitivity: 0.005,
            axis_scale: Vec3::ONE,
            view_matrix: Mat4::IDENTITY,
            projection_matrix: Mat4::IDENTITY,
            view_proj_dirty: true,
        };
        camera.update_matrices();
        camera
    }

    /// Create a 2D orthographic camera for 2D plotting
    pub fn new_2d(bounds: (f32, f32, f32, f32)) -> Self {
        let (left, right, bottom, top) = bounds;
        let center_x = (left + right) / 2.0;
        let center_y = (bottom + top) / 2.0;
        let mut camera = Self {
            // Keep the eye directly above the data center for true 2D top-down view.
            // If position.x/y differ from target.x/y, the look-at matrix tilts and the plot
            // appears skewed / edge-on for data ranges far from the origin.
            position: Vec3::new(center_x, center_y, 1.0),
            target: Vec3::new(center_x, center_y, 0.0),
            // For 2D views we look down -Z; keep a stable screen-up (+Y).
            up: Vec3::Y,
            projection: ProjectionType::Orthographic {
                left,
                right,
                bottom,
                top,
                near: -1.0,
                far: 1.0,
            },
            aspect_ratio: (right - left) / (top - bottom),
            depth_mode: DepthMode::default(),
            zoom: 1.0,
            rotation: Quat::IDENTITY,
            pan_sensitivity: 0.01,
            zoom_sensitivity: 0.1,
            rotate_sensitivity: 0.0, // Disable rotation for 2D
            axis_scale: Vec3::ONE,
            view_matrix: Mat4::IDENTITY,
            projection_matrix: Mat4::IDENTITY,
            view_proj_dirty: true,
        };
        camera.update_matrices();
        camera
    }

    /// Update aspect ratio (call when window resizes)
    pub fn update_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.aspect_ratio = aspect_ratio;
        self.view_proj_dirty = true;
    }

    /// Get the view-projection matrix
    pub fn view_proj_matrix(&mut self) -> Mat4 {
        if self.view_proj_dirty {
            self.update_matrices();
        }
        self.projection_matrix * self.view_matrix * Mat4::from_scale(self.axis_scale)
    }

    /// Mark the camera matrices as dirty (call after manually modifying projection)
    pub fn mark_dirty(&mut self) {
        self.view_proj_dirty = true;
    }

    /// Get the view matrix
    pub fn view_matrix(&mut self) -> Mat4 {
        if self.view_proj_dirty {
            self.update_matrices();
        }
        self.view_matrix
    }

    /// Get the projection matrix
    pub fn projection_matrix(&mut self) -> Mat4 {
        if self.view_proj_dirty {
            self.update_matrices();
        }
        self.projection_matrix
    }

    /// Pan the camera (screen-space movement)
    pub fn pan(&mut self, delta: Vec2) {
        // Ensure view axes are up-to-date before using them.
        let view = self.view_matrix();
        let right = view.x_axis.truncate();
        let up = view.y_axis.truncate();

        // Scale pan by camera distance in 3D so it feels consistent while zooming.
        let dist = (self.position - self.target).length().max(1e-3);
        // CAD-style "grab": dragging right moves the scene right (camera left).
        // Screen Y increases downward, so we keep the Y sign (drag down pans down).
        let delta = Vec2::new(-delta.x, delta.y);
        let pan_amount = delta * self.pan_sensitivity * dist;
        let world_delta = right * pan_amount.x + up * pan_amount.y;

        self.position += world_delta;
        self.target += world_delta;
        self.view_proj_dirty = true;
    }

    /// Zoom the camera (positive = zoom in, negative = zoom out)
    pub fn zoom(&mut self, delta: f32) {
        // Convert wheel delta into multiplicative zoom with deadzone and clamping
        let mut factor = 1.0 - delta * self.zoom_sensitivity;
        if factor.abs() < 1e-3 {
            return;
        }
        factor = factor.clamp(0.2, 5.0);
        self.zoom = (self.zoom * factor).clamp(0.01, 100.0);

        match &mut self.projection {
            ProjectionType::Perspective { .. } => {
                // For perspective, dolly camera closer/farther to target.
                let delta_vec = self.position - self.target;
                let distance = delta_vec.length();
                if !distance.is_finite() || distance < 1e-4 {
                    // Avoid NaNs from normalizing a zero-length vector (which would make the scene vanish).
                    return;
                }
                let direction = delta_vec / distance;
                let new_distance = (distance * factor).clamp(0.1, 1000.0);
                self.position = self.target + direction * new_distance;
            }
            ProjectionType::Orthographic {
                left,
                right,
                bottom,
                top,
                ..
            } => {
                // For orthographic, scale the view bounds
                let center_x = (*left + *right) / 2.0;
                let center_y = (*bottom + *top) / 2.0;
                let width = (*right - *left) * factor;
                let height = (*top - *bottom) * factor;

                *left = center_x - width / 2.0;
                *right = center_x + width / 2.0;
                *bottom = center_y - height / 2.0;
                *top = center_y + height / 2.0;
            }
        }

        self.view_proj_dirty = true;
    }

    /// Rotate the camera around the target (for 3D)
    pub fn rotate(&mut self, delta: Vec2) {
        if self.rotate_sensitivity == 0.0 {
            return; // Rotation disabled (e.g., for 2D mode)
        }

        // Orbit-like controls:
        // - yaw around world up
        // - pitch around camera right
        //
        // This feels closer to typical 3D viewport / game-camera interaction than pitching around
        // a fixed world X axis.
        let yaw = -delta.x * self.rotate_sensitivity;
        let pitch = -delta.y * self.rotate_sensitivity;

        // Keep orbit constraints aligned to true world-up (+Z) even if the camera
        // is rolled for a CAD-like default orientation.
        let world_up = Vec3::Z;
        let mut offset = self.position - self.target;
        if offset.length_squared() < 1e-9 {
            offset = Vec3::new(0.0, 0.0, 1.0);
        }

        // Yaw around world up.
        let yaw_rot = Quat::from_axis_angle(world_up, yaw);
        offset = yaw_rot * offset;

        // Pitch around camera right axis after yaw.
        let forward = (-offset).normalize_or_zero();
        let right = forward.cross(world_up).normalize_or_zero();
        if right.length_squared() > 1e-9 {
            let pitch_rot = Quat::from_axis_angle(right, pitch);
            let candidate = pitch_rot * offset;
            // Avoid flipping over the poles (when looking straight up/down).
            let up_dot = candidate.normalize_or_zero().dot(world_up).abs();
            if up_dot < 0.995 {
                offset = candidate;
            }
        }

        self.position = self.target + offset;

        self.view_proj_dirty = true;
    }

    /// Set camera to look at a specific target
    pub fn look_at(&mut self, target: Vec3, distance: Option<f32>) {
        self.target = target;

        if let Some(dist) = distance {
            let direction = (self.position - self.target).normalize();
            self.position = self.target + direction * dist;
        }

        self.view_proj_dirty = true;
    }

    /// The unit direction from the target to the viewpoint of MATLAB's `view(az, el)`: azimuth
    /// is measured from the negative y-axis, counterclockwise seen from +z, so `view(0, 0)`
    /// looks along +y, `view(90, 0)` along -x, and the default `view(-37.5, 30)` from -x, -y.
    pub fn view_direction_deg(azimuth_deg: f32, elevation_deg: f32) -> Vec3 {
        let az = azimuth_deg.to_radians();
        let el = elevation_deg.to_radians();
        Vec3::new(el.cos() * az.sin(), -el.cos() * az.cos(), el.sin())
    }

    pub fn set_view_angles_deg(&mut self, azimuth_deg: f32, elevation_deg: f32) {
        let distance = (self.position - self.target).length().max(0.1);
        let dir = Self::view_direction_deg(azimuth_deg, elevation_deg);
        self.up = Vec3::Z;
        self.position = self.target + dir * distance;
        self.view_proj_dirty = true;
    }

    /// Reset camera to default position
    pub fn reset(&mut self) {
        match self.projection {
            ProjectionType::Perspective { .. } => {
                self.position = Vec3::new(3.5, 3.5, 3.5);
                self.target = Vec3::ZERO;
                self.rotation = Quat::IDENTITY;
                self.up = Vec3::Z;
            }
            ProjectionType::Orthographic { .. } => {
                self.zoom = 1.0;
                self.target = Vec3::ZERO;
            }
        }
        self.view_proj_dirty = true;
    }

    /// Fit the camera to show all data within the given bounds
    pub fn fit_bounds(&mut self, min_bounds: Vec3, max_bounds: Vec3) {
        // Fit in the camera's (axis-scaled) view space.
        let (min_bounds, max_bounds) = (min_bounds * self.axis_scale, max_bounds * self.axis_scale);
        let center = (min_bounds + max_bounds) / 2.0;
        let size = max_bounds - min_bounds;

        match &mut self.projection {
            ProjectionType::Perspective { fov, near, far } => {
                // Keep the current viewing direction; read it before moving the target.
                let direction = match (self.position - self.target).normalize_or_zero() {
                    d if d == Vec3::ZERO => Vec3::new(1.0, -1.0, 1.0).normalize(),
                    d => d,
                };
                self.target = center;
                // Back off just far enough that all eight box corners project inside the
                // view with a small margin, so the plot fills its axes like MATLAB's.
                let up = if direction.cross(self.up).length_squared() > 1.0e-8 {
                    self.up
                } else {
                    Vec3::Z
                };
                let right = up.cross(direction).normalize();
                let true_up = direction.cross(right).normalize();
                let fill = 0.92;
                let tan_y = (*fov * 0.5).tan() * fill;
                let tan_x = tan_y * self.aspect_ratio.max(1.0e-3);
                let half = size * 0.5;
                let mut distance: f32 = 1.0e-3;
                for i in 0..8 {
                    let corner = Vec3::new(
                        if i & 1 == 0 { -half.x } else { half.x },
                        if i & 2 == 0 { -half.y } else { half.y },
                        if i & 4 == 0 { -half.z } else { half.z },
                    );
                    // Depth toward the camera, and offsets across the view.
                    let toward = corner.dot(direction);
                    let u = corner.dot(right).abs();
                    let v = corner.dot(true_up).abs();
                    distance = distance.max(toward + u / tan_x).max(toward + v / tan_y);
                }
                if !distance.is_finite() {
                    distance = size.x.max(size.y).max(size.z) * 2.0;
                }
                self.position = self.target + direction * distance;

                // Keep clip planes sane relative to the new view distance.
                // Animated surfaces can have very large Z ranges; if `far` is too small,
                // everything gets clipped and the plot appears to "clear".
                let radius = (size.length() * 0.5).max(1e-3);
                let dist = (self.position - self.target).length().max(1e-3);
                let desired_near = (dist - radius * 4.0).max(0.01);
                let desired_far = (dist + radius * 4.0).max(desired_near + 1.0);
                *near = desired_near;
                *far = desired_far;
            }
            ProjectionType::Orthographic {
                left,
                right,
                bottom,
                top,
                ..
            } => {
                let margin = 0.1; // 10% margin
                let width = size.x * (1.0 + margin);
                let height = size.y * (1.0 + margin);

                // Maintain aspect ratio
                let display_width = width.max(height * self.aspect_ratio);
                let display_height = height.max(width / self.aspect_ratio);

                *left = center.x - display_width / 2.0;
                *right = center.x + display_width / 2.0;
                *bottom = center.y - display_height / 2.0;
                *top = center.y + display_height / 2.0;

                self.target = center;
            }
        }

        self.view_proj_dirty = true;
    }

    /// Fit a Figure's perspective camera to every displayed box corner at its plot-rect aspect.
    /// This deliberately does not alter the legacy geometry-scene `fit_bounds` policy.
    pub fn fit_figure_perspective_bounds(
        &mut self,
        min_bounds: Vec3,
        max_bounds: Vec3,
        viewport: (u32, u32),
    ) -> bool {
        let ProjectionType::Perspective { fov, .. } = self.projection else {
            return false;
        };
        // The view-projection matrix scales data coordinates before applying the
        // camera. Fit in that same space, including the target and clip planes.
        let min_bounds = min_bounds * self.axis_scale;
        let max_bounds = max_bounds * self.axis_scale;
        if !min_bounds.is_finite()
            || !max_bounds.is_finite()
            || min_bounds.cmpgt(max_bounds).any()
            || viewport.0 == 0
            || viewport.1 == 0
        {
            return false;
        }
        let center = (min_bounds + max_bounds) * 0.5;
        let direction = (self.position - self.target).normalize_or_zero();
        let forward = -direction;
        let right = forward.cross(self.up).normalize_or_zero();
        let screen_up = right.cross(forward).normalize_or_zero();
        if direction.length_squared() < 0.5
            || right.length_squared() < 0.5
            || !fov.is_finite()
            || fov <= 0.0
            || fov >= std::f32::consts::PI
        {
            return false;
        }
        let aspect = viewport.0 as f32 / viewport.1 as f32;
        let tangent = (fov * 0.5).tan();
        let mut distance = 0.0_f32;
        let mut closest = f32::NEG_INFINITY;
        let mut farthest = f32::INFINITY;
        for x in [min_bounds.x, max_bounds.x] {
            for y in [min_bounds.y, max_bounds.y] {
                for z in [min_bounds.z, max_bounds.z] {
                    let offset = Vec3::new(x, y, z) - center;
                    let toward_eye = offset.dot(direction);
                    closest = closest.max(toward_eye);
                    farthest = farthest.min(toward_eye);
                    let horizontal = offset.dot(right).abs() / (tangent * aspect);
                    let vertical = offset.dot(screen_up).abs() / tangent;
                    distance = distance.max(toward_eye + horizontal.max(vertical));
                }
            }
        }
        let radius = (max_bounds - min_bounds).length() * 0.5;
        distance = distance.max(closest + radius * 0.01).max(1.0e-3) * 1.1;
        if !distance.is_finite() {
            return false;
        }
        self.target = center;
        self.position = center + direction * distance;
        self.aspect_ratio = aspect;
        if let ProjectionType::Perspective { near, far, .. } = &mut self.projection {
            *near = (distance - closest).max(1.0e-3) * 0.5;
            *far = ((distance - farthest) * 1.5).max(*near + 1.0);
        }
        self.view_proj_dirty = true;
        true
    }

    pub fn set_clip_planes(&mut self, near: f32, far: f32) {
        match &mut self.projection {
            ProjectionType::Perspective {
                near: n, far: f, ..
            } => {
                *n = near.max(1e-4);
                *f = far.max(*n + 1e-3);
                self.view_proj_dirty = true;
            }
            ProjectionType::Orthographic {
                near: n, far: f, ..
            } => {
                *n = near;
                *f = far;
                self.view_proj_dirty = true;
            }
        }
    }

    /// Update near/far clip planes from a world-space AABB using a policy.
    ///
    /// This is meant to be run per-frame for 3D (CAD-like) robustness: keep near as large as
    /// possible without slicing the scene, and keep far just large enough to contain it.
    pub fn update_clip_planes_from_world_aabb(
        &mut self,
        world_min: Vec3,
        world_max: Vec3,
        policy: &ClipPolicy,
    ) {
        if !policy.dynamic {
            return;
        }
        let ProjectionType::Perspective { .. } = self.projection else {
            return;
        };

        // View space from current pose (independent of projection), after the axis scale.
        let view = Mat4::look_at_rh(self.position, self.target, self.up)
            * Mat4::from_scale(self.axis_scale);
        let corners = [
            Vec3::new(world_min.x, world_min.y, world_min.z),
            Vec3::new(world_max.x, world_min.y, world_min.z),
            Vec3::new(world_min.x, world_max.y, world_min.z),
            Vec3::new(world_max.x, world_max.y, world_min.z),
            Vec3::new(world_min.x, world_min.y, world_max.z),
            Vec3::new(world_max.x, world_min.y, world_max.z),
            Vec3::new(world_min.x, world_max.y, world_max.z),
            Vec3::new(world_max.x, world_max.y, world_max.z),
        ];

        let mut min_depth = f32::INFINITY;
        let mut max_depth = f32::NEG_INFINITY;
        for c in corners {
            let v = (view * c.extend(1.0)).truncate();
            if !(v.x.is_finite() && v.y.is_finite() && v.z.is_finite()) {
                continue;
            }
            // RH look-at: camera looks down -Z; points in front have negative z.
            let depth = (-v.z).max(0.0);
            if depth > 0.0 {
                min_depth = min_depth.min(depth);
                max_depth = max_depth.max(depth);
            }
        }
        if !min_depth.is_finite() || !max_depth.is_finite() || max_depth <= 0.0 {
            return;
        }

        let mut near = (min_depth * policy.near_padding).max(policy.min_near);
        let mut far = (max_depth * policy.far_padding).max(near + 1.0);
        if far > policy.max_far {
            far = policy.max_far.max(near + 1.0);
        }
        // Keep a modest ratio when possible to preserve precision (avoid near->0).
        if (far / near).is_finite() && far / near > 1.0e6 {
            near = (far / 1.0e6).max(policy.min_near);
        }
        self.set_clip_planes(near, far);
    }

    /// Unproject to data coordinates, including inverse axis scaling, for picking.
    pub fn screen_to_world(&mut self, screen_pos: Vec2, screen_size: Vec2, depth: f32) -> Vec3 {
        self.screen_to_render(screen_pos, screen_size, depth) / self.axis_scale
    }

    /// Unproject to the axis-scaled coordinates used by camera positions and targets.
    pub fn screen_to_render(&mut self, screen_pos: Vec2, screen_size: Vec2, depth: f32) -> Vec3 {
        if self.view_proj_dirty {
            self.update_matrices();
        }
        // Convert screen coordinates to normalized device coordinates
        let ndc_x = (2.0 * screen_pos.x) / screen_size.x - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_pos.y) / screen_size.y;
        let ndc = Vec3::new(ndc_x, ndc_y, depth * 2.0 - 1.0);

        // Unproject to world coordinates
        let view_proj_inv = (self.projection_matrix * self.view_matrix).inverse();
        let world_pos = view_proj_inv * ndc.extend(1.0);

        if world_pos.w != 0.0 {
            world_pos.truncate() / world_pos.w
        } else {
            world_pos.truncate()
        }
    }

    /// Reset a perspective view around a data-space center, preserving render-space distance.
    pub(crate) fn reset_perspective_orientation(&mut self, data_center: Option<Vec3>) {
        let center = data_center.map_or(self.target, |point| point * self.axis_scale);
        let distance = self.position.distance(self.target).max(0.1);
        self.target = center;
        self.up = Vec3::Z;
        self.position = center + Vec3::new(1.0, -1.0, 1.0).normalize() * distance;
        self.mark_dirty();
    }

    /// Update the view and projection matrices
    fn update_matrices(&mut self) {
        // Update view matrix
        self.view_matrix = Mat4::look_at_rh(self.position, self.target, self.up);

        // Update projection matrix
        self.projection_matrix = match self.projection {
            ProjectionType::Perspective { fov, near, far } => {
                match self.depth_mode {
                    DepthMode::Standard => Mat4::perspective_rh(fov, self.aspect_ratio, near, far),
                    DepthMode::ReversedZ => {
                        // Right-handed, depth range 0..1, reversed-Z mapping.
                        // This keeps the same clip volume but flips the depth distribution so
                        // far distances retain more precision (game-engine style).
                        let f = 1.0 / (0.5 * fov).tan();
                        let a = self.aspect_ratio.max(1e-6);
                        let nf = (far - near).max(1e-6);
                        let m00 = f / a;
                        let m11 = f;
                        // Reversed-Z: near->1, far->0
                        let m22 = near / nf;
                        let m32 = (near * far) / nf;
                        Mat4::from_cols_array(&[
                            m00, 0.0, 0.0, 0.0, //
                            0.0, m11, 0.0, 0.0, //
                            0.0, 0.0, m22, -1.0, //
                            0.0, 0.0, m32, 0.0, //
                        ])
                    }
                }
            }
            ProjectionType::Orthographic {
                left,
                right,
                bottom,
                top,
                near,
                far,
            } => {
                log::trace!(
                    target: "runmat_plot",
                    "ortho matrix bounds l={} r={} b={} t={} n={} f={}",
                    left, right, bottom, top, near, far
                );
                log::trace!(target: "runmat_plot", "camera aspect_ratio={}", self.aspect_ratio);
                Mat4::orthographic_rh(left, right, bottom, top, near, far)
            }
        };

        self.view_proj_dirty = false;
    }
}

/// Camera controller for handling input events
#[derive(Debug, Default)]
pub struct CameraController {
    pub active_button: Option<MouseButton>,
    pub last_mouse_pos: Vec2,
    pub mouse_delta: Vec2,
    navigation: NavigationOptions,
    cursor_orbit_anchor: Option<Vec3>,
    drag_orbit_pivot: Option<Vec3>,
}

impl CameraController {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn navigation_options(&self) -> NavigationOptions {
        self.navigation
    }

    pub fn set_navigation_options(
        &mut self,
        options: NavigationOptions,
    ) -> Result<(), &'static str> {
        options.validate()?;
        self.navigation = options;
        self.active_button = None;
        self.cursor_orbit_anchor = None;
        self.drag_orbit_pivot = None;
        Ok(())
    }

    /// Supply an exact host-picked point in data coordinates for under-cursor orbit. If absent, use
    /// the cursor ray's intersection with the view-target plane, not the origin.
    pub fn set_cursor_orbit_anchor(&mut self, anchor: Option<Vec3>) -> Result<(), &'static str> {
        if anchor.is_some_and(|point| !point.is_finite()) {
            return Err("orbit anchor must be finite");
        }
        self.cursor_orbit_anchor = anchor;
        self.drag_orbit_pivot = None;
        Ok(())
    }

    fn orbit(&mut self, camera: &mut Camera, delta: Vec2, position: Vec2, viewport: (u32, u32)) {
        let pivot = self
            .drag_orbit_pivot
            .unwrap_or_else(|| match self.navigation.orbit_center {
                OrbitCenter::ViewCenter => camera.target,
                OrbitCenter::Origin => Vec3::ZERO,
                OrbitCenter::UnderCursor => self
                    .cursor_orbit_anchor
                    .map(|point| point * camera.axis_scale)
                    .unwrap_or_else(|| {
                        let size = Vec2::new(viewport.0.max(1) as f32, viewport.1.max(1) as f32);
                        let near = camera.screen_to_render(position, size, 0.0);
                        let far = camera.screen_to_render(position, size, 1.0);
                        let direction = (far - near).normalize_or_zero();
                        let normal = (camera.target - camera.position).normalize_or_zero();
                        let denominator = direction.dot(normal);
                        if denominator.abs() < 1e-6 {
                            return camera.target;
                        }
                        let distance = (camera.target - near).dot(normal) / denominator;
                        let point = near + direction * distance;
                        if point.is_finite() && distance >= 0.0 {
                            point
                        } else {
                            camera.target
                        }
                    }),
            });
        self.drag_orbit_pivot = Some(pivot);
        let target = camera.target;
        let before = camera.position - pivot;
        camera.target = pivot;
        camera.rotate(delta);
        // Rotate the look target with the eye so a non-central pivot stays at
        // its screen position instead of snapping the view to the cursor/origin.
        let after = camera.position - pivot;
        if before.length_squared() > 1e-9 && after.length_squared() > 1e-9 {
            let rotation = Quat::from_rotation_arc(before.normalize(), after.normalize());
            camera.target = pivot + rotation * (target - pivot);
            if self.navigation.orbit_center != OrbitCenter::ViewCenter {
                camera.up = (rotation * camera.up).normalize_or_zero();
            }
            camera.mark_dirty();
        }
    }

    /// Two-finger scroll pans when enabled; Ctrl/Cmd pinch remains zoom. The
    /// host must identify trackpad scroll separately from a detented wheel.
    pub fn trackpad_scroll(
        &mut self,
        delta: Vec2,
        position: Vec2,
        viewport: (u32, u32),
        mut modifiers: Modifiers,
        camera: &mut Camera,
    ) {
        if self.navigation.trackpad_pan && !modifiers.ctrl && !modifiers.meta {
            modifiers.shift = true;
        }
        self.mouse_wheel(delta, position, viewport, modifiers, camera);
    }

    /// Handle mouse press
    pub fn mouse_press(&mut self, position: Vec2, button: MouseButton, _modifiers: Modifiers) {
        self.last_mouse_pos = position;
        self.active_button = Some(button);
        self.drag_orbit_pivot = None;
    }

    /// Handle mouse release
    pub fn mouse_release(&mut self, _position: Vec2, button: MouseButton, _modifiers: Modifiers) {
        if self.active_button == Some(button) {
            self.active_button = None;
            self.drag_orbit_pivot = None;
        }
    }

    /// Handle mouse movement
    ///
    /// For 3D (perspective) cameras:
    /// - left drag: orbit/rotate
    /// - right drag: pan
    ///
    /// For 2D (orthographic) cameras, we treat drag as pan by shifting the
    /// orthographic bounds in data-space. We avoid translating the view matrix in X/Y
    /// because the ortho bounds already live in data coordinates.
    pub fn mouse_move(
        &mut self,
        position: Vec2,
        delta: Vec2,
        viewport_px: (u32, u32),
        modifiers: Modifiers,
        camera: &mut Camera,
    ) {
        let Some(button) = self.active_button else {
            self.last_mouse_pos = position;
            return;
        };
        // Prefer the host-provided delta; fall back to position diff.
        self.mouse_delta = if delta.length_squared() > 0.0 {
            delta
        } else {
            position - self.last_mouse_pos
        };

        match camera.projection {
            ProjectionType::Perspective { .. } => {
                // CAD-like bindings (support common schemes simultaneously):
                // - MMB drag: orbit; Shift+MMB: pan
                // - RMB drag: orbit; Shift+RMB: pan
                // - Alt+LMB: orbit; Alt+MMB: pan; Alt+RMB: dolly/zoom
                //
                // Also keep LMB orbit + Shift+LMB pan as a convenient fallback.
                let fine = if modifiers.ctrl || modifiers.meta {
                    0.35
                } else {
                    1.0
                };
                let d = self.mouse_delta * fine;

                if self.navigation.preset != NavigationPreset::Legacy {
                    let orbit = match self.navigation.preset {
                        NavigationPreset::Cad => button == MouseButton::Middle && !modifiers.shift,
                        NavigationPreset::Dcc => modifiers.alt && button == MouseButton::Left,
                        NavigationPreset::Legacy => unreachable!(),
                    };
                    let pan = match self.navigation.preset {
                        NavigationPreset::Cad => {
                            button == MouseButton::Right
                                || (button == MouseButton::Middle && modifiers.shift)
                        }
                        NavigationPreset::Dcc => modifiers.alt && button == MouseButton::Middle,
                        NavigationPreset::Legacy => unreachable!(),
                    };
                    if orbit {
                        self.orbit(camera, d, self.last_mouse_pos, viewport_px);
                    } else if pan {
                        camera.pan(d);
                    } else if self.navigation.preset == NavigationPreset::Dcc
                        && modifiers.alt
                        && button == MouseButton::Right
                    {
                        self.mouse_wheel(
                            Vec2::new(0.0, (-d.y / 120.0).clamp(-5.0, 5.0)),
                            position,
                            viewport_px,
                            modifiers,
                            camera,
                        );
                    }
                    self.last_mouse_pos = position;
                    return;
                }

                if modifiers.alt {
                    match button {
                        MouseButton::Left => {
                            self.orbit(camera, d, self.last_mouse_pos, viewport_px)
                        }
                        MouseButton::Middle => camera.pan(d),
                        MouseButton::Right => {
                            // Alt+RMB drag zoom (dolly). Positive drag up should zoom in.
                            let zoom_delta = (-d.y / 120.0).clamp(-5.0, 5.0);
                            self.mouse_wheel(
                                Vec2::new(0.0, zoom_delta),
                                position,
                                viewport_px,
                                modifiers,
                                camera,
                            );
                        }
                    }
                } else {
                    let want_pan = modifiers.shift;
                    match button {
                        MouseButton::Middle | MouseButton::Right => {
                            if want_pan {
                                camera.pan(d);
                            } else if modifiers.ctrl || modifiers.meta {
                                // Ctrl/Cmd + (MMB/RMB) drag: zoom/dolly (very common in CAD/DCC).
                                let zoom_delta = (-d.y / 120.0).clamp(-5.0, 5.0);
                                self.mouse_wheel(
                                    Vec2::new(0.0, zoom_delta),
                                    position,
                                    viewport_px,
                                    modifiers,
                                    camera,
                                );
                            } else {
                                self.orbit(camera, d, self.last_mouse_pos, viewport_px);
                            }
                        }
                        MouseButton::Left => {
                            if want_pan {
                                camera.pan(d);
                            } else if modifiers.ctrl || modifiers.meta {
                                let zoom_delta = (-d.y / 120.0).clamp(-5.0, 5.0);
                                self.mouse_wheel(
                                    Vec2::new(0.0, zoom_delta),
                                    position,
                                    viewport_px,
                                    modifiers,
                                    camera,
                                );
                            } else {
                                self.orbit(camera, d, self.last_mouse_pos, viewport_px);
                            }
                        }
                    }
                }
            }
            ProjectionType::Orthographic {
                ref mut left,
                ref mut right,
                ref mut bottom,
                ref mut top,
                ..
            } => {
                // For 2D, treat any drag (and Shift-drag) as panning the ortho bounds.
                let _ = (button, modifiers);
                {
                    let (vw, vh) = (viewport_px.0.max(1) as f32, viewport_px.1.max(1) as f32);
                    let width = (*right - *left).abs().max(1e-6);
                    let height = (*top - *bottom).abs().max(1e-6);

                    // Convert pixel delta to data-space delta.
                    // Screen +X should move the view right; dragging right should move the data left,
                    // so we subtract.
                    let dx = -self.mouse_delta.x * (width / vw);
                    // Screen +Y is down in most DOM coordinate systems; dragging down should move
                    // the data up, so we add.
                    let dy = self.mouse_delta.y * (height / vh);

                    *left += dx;
                    *right += dx;
                    *bottom += dy;
                    *top += dy;
                    camera.mark_dirty();
                }
            }
        }

        self.last_mouse_pos = position;
    }

    /// Handle mouse wheel
    pub fn mouse_wheel(
        &mut self,
        delta: Vec2,
        position_px: Vec2,
        viewport_px: (u32, u32),
        modifiers: Modifiers,
        camera: &mut Camera,
    ) {
        // CAD-ish wheel semantics:
        // - default: zoom/dolly to cursor using vertical wheel component
        // - Shift: pan (screen-space) using both wheel components
        //
        // Don't treat Ctrl/Cmd as "fine wheel" because macOS trackpad pinch-to-zoom gestures
        // report Ctrl as pressed. Keeping wheel zoom consistent feels more natural.
        let delta_y = delta.y
            * self.navigation.zoom_speed
            * if self.navigation.reverse_zoom {
                -1.0
            } else {
                1.0
            };
        let position_px = if self.navigation.zoom_to_cursor {
            position_px
        } else {
            Vec2::new(viewport_px.0 as f32, viewport_px.1 as f32) * 0.5
        };

        match &mut camera.projection {
            ProjectionType::Perspective { .. } => {
                if modifiers.shift {
                    // Wheel-pan in the view plane. Scale by distance for a consistent feel.
                    // Positive wheel deltas should pan "with" the gesture (down scroll moves view down).
                    // NOTE: `Camera::pan` already scales by camera distance; don't multiply by distance here.
                    let pan_px = Vec2::new(delta.x, -delta.y);
                    camera.pan(pan_px * 6.0);
                    return;
                }

                let sens = camera.zoom_sensitivity;
                let mut factor = 1.0 - delta_y * sens;
                if factor.abs() < 1e-3 {
                    return;
                }
                factor = factor.clamp(0.2, 5.0);

                let (vw, vh) = (viewport_px.0.max(1) as f32, viewport_px.1.max(1) as f32);
                let screen_size = Vec2::new(vw, vh);
                let pos = Vec2::new(position_px.x.clamp(0.0, vw), position_px.y.clamp(0.0, vh));

                // Build a ray from the cursor through the view frustum.
                let p_near = camera.screen_to_render(pos, screen_size, 0.0);
                let p_far = camera.screen_to_render(pos, screen_size, 1.0);
                let dir = (p_far - p_near).normalize_or_zero();
                if dir.length_squared() < 1e-9 {
                    return;
                }

                // Anchor zoom to a plane through the current target and perpendicular to the view
                // direction. A global Z=0 plane feels natural for flat plots, but CAD models can
                // sit anywhere in model space; using the view target keeps close-range wheel zoom
                // responsive and cursor-local.
                let origin = camera.position;
                let mut pivot = None;
                let forward = (camera.target - camera.position).normalize_or_zero();
                if forward.length_squared() > 1e-9 {
                    let denom = dir.dot(forward);
                    if denom.abs() > 1e-6 {
                        let t = (camera.target - origin).dot(forward) / denom;
                        if t.is_finite() && t > 0.0 {
                            pivot = Some(origin + dir * t);
                        }
                    }
                }
                if pivot.is_none() && dir.z.abs() > 1e-6 {
                    let t = (-origin.z) / dir.z;
                    if t.is_finite() && t > 0.0 {
                        pivot = Some(origin + dir * t);
                    }
                }
                let pivot = pivot.unwrap_or(camera.target);

                let s = (pivot - origin).length().max(1e-3);
                let new_s = (s * factor).clamp(0.005, 1.0e9);
                let delta_dist = s - new_s;
                let translate = dir * delta_dist;

                // Dolly along the cursor ray while keeping orientation stable (translate both
                // position + target so the cursor stays anchored).
                camera.position += translate;
                camera.target += translate;
                camera.view_proj_dirty = true;
            }
            ProjectionType::Orthographic {
                left,
                right,
                bottom,
                top,
                ..
            } => {
                if modifiers.shift {
                    // Wheel-pan in 2D (treat wheel deltas as pixel-ish movement).
                    let vw = viewport_px.0.max(1) as f32;
                    let vh = viewport_px.1.max(1) as f32;
                    let w = (*right - *left).max(1e-6);
                    let h = (*top - *bottom).max(1e-6);
                    let dx = -delta.x * (w / vw);
                    let dy = delta.y * (h / vh);
                    *left += dx;
                    *right += dx;
                    *bottom += dy;
                    *top += dy;
                    camera.mark_dirty();
                    return;
                }

                let sens = camera.zoom_sensitivity;
                let mut factor = 1.0 - delta_y * sens;
                if factor.abs() < 1e-3 {
                    return;
                }
                factor = factor.clamp(0.2, 5.0);

                // Cursor-anchored 2D zoom: scale the ortho bounds around the cursor.
                let w = (*right - *left).max(1e-6);
                let h = (*top - *bottom).max(1e-6);
                let vw = viewport_px.0.max(1) as f32;
                let vh = viewport_px.1.max(1) as f32;
                let tx = (position_px.x / vw).clamp(0.0, 1.0);
                let ty = (position_px.y / vh).clamp(0.0, 1.0);
                let pivot_x = *left + tx * w;
                let pivot_y = *top - ty * h;
                let new_left = pivot_x - (pivot_x - *left) * factor;
                let new_right = pivot_x + (*right - pivot_x) * factor;
                let new_bottom = pivot_y - (pivot_y - *bottom) * factor;
                let new_top = pivot_y + (*top - pivot_y) * factor;
                *left = new_left;
                *right = new_right;
                *bottom = new_bottom;
                *top = new_top;
                camera.mark_dirty();
            }
        }
    }
}

/// Mouse button enum for camera control
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

#[cfg(test)]
mod tests {
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
        assert!((offset.x - 2.0).abs() < 1e-4);
        assert!(offset.y.abs() < 1e-4);
    }

    #[test]
    fn view_angles_follow_matlab_azimuth_convention() {
        let close = |a: Vec3, b: Vec3| (a - b).length() < 1.0e-5;
        assert!(close(Camera::view_direction_deg(0.0, 0.0), Vec3::new(0.0, -1.0, 0.0)));
        assert!(close(Camera::view_direction_deg(90.0, 0.0), Vec3::new(1.0, 0.0, 0.0)));
        assert!(close(Camera::view_direction_deg(0.0, 90.0), Vec3::Z));
        let default = Camera::view_direction_deg(-37.5, 30.0);
        assert!(default.x < 0.0 && default.y < 0.0 && default.z > 0.0);
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
}
