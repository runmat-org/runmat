//! Camera system for 3D navigation and 2D plotting
//!
//! Provides both perspective and orthographic cameras with smooth
//! navigation controls for interactive plotting.

mod controller;
#[cfg(test)]
mod tests;
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

    /// A 3-D view drawn as a parallel (orthographic) projection: the perspective camera's
    /// position, target and field of view still define the view, and the visible height is the
    /// perspective view's height at the target (`2 · distance · tan(fov / 2)`), so orbit, zoom
    /// (dolly), fit and standard views keep working and switching modes keeps the framing.
    /// The 2-D plot camera (`ProjectionType::Orthographic`) is separate.
    pub parallel: bool,

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
            parallel: false,
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
            parallel: false,
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
                    // Depth toward the camera, and offsets across the view. A parallel view's
                    // image doesn't grow with depth, so only the offsets count.
                    let toward = if self.parallel {
                        0.0
                    } else {
                        corner.dot(direction)
                    };
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

    /// Unproject to the axis-scaled coordinates used by camera positions and targets. `depth`
    /// runs from the near plane (0) to the far plane (1) in either depth mode, so a ray from
    /// depth 0 to depth 1 points away from the eye.
    pub fn screen_to_render(&mut self, screen_pos: Vec2, screen_size: Vec2, depth: f32) -> Vec3 {
        if self.view_proj_dirty {
            self.update_matrices();
        }
        // Convert screen coordinates to normalized device coordinates
        let ndc_x = (2.0 * screen_pos.x) / screen_size.x - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_pos.y) / screen_size.y;
        // The projections map depth to 0..1 (near to far, or far to near under reversed-Z).
        let ndc_z = match self.depth_mode {
            DepthMode::Standard => depth,
            DepthMode::ReversedZ => 1.0 - depth,
        };
        let ndc = Vec3::new(ndc_x, ndc_y, ndc_z);

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
            ProjectionType::Perspective { fov, far, .. } if self.parallel => {
                let distance = (self.position - self.target).length().max(1.0e-6);
                let half_height = distance * (0.5 * fov).tan();
                let half_width = half_height * self.aspect_ratio.max(1e-6);
                // Geometry behind the eye still draws (a view zoomed into a part), so the box
                // runs from `far` behind the eye to `far` in front of it.
                let far = far.max(distance * 2.0);
                let (near, far) = match self.depth_mode {
                    DepthMode::Standard => (-far, far),
                    // Reversed-Z: near->1, far->0.
                    DepthMode::ReversedZ => (far, -far),
                };
                Mat4::orthographic_rh(
                    -half_width,
                    half_width,
                    -half_height,
                    half_height,
                    near,
                    far,
                )
            }
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

#[allow(unused_imports)]
pub use controller::*;
