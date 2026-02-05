//! Camera system for 3D navigation and 2D plotting
//!
//! Provides both perspective and orthographic cameras with smooth
//! navigation controls for interactive plotting.

use crate::core::interaction::Modifiers;
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

    // Navigation state
    pub zoom: f32,
    pub rotation: Quat,

    // Interaction settings
    pub pan_sensitivity: f32,
    pub zoom_sensitivity: f32,
    pub rotate_sensitivity: f32,

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
            zoom: 1.0,
            rotation: Quat::IDENTITY,
            pan_sensitivity: 0.01,
            zoom_sensitivity: 0.1,
            rotate_sensitivity: 0.005,
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
        let mut camera = Self {
            position: Vec3::new(0.0, 0.0, 1.0),
            target: Vec3::new((left + right) / 2.0, (bottom + top) / 2.0, 0.0),
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
            zoom: 1.0,
            rotation: Quat::IDENTITY,
            pan_sensitivity: 0.01,
            zoom_sensitivity: 0.1,
            rotate_sensitivity: 0.0, // Disable rotation for 2D
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
        self.projection_matrix * self.view_matrix
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
        let center = (min_bounds + max_bounds) / 2.0;
        let size = max_bounds - min_bounds;

        match &mut self.projection {
            ProjectionType::Perspective { near, far, .. } => {
                let max_size = size.x.max(size.y).max(size.z);
                let distance = max_size * 2.0; // Ensure everything fits

                self.target = center;
                let direction = (self.position - self.target).normalize();
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

    /// Convert screen coordinates to world coordinates (for picking)
    pub fn screen_to_world(&mut self, screen_pos: Vec2, screen_size: Vec2, depth: f32) -> Vec3 {
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

    /// Update the view and projection matrices
    fn update_matrices(&mut self) {
        // Update view matrix
        self.view_matrix = Mat4::look_at_rh(self.position, self.target, self.up);

        // Update projection matrix
        self.projection_matrix = match self.projection {
            ProjectionType::Perspective { fov, near, far } => {
                Mat4::perspective_rh(fov, self.aspect_ratio, near, far)
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
}

impl CameraController {
    pub fn new() -> Self {
        Self::default()
    }

    /// Handle mouse press
    pub fn mouse_press(&mut self, position: Vec2, button: MouseButton, _modifiers: Modifiers) {
        self.last_mouse_pos = position;
        self.active_button = Some(button);
    }

    /// Handle mouse release
    pub fn mouse_release(&mut self, _position: Vec2, button: MouseButton, _modifiers: Modifiers) {
        if self.active_button == Some(button) {
            self.active_button = None;
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
                let fine = if modifiers.ctrl || modifiers.meta { 0.35 } else { 1.0 };
                let d = self.mouse_delta * fine;

                if modifiers.alt {
                    match button {
                        MouseButton::Left => camera.rotate(d),
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
                                camera.rotate(d);
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
                                camera.rotate(d);
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
        let delta_y = delta.y;

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
                let p_near = camera.screen_to_world(pos, screen_size, 0.0);
                let p_far = camera.screen_to_world(pos, screen_size, 1.0);
                let dir = (p_far - p_near).normalize_or_zero();
                if dir.length_squared() < 1e-9 {
                    return;
                }

                // Prefer anchoring to the XY plane (Z=0). If near-parallel, fall back to a plane
                // through the current target perpendicular to the view direction.
                let origin = camera.position;
                let mut pivot = None;
                if dir.z.abs() > 1e-6 {
                    let t = (-origin.z) / dir.z;
                    if t.is_finite() && t > 0.0 {
                        pivot = Some(origin + dir * t);
                    }
                }
                if pivot.is_none() {
                    let forward = (camera.target - camera.position).normalize_or_zero();
                    let denom = dir.dot(forward);
                    if denom.abs() > 1e-6 {
                        let t = (camera.target - origin).dot(forward) / denom;
                        if t.is_finite() && t > 0.0 {
                            pivot = Some(origin + dir * t);
                        }
                    }
                }
                let pivot = pivot.unwrap_or(camera.target);

                let s = (pivot - origin).length().max(1e-3);
                let new_s = (s * factor).clamp(0.05, 1.0e9);
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
    fn test_camera_creation() {
        let camera = Camera::new();
        assert_eq!(camera.position, Vec3::new(3.5, 3.5, 3.5));
        assert_eq!(camera.target, Vec3::ZERO);
        assert_eq!(camera.up, Vec3::Z);
    }

    #[test]
    fn test_2d_camera() {
        let camera = Camera::new_2d((-10.0, 10.0, -10.0, 10.0));
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
}
