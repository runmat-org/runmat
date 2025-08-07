//! Camera system for 3D navigation and 2D plotting
//!
//! Provides both perspective and orthographic cameras with smooth
//! navigation controls for interactive plotting.

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
            position: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
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
        let right = self.view_matrix.x_axis.truncate();
        let up = self.view_matrix.y_axis.truncate();

        let pan_amount = delta * self.pan_sensitivity * self.zoom;
        let world_delta = right * pan_amount.x + up * pan_amount.y;

        self.position += world_delta;
        self.target += world_delta;
        self.view_proj_dirty = true;
    }

    /// Zoom the camera (positive = zoom in, negative = zoom out)
    pub fn zoom(&mut self, delta: f32) {
        self.zoom *= 1.0 + delta * self.zoom_sensitivity;
        self.zoom = self.zoom.clamp(0.01, 100.0);

        match &mut self.projection {
            ProjectionType::Perspective { .. } => {
                // For perspective, move camera closer/farther
                let direction = (self.position - self.target).normalize();
                let distance = (self.position - self.target).length();
                let new_distance = distance * (1.0 + delta * self.zoom_sensitivity);
                self.position = self.target + direction * new_distance.clamp(0.1, 1000.0);
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
                let width = (*right - *left) * self.zoom;
                let height = (*top - *bottom) * self.zoom;

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

        let yaw_delta = -delta.x * self.rotate_sensitivity;
        let pitch_delta = -delta.y * self.rotate_sensitivity;

        // Create rotation quaternions
        let yaw_rotation = Quat::from_axis_angle(Vec3::Y, yaw_delta);
        let pitch_rotation = Quat::from_axis_angle(Vec3::X, pitch_delta);

        // Apply rotations
        self.rotation = yaw_rotation * self.rotation * pitch_rotation;

        // Update position based on rotation
        let distance = (self.position - self.target).length();
        let direction = self.rotation * Vec3::new(0.0, 0.0, distance);
        self.position = self.target + direction;

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
                self.position = Vec3::new(0.0, 0.0, 5.0);
                self.target = Vec3::ZERO;
                self.rotation = Quat::IDENTITY;
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
            ProjectionType::Perspective { .. } => {
                let max_size = size.x.max(size.y).max(size.z);
                let distance = max_size * 2.0; // Ensure everything fits

                self.target = center;
                let direction = (self.position - self.target).normalize();
                self.position = self.target + direction * distance;
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
    pub fn screen_to_world(&self, screen_pos: Vec2, screen_size: Vec2, depth: f32) -> Vec3 {
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
                println!("ORTHO: Creating matrix with bounds: left={}, right={}, bottom={}, top={}, near={}, far={}", 
                         left, right, bottom, top, near, far);
                println!("ORTHO: Camera aspect_ratio={}", self.aspect_ratio);
                Mat4::orthographic_rh(left, right, bottom, top, near, far)
            }
        };

        self.view_proj_dirty = false;
    }
}

/// Camera controller for handling input events
#[derive(Debug, Default)]
pub struct CameraController {
    pub is_dragging: bool,
    pub is_panning: bool,
    pub last_mouse_pos: Vec2,
    pub mouse_delta: Vec2,
}

impl CameraController {
    pub fn new() -> Self {
        Self::default()
    }

    /// Handle mouse press
    pub fn mouse_press(&mut self, position: Vec2, button: MouseButton) {
        self.last_mouse_pos = position;
        match button {
            MouseButton::Left => self.is_dragging = true,
            MouseButton::Right => self.is_panning = true,
            _ => {}
        }
    }

    /// Handle mouse release
    pub fn mouse_release(&mut self, _button: MouseButton) {
        self.is_dragging = false;
        self.is_panning = false;
    }

    /// Handle mouse movement
    pub fn mouse_move(&mut self, position: Vec2, camera: &mut Camera) {
        self.mouse_delta = position - self.last_mouse_pos;

        if self.is_dragging {
            camera.rotate(self.mouse_delta);
        } else if self.is_panning {
            camera.pan(self.mouse_delta);
        }

        self.last_mouse_pos = position;
    }

    /// Handle mouse wheel
    pub fn mouse_wheel(&mut self, delta: f32, camera: &mut Camera) {
        camera.zoom(delta);
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
        assert_eq!(camera.position, Vec3::new(0.0, 0.0, 5.0));
        assert_eq!(camera.target, Vec3::ZERO);
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
