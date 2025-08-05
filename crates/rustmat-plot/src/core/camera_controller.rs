//! Advanced camera controller for interactive plot navigation
//! 
//! Provides smooth, responsive camera controls for both 2D and 3D plotting
//! with momentum, constraints, and configurable interaction modes.

use crate::core::camera::{Camera, ProjectionType, CameraMode};
use glam::{Vec2, Vec3, Mat4, Quat};
use std::time::{Duration, Instant};

/// Interactive camera controller with smooth motion and constraints
#[derive(Debug)]
pub struct CameraController {
    /// Camera being controlled
    pub camera: Camera,
    
    /// Input state
    input_state: InputState,
    
    /// Motion parameters
    pub pan_sensitivity: f32,
    pub zoom_sensitivity: f32,
    pub rotation_sensitivity: f32,
    
    /// Momentum and smoothing
    pub momentum_enabled: bool,
    pub momentum_decay: f32,
    velocity: MotionVelocity,
    
    /// Constraints
    pub constraints: CameraConstraints,
    
    /// Animation state
    animation: Option<CameraAnimation>,
    last_update: Instant,
}

/// Current input state for camera control
#[derive(Debug, Default)]
struct InputState {
    /// Mouse position (normalized -1 to 1)
    mouse_pos: Vec2,
    last_mouse_pos: Vec2,
    
    /// Mouse button states
    left_button_down: bool,
    right_button_down: bool,
    middle_button_down: bool,
    
    /// Keyboard states
    keys_down: std::collections::HashSet<VirtualKeyCode>,
    
    /// Scroll wheel delta
    scroll_delta: f32,
}

/// Motion velocity for momentum
#[derive(Debug, Default)]
struct MotionVelocity {
    pan: Vec2,
    zoom: f32,
    rotation: Vec2,
}

/// Camera movement constraints
#[derive(Debug, Clone)]
pub struct CameraConstraints {
    /// Zoom limits
    pub min_zoom: f32,
    pub max_zoom: f32,
    
    /// Pan limits (in world coordinates)
    pub pan_bounds: Option<(Vec3, Vec3)>, // (min, max)
    
    /// Rotation limits (in radians)
    pub pitch_limits: Option<(f32, f32)>, // (min, max)
    pub yaw_limits: Option<(f32, f32)>,   // (min, max)
    
    /// Prevent camera from going inside objects
    pub collision_radius: f32,
    
    /// Lock axes for 2D navigation
    pub lock_x: bool,
    pub lock_y: bool,
    pub lock_z: bool,
}

/// Camera animation for smooth transitions
#[derive(Debug)]
struct CameraAnimation {
    start_position: Vec3,
    target_position: Vec3,
    start_rotation: Quat,
    target_rotation: Quat,
    start_zoom: f32,
    target_zoom: f32,
    duration: Duration,
    elapsed: Duration,
    easing: EasingType,
}

/// Easing functions for camera animations
#[derive(Debug, Clone, Copy)]
pub enum EasingType {
    Linear,
    EaseOut,
    EaseInOut,
    Bounce,
}

/// Simplified key codes for camera control
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VirtualKeyCode {
    W, A, S, D,
    Q, E,
    Shift, Ctrl,
    Space,
    ArrowUp, ArrowDown, ArrowLeft, ArrowRight,
}

/// Camera interaction modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InteractionMode {
    /// Standard orbit camera (good for 3D data)
    Orbit,
    /// First-person camera (good for immersive 3D)
    FirstPerson,
    /// 2D pan/zoom only (good for plots)
    Pan2D,
    /// Isometric view (good for technical diagrams)
    Isometric,
}

impl Default for CameraConstraints {
    fn default() -> Self {
        Self {
            min_zoom: 0.1,
            max_zoom: 100.0,
            pan_bounds: None,
            pitch_limits: Some((-1.5, 1.5)), // ~85 degrees
            yaw_limits: None,
            collision_radius: 0.1,
            lock_x: false,
            lock_y: false,
            lock_z: false,
        }
    }
}

impl CameraController {
    /// Create a new camera controller
    pub fn new(camera: Camera) -> Self {
        Self {
            camera,
            input_state: InputState::default(),
            pan_sensitivity: 1.0,
            zoom_sensitivity: 1.0,
            rotation_sensitivity: 1.0,
            momentum_enabled: true,
            momentum_decay: 0.9,
            velocity: MotionVelocity::default(),
            constraints: CameraConstraints::default(),
            animation: None,
            last_update: Instant::now(),
        }
    }
    
    /// Create a controller optimized for 2D plotting
    pub fn for_2d_plotting(viewport_width: f32, viewport_height: f32) -> Self {
        let camera = Camera::new_2d(viewport_width, viewport_height);
        let mut controller = Self::new(camera);
        
        // 2D-specific settings
        controller.constraints.lock_z = true;
        controller.constraints.pitch_limits = Some((0.0, 0.0)); // No pitch in 2D
        controller.constraints.yaw_limits = Some((0.0, 0.0));   // No yaw in 2D
        controller.pan_sensitivity = 1.5;
        controller.zoom_sensitivity = 0.1;
        
        controller
    }
    
    /// Create a controller optimized for 3D plotting
    pub fn for_3d_plotting() -> Self {
        let camera = Camera::new_3d(1920.0 / 1080.0);
        let mut controller = Self::new(camera);
        
        // 3D-specific settings
        controller.rotation_sensitivity = 0.5;
        controller.pan_sensitivity = 2.0;
        controller.zoom_sensitivity = 0.2;
        
        controller
    }
    
    /// Set interaction mode
    pub fn set_interaction_mode(&mut self, mode: InteractionMode) {
        match mode {
            InteractionMode::Orbit => {
                self.rotation_sensitivity = 0.5;
                self.constraints.pitch_limits = Some((-1.5, 1.5));
                self.constraints.yaw_limits = None;
            },
            InteractionMode::FirstPerson => {
                self.rotation_sensitivity = 0.3;
                self.constraints.pitch_limits = Some((-1.5, 1.5));
                self.constraints.yaw_limits = None;
            },
            InteractionMode::Pan2D => {
                self.constraints.lock_z = true;
                self.constraints.pitch_limits = Some((0.0, 0.0));
                self.constraints.yaw_limits = Some((0.0, 0.0));
                self.rotation_sensitivity = 0.0;
            },
            InteractionMode::Isometric => {
                self.constraints.pitch_limits = Some((0.615, 0.615)); // 35.26 degrees
                self.constraints.yaw_limits = Some((0.785, 0.785));   // 45 degrees
                self.rotation_sensitivity = 0.0;
            },
        }
    }
    
    /// Update camera based on input
    pub fn update(&mut self, delta_time: Duration) {
        let dt = delta_time.as_secs_f32();
        
        // Update animation if active
        if let Some(ref mut anim) = self.animation {
            self.update_animation(anim, dt);
            if anim.elapsed >= anim.duration {
                self.animation = None;
            }
            return;
        }
        
        // Process input-based movement
        self.process_input(dt);
        
        // Apply momentum
        if self.momentum_enabled {
            self.apply_momentum(dt);
        }
        
        // Apply constraints
        self.apply_constraints();
        
        self.last_update = Instant::now();
    }
    
    /// Handle mouse movement
    pub fn handle_mouse_move(&mut self, position: Vec2) {
        self.input_state.last_mouse_pos = self.input_state.mouse_pos;
        self.input_state.mouse_pos = position;
    }
    
    /// Handle mouse button press/release
    pub fn handle_mouse_button(&mut self, button: MouseButton, pressed: bool) {
        match button {
            MouseButton::Left => self.input_state.left_button_down = pressed,
            MouseButton::Right => self.input_state.right_button_down = pressed,
            MouseButton::Middle => self.input_state.middle_button_down = pressed,
        }
    }
    
    /// Handle scroll wheel
    pub fn handle_scroll(&mut self, delta: f32) {
        self.input_state.scroll_delta += delta;
    }
    
    /// Handle keyboard input
    pub fn handle_keyboard(&mut self, key: VirtualKeyCode, pressed: bool) {
        if pressed {
            self.input_state.keys_down.insert(key);
        } else {
            self.input_state.keys_down.remove(&key);
        }
    }
    
    /// Process input and update camera
    fn process_input(&mut self, dt: f32) {
        let mouse_delta = self.input_state.mouse_pos - self.input_state.last_mouse_pos;
        
        // Handle mouse-based navigation
        if self.input_state.left_button_down {
            self.handle_rotation(mouse_delta, dt);
        }
        
        if self.input_state.right_button_down || self.input_state.middle_button_down {
            self.handle_pan(mouse_delta, dt);
        }
        
        // Handle scroll-based zoom
        if self.input_state.scroll_delta.abs() > 0.001 {
            self.handle_zoom(self.input_state.scroll_delta, dt);
            self.input_state.scroll_delta = 0.0;
        }
        
        // Handle keyboard-based movement
        self.handle_keyboard_movement(dt);
    }
    
    /// Handle rotation input
    fn handle_rotation(&mut self, mouse_delta: Vec2, dt: f32) {
        if self.rotation_sensitivity <= 0.0 {
            return;
        }
        
        let rotation_speed = self.rotation_sensitivity * dt;
        let yaw_delta = -mouse_delta.x * rotation_speed;
        let pitch_delta = -mouse_delta.y * rotation_speed;
        
        // Apply rotation based on camera mode
        match self.camera.mode {
            CameraMode::Orbit => {
                self.camera.orbit_yaw += yaw_delta;
                self.camera.orbit_pitch += pitch_delta;
                
                // Update velocity for momentum
                if self.momentum_enabled {
                    self.velocity.rotation.x += pitch_delta;
                    self.velocity.rotation.y += yaw_delta;
                }
            },
            CameraMode::FirstPerson => {
                self.camera.yaw += yaw_delta;
                self.camera.pitch += pitch_delta;
                
                if self.momentum_enabled {
                    self.velocity.rotation.x += pitch_delta;
                    self.velocity.rotation.y += yaw_delta;
                }
            },
        }
    }
    
    /// Handle pan input
    fn handle_pan(&mut self, mouse_delta: Vec2, dt: f32) {
        let pan_speed = self.pan_sensitivity * dt;
        let pan_delta = mouse_delta * pan_speed;
        
        // Apply pan relative to camera orientation
        let right = self.camera.get_right_vector();
        let up = self.camera.get_up_vector();
        
        let world_pan = -right * pan_delta.x + up * pan_delta.y;
        
        match self.camera.mode {
            CameraMode::Orbit => {
                self.camera.orbit_target += world_pan;
            },
            CameraMode::FirstPerson => {
                self.camera.position += world_pan;
            },
        }
        
        // Update velocity for momentum
        if self.momentum_enabled {
            self.velocity.pan += pan_delta;
        }
    }
    
    /// Handle zoom input
    fn handle_zoom(&mut self, scroll_delta: f32, dt: f32) {
        let zoom_speed = self.zoom_sensitivity * dt;
        let zoom_factor = 1.0 + scroll_delta * zoom_speed;
        
        match self.camera.mode {
            CameraMode::Orbit => {
                self.camera.orbit_distance *= zoom_factor;
            },
            CameraMode::FirstPerson => {
                // Move camera forward/backward
                let forward = self.camera.get_forward_vector();
                let movement = forward * scroll_delta * zoom_speed * 5.0;
                self.camera.position += movement;
            },
        }
        
        // Update velocity for momentum
        if self.momentum_enabled {
            self.velocity.zoom += scroll_delta * zoom_speed;
        }
    }
    
    /// Handle keyboard movement
    fn handle_keyboard_movement(&mut self, dt: f32) {
        let move_speed = 5.0 * dt;
        let mut movement = Vec3::ZERO;
        
        if self.input_state.keys_down.contains(&VirtualKeyCode::W) {
            movement += self.camera.get_forward_vector();
        }
        if self.input_state.keys_down.contains(&VirtualKeyCode::S) {
            movement -= self.camera.get_forward_vector();
        }
        if self.input_state.keys_down.contains(&VirtualKeyCode::A) {
            movement -= self.camera.get_right_vector();
        }
        if self.input_state.keys_down.contains(&VirtualKeyCode::D) {
            movement += self.camera.get_right_vector();
        }
        if self.input_state.keys_down.contains(&VirtualKeyCode::Q) {
            movement -= self.camera.get_up_vector();
        }
        if self.input_state.keys_down.contains(&VirtualKeyCode::E) {
            movement += self.camera.get_up_vector();
        }
        
        if movement.length() > 0.0 {
            movement = movement.normalize() * move_speed;
            
            match self.camera.mode {
                CameraMode::Orbit => {
                    self.camera.orbit_target += movement;
                },
                CameraMode::FirstPerson => {
                    self.camera.position += movement;
                },
            }
        }
    }
    
    /// Apply momentum to camera movement
    fn apply_momentum(&mut self, dt: f32) {
        let decay = self.momentum_decay.powf(dt);
        
        // Apply and decay pan momentum
        if self.velocity.pan.length() > 0.001 {
            let pan_delta = self.velocity.pan * dt;
            let right = self.camera.get_right_vector();
            let up = self.camera.get_up_vector();
            let world_pan = -right * pan_delta.x + up * pan_delta.y;
            
            match self.camera.mode {
                CameraMode::Orbit => {
                    self.camera.orbit_target += world_pan;
                },
                CameraMode::FirstPerson => {
                    self.camera.position += world_pan;
                },
            }
            
            self.velocity.pan *= decay;
        }
        
        // Apply and decay rotation momentum
        if self.velocity.rotation.length() > 0.001 {
            let rotation_delta = self.velocity.rotation * dt;
            
            match self.camera.mode {
                CameraMode::Orbit => {
                    self.camera.orbit_pitch += rotation_delta.x;
                    self.camera.orbit_yaw += rotation_delta.y;
                },
                CameraMode::FirstPerson => {
                    self.camera.pitch += rotation_delta.x;
                    self.camera.yaw += rotation_delta.y;
                },
            }
            
            self.velocity.rotation *= decay;
        }
        
        // Apply and decay zoom momentum
        if self.velocity.zoom.abs() > 0.001 {
            let zoom_delta = self.velocity.zoom * dt;
            
            match self.camera.mode {
                CameraMode::Orbit => {
                    self.camera.orbit_distance *= 1.0 + zoom_delta;
                },
                CameraMode::FirstPerson => {
                    let forward = self.camera.get_forward_vector();
                    let movement = forward * zoom_delta * 5.0;
                    self.camera.position += movement;
                },
            }
            
            self.velocity.zoom *= decay;
        }
    }
    
    /// Apply camera constraints
    fn apply_constraints(&mut self) {
        // Zoom constraints
        match self.camera.mode {
            CameraMode::Orbit => {
                self.camera.orbit_distance = self.camera.orbit_distance
                    .max(self.constraints.min_zoom)
                    .min(self.constraints.max_zoom);
            },
            CameraMode::FirstPerson => {
                // For first person, we constrain position instead of distance
                if let Some((min_bounds, max_bounds)) = self.constraints.pan_bounds {
                    self.camera.position = self.camera.position.max(min_bounds).min(max_bounds);
                }
            },
        }
        
        // Rotation constraints
        if let Some((min_pitch, max_pitch)) = self.constraints.pitch_limits {
            match self.camera.mode {
                CameraMode::Orbit => {
                    self.camera.orbit_pitch = self.camera.orbit_pitch.max(min_pitch).min(max_pitch);
                },
                CameraMode::FirstPerson => {
                    self.camera.pitch = self.camera.pitch.max(min_pitch).min(max_pitch);
                },
            }
        }
        
        if let Some((min_yaw, max_yaw)) = self.constraints.yaw_limits {
            match self.camera.mode {
                CameraMode::Orbit => {
                    self.camera.orbit_yaw = self.camera.orbit_yaw.max(min_yaw).min(max_yaw);
                },
                CameraMode::FirstPerson => {
                    self.camera.yaw = self.camera.yaw.max(min_yaw).min(max_yaw);
                },
            }
        }
        
        // Pan bounds constraints
        if let Some((min_bounds, max_bounds)) = self.constraints.pan_bounds {
            match self.camera.mode {
                CameraMode::Orbit => {
                    self.camera.orbit_target = self.camera.orbit_target.max(min_bounds).min(max_bounds);
                },
                CameraMode::FirstPerson => {
                    self.camera.position = self.camera.position.max(min_bounds).min(max_bounds);
                },
            }
        }
        
        // Axis locks
        if self.constraints.lock_x {
            match self.camera.mode {
                CameraMode::Orbit => { self.camera.orbit_target.x = 0.0; },
                CameraMode::FirstPerson => { self.camera.position.x = 0.0; },
            }
        }
        if self.constraints.lock_y {
            match self.camera.mode {
                CameraMode::Orbit => { self.camera.orbit_target.y = 0.0; },
                CameraMode::FirstPerson => { self.camera.position.y = 0.0; },
            }
        }
        if self.constraints.lock_z {
            match self.camera.mode {
                CameraMode::Orbit => { self.camera.orbit_target.z = 0.0; },
                CameraMode::FirstPerson => { self.camera.position.z = 0.0; },
            }
        }
    }
    
    /// Update camera animation
    fn update_animation(&mut self, animation: &mut CameraAnimation, dt: f32) {
        animation.elapsed += Duration::from_secs_f32(dt);
        let t = (animation.elapsed.as_secs_f32() / animation.duration.as_secs_f32()).min(1.0);
        
        // Apply easing
        let eased_t = match animation.easing {
            EasingType::Linear => t,
            EasingType::EaseOut => 1.0 - (1.0 - t).powi(3),
            EasingType::EaseInOut => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
                }
            },
            EasingType::Bounce => {
                if t < 1.0 / 2.75 {
                    7.5625 * t * t
                } else if t < 2.0 / 2.75 {
                    let t = t - 1.5 / 2.75;
                    7.5625 * t * t + 0.75
                } else if t < 2.5 / 2.75 {
                    let t = t - 2.25 / 2.75;
                    7.5625 * t * t + 0.9375
                } else {
                    let t = t - 2.625 / 2.75;
                    7.5625 * t * t + 0.984375
                }
            },
        };
        
        // Interpolate position and rotation
        let position = animation.start_position.lerp(animation.target_position, eased_t);
        let rotation = animation.start_rotation.slerp(animation.target_rotation, eased_t);
        let zoom = animation.start_zoom + (animation.target_zoom - animation.start_zoom) * eased_t;
        
        // Apply to camera
        match self.camera.mode {
            CameraMode::Orbit => {
                self.camera.orbit_target = position;
                self.camera.orbit_distance = zoom;
                
                // Extract euler angles from quaternion
                let (yaw, pitch, _roll) = rotation.to_euler(glam::EulerRot::YXZ);
                self.camera.orbit_yaw = yaw;
                self.camera.orbit_pitch = pitch;
            },
            CameraMode::FirstPerson => {
                self.camera.position = position;
                
                let (yaw, pitch, _roll) = rotation.to_euler(glam::EulerRot::YXZ);
                self.camera.yaw = yaw;
                self.camera.pitch = pitch;
            },
        }
    }
    
    /// Animate camera to a target position
    pub fn animate_to(&mut self, target_position: Vec3, target_rotation: Quat, target_zoom: f32, duration: Duration) {
        let current_rotation = match self.camera.mode {
            CameraMode::Orbit => {
                Quat::from_euler(glam::EulerRot::YXZ, self.camera.orbit_yaw, self.camera.orbit_pitch, 0.0)
            },
            CameraMode::FirstPerson => {
                Quat::from_euler(glam::EulerRot::YXZ, self.camera.yaw, self.camera.pitch, 0.0)
            },
        };
        
        let current_position = match self.camera.mode {
            CameraMode::Orbit => self.camera.orbit_target,
            CameraMode::FirstPerson => self.camera.position,
        };
        
        let current_zoom = match self.camera.mode {
            CameraMode::Orbit => self.camera.orbit_distance,
            CameraMode::FirstPerson => 1.0, // Not applicable for first person
        };
        
        self.animation = Some(CameraAnimation {
            start_position: current_position,
            target_position,
            start_rotation: current_rotation,
            target_rotation,
            start_zoom: current_zoom,
            target_zoom,
            duration,
            elapsed: Duration::ZERO,
            easing: EasingType::EaseOut,
        });
    }
    
    /// Fit camera to show all data in the given bounding box
    pub fn fit_to_bounds(&mut self, min_bounds: Vec3, max_bounds: Vec3, duration: Option<Duration>) {
        let center = (min_bounds + max_bounds) / 2.0;
        let size = max_bounds - min_bounds;
        let max_dimension = size.max_element();
        
        let distance = match self.camera.projection_type {
            ProjectionType::Perspective => {
                // For perspective, distance based on FOV
                let fov_factor = (self.camera.fov_y / 2.0).tan();
                max_dimension / (2.0 * fov_factor) * 1.5
            },
            ProjectionType::Orthographic => {
                // For orthographic, just ensure everything fits
                max_dimension * 1.2
            },
        };
        
        match duration {
            Some(duration) => {
                let target_rotation = Quat::from_euler(glam::EulerRot::YXZ, 0.0, 0.0, 0.0);
                self.animate_to(center, target_rotation, distance, duration);
            },
            None => {
                match self.camera.mode {
                    CameraMode::Orbit => {
                        self.camera.orbit_target = center;
                        self.camera.orbit_distance = distance;
                        self.camera.orbit_yaw = 0.0;
                        self.camera.orbit_pitch = 0.0;
                    },
                    CameraMode::FirstPerson => {
                        self.camera.position = center - Vec3::Z * distance;
                        self.camera.yaw = 0.0;
                        self.camera.pitch = 0.0;
                    },
                }
            },
        }
    }
    
    /// Reset camera to default position
    pub fn reset(&mut self, duration: Option<Duration>) {
        let default_position = Vec3::new(0.0, 0.0, 5.0);
        let default_rotation = Quat::IDENTITY;
        let default_zoom = 5.0;
        
        match duration {
            Some(duration) => {
                self.animate_to(default_position, default_rotation, default_zoom, duration);
            },
            None => {
                match self.camera.mode {
                    CameraMode::Orbit => {
                        self.camera.orbit_target = Vec3::ZERO;
                        self.camera.orbit_distance = default_zoom;
                        self.camera.orbit_yaw = 0.0;
                        self.camera.orbit_pitch = 0.0;
                    },
                    CameraMode::FirstPerson => {
                        self.camera.position = default_position;
                        self.camera.yaw = 0.0;
                        self.camera.pitch = 0.0;
                    },
                }
                
                // Clear velocity
                self.velocity = MotionVelocity::default();
            },
        }
    }
    
    /// Get the current view matrix
    pub fn view_matrix(&self) -> Mat4 {
        self.camera.view_matrix()
    }
    
    /// Get the current projection matrix
    pub fn projection_matrix(&self) -> Mat4 {
        self.camera.projection_matrix()
    }
    
    /// Get the combined view-projection matrix
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.camera.view_projection_matrix()
    }
}

/// Mouse button enum
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_camera_controller_creation() {
        let controller = CameraController::for_2d_plotting(800.0, 600.0);
        
        assert!(controller.constraints.lock_z);
        assert_eq!(controller.constraints.pitch_limits, Some((0.0, 0.0)));
        assert_eq!(controller.constraints.yaw_limits, Some((0.0, 0.0)));
    }
    
    #[test]
    fn test_interaction_modes() {
        let mut controller = CameraController::for_3d_plotting();
        
        controller.set_interaction_mode(InteractionMode::Pan2D);
        assert!(controller.constraints.lock_z);
        assert_eq!(controller.rotation_sensitivity, 0.0);
        
        controller.set_interaction_mode(InteractionMode::Orbit);
        assert!(!controller.constraints.lock_z);
        assert!(controller.rotation_sensitivity > 0.0);
    }
    
    #[test]
    fn test_zoom_constraints() {
        let mut controller = CameraController::for_3d_plotting();
        controller.constraints.min_zoom = 1.0;
        controller.constraints.max_zoom = 10.0;
        
        controller.camera.orbit_distance = 0.5; // Below minimum
        controller.apply_constraints();
        assert_eq!(controller.camera.orbit_distance, 1.0);
        
        controller.camera.orbit_distance = 15.0; // Above maximum
        controller.apply_constraints();
        assert_eq!(controller.camera.orbit_distance, 10.0);
    }
    
    #[test]
    fn test_momentum_decay() {
        let mut controller = CameraController::for_2d_plotting(800.0, 600.0);
        controller.velocity.pan = Vec2::new(1.0, 1.0);
        
        let initial_velocity = controller.velocity.pan.length();
        controller.apply_momentum(1.0); // 1 second
        
        assert!(controller.velocity.pan.length() < initial_velocity);
    }
    
    #[test]
    fn test_fit_to_bounds() {
        let mut controller = CameraController::for_3d_plotting();
        let min_bounds = Vec3::new(-5.0, -5.0, -5.0);
        let max_bounds = Vec3::new(5.0, 5.0, 5.0);
        
        controller.fit_to_bounds(min_bounds, max_bounds, None);
        
        assert_eq!(controller.camera.orbit_target, Vec3::ZERO);
        assert!(controller.camera.orbit_distance > 10.0); // Should be at a reasonable distance
    }
    
    #[test]
    fn test_animation_setup() {
        let mut controller = CameraController::for_3d_plotting();
        let target_pos = Vec3::new(10.0, 0.0, 0.0);
        let target_rot = Quat::IDENTITY;
        let duration = Duration::from_secs(2);
        
        controller.animate_to(target_pos, target_rot, 5.0, duration);
        
        assert!(controller.animation.is_some());
        assert_eq!(controller.animation.as_ref().unwrap().target_position, target_pos);
    }
}