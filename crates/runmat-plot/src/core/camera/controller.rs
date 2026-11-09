//! The camera controller: orbit, pan and zoom from mouse, wheel and trackpad input.

use super::*;

/// Camera controller for handling input events
#[derive(Debug, Default)]
pub struct CameraController {
    pub active_button: Option<MouseButton>,
    pub last_mouse_pos: Vec2,
    pub mouse_delta: Vec2,
    pub(super) navigation: NavigationOptions,
    pub(super) cursor_orbit_anchor: Option<Vec3>,
    pub(super) drag_orbit_pivot: Option<Vec3>,
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

    pub(super) fn orbit(
        &mut self,
        camera: &mut Camera,
        delta: Vec2,
        position: Vec2,
        viewport: (u32, u32),
    ) {
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
