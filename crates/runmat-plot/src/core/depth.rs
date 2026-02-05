//! Depth and clip-plane policy for 3D rendering.
//!
//! This module centralizes decisions about:
//! - Depth buffer mode (standard vs reversed-Z)
//! - Clip plane selection (static vs dynamic)
//!
//! The goal is robustness across wide scale ranges (LiDAR, radar, etc.).

/// Depth mapping mode for the primary depth buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthMode {
    /// Standard depth: near maps to 0, far maps to 1.0. Compare: LessEqual. Clear: 1.0.
    Standard,
    /// Reversed-Z depth: near maps to 1.0, far maps to 0. Compare: GreaterEqual. Clear: 0.0.
    ReversedZ,
}

impl Default for DepthMode {
    fn default() -> Self {
        // Prefer reversed-Z by default for better depth precision on large scenes.
        Self::ReversedZ
    }
}

/// Clip-plane selection policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipPolicy {
    /// If true, update near/far every frame based on visible scene bounds.
    pub dynamic: bool,
    /// Minimum allowed near plane distance (prevents precision collapse).
    pub min_near: f32,
    /// Multiply the computed near by this padding factor (<1 pulls near closer).
    pub near_padding: f32,
    /// Multiply the computed far by this padding factor (>1 pushes far further).
    pub far_padding: f32,
    /// Clamp far to this max (safety against runaway bounds).
    pub max_far: f32,
}

impl Default for ClipPolicy {
    fn default() -> Self {
        Self {
            dynamic: true,
            min_near: 0.02,
            near_padding: 0.8,
            far_padding: 1.2,
            max_far: 1.0e9,
        }
    }
}

