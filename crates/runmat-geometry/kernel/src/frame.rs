//! Evaluated frames.

use serde::{Deserialize, Serialize};

/// An evaluated frame in part coordinates (SI). Right-handed and orthonormal.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    pub origin: [f64; 3],
    pub x: [f64; 3],
    pub y: [f64; 3],
    pub z: [f64; 3],
}

impl Frame {
    pub const WORLD: Frame = Frame {
        origin: [0.0; 3],
        x: [1.0, 0.0, 0.0],
        y: [0.0, 1.0, 0.0],
        z: [0.0, 0.0, 1.0],
    };

    /// The frame moved `distance` along its own z.
    pub fn offset(&self, distance: f64) -> Frame {
        Frame {
            origin: [
                self.origin[0] + self.z[0] * distance,
                self.origin[1] + self.z[1] * distance,
                self.origin[2] + self.z[2] * distance,
            ],
            ..*self
        }
    }

    /// Map sketch coordinates to part coordinates.
    pub fn to_world(&self, p: [f64; 2]) -> [f64; 3] {
        [
            // Same association as before the move, so results stay bit-identical.
            self.origin[0] + (self.x[0] * p[0] + self.y[0] * p[1]),
            self.origin[1] + (self.x[1] * p[0] + self.y[1] * p[1]),
            self.origin[2] + (self.x[2] * p[0] + self.y[2] * p[1]),
        ]
    }
}
