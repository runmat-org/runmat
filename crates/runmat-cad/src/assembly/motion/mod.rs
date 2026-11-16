//! Motion studies: an assembly's mechanism moved through time by driving its mates.
//!
//! A [`MotionStudy`] names one or more mates and the range each one is driven through
//! (`from` → `to`, an angle for a revolute, cylindrical or angle mate, a length for a slider,
//! cylindrical or distance mate) over a duration, with a motion law ([`MotionProfile`]). The
//! file form is
//!
//! ```text
//! motion open_close = { duration = 4 s, drives = [{ mate = hinge, from = 0 deg, to = 90 deg }], track = lid }
//! ```
//!
//! [`kinematics::run_study`] solves the mechanism's position at each sample time (the mate solve
//! of [`crate::assembly::resolve`], warm-started from the previous sample), and differentiates
//! the positions for velocity and acceleration. Nothing is interpolated between poses: every
//! frame the Desktop plays is a solved pose.
//!
//! Named mate positions ([`MatePosition`], `position open = { values = [{ mate = hinge, value =
//! 90 deg }] }`) save values for a set of driving mates, so the mechanism can be stepped between
//! them.

pub mod drag;
pub mod kinematics;

use crate::value::ValueExpr;
use serde::{Deserialize, Serialize};

/// How a drive moves from `from` to `to` over the study's duration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotionProfile {
    /// Constant speed: `s(τ) = τ`.
    #[default]
    Linear,
    /// Cycloidal: `s(τ) = τ − sin(2πτ)/2π`, zero velocity and acceleration at both ends, the
    /// standard rise law of cam design (Norton, *Cam Design and Manufacturing Handbook*, 2009,
    /// ch. 3).
    Cycloidal,
}

impl MotionProfile {
    /// The fraction of the rise done at `tau` (0 to 1).
    pub fn rise(self, tau: f64) -> f64 {
        let tau = tau.clamp(0.0, 1.0);
        match self {
            MotionProfile::Linear => tau,
            MotionProfile::Cycloidal => {
                tau - (2.0 * std::f64::consts::PI * tau).sin() / (2.0 * std::f64::consts::PI)
            }
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            MotionProfile::Linear => "linear",
            MotionProfile::Cycloidal => "cycloidal",
        }
    }
}

/// One driven mate: its value goes from `from` to `to` over the study.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionDrive {
    pub mate: String,
    pub from: ValueExpr,
    pub to: ValueExpr,
}

/// A motion study of the assembly (`motion <name> = { … }`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionStudy {
    pub name: String,
    pub duration: ValueExpr,
    pub drives: Vec<MotionDrive>,
    /// Solved frames per second (default [`DEFAULT_RATE`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate: Option<f64>,
    /// Written `law = cycloidal` (`profile` is the sketch-profile field across the models).
    #[serde(rename = "law", default, skip_serializing_if = "is_linear")]
    pub profile: MotionProfile,
    /// The occurrence whose motion is plotted (its box centre's path, speed and acceleration).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track: Option<String>,
}

fn is_linear(p: &MotionProfile) -> bool {
    *p == MotionProfile::Linear
}

/// Frames per second when a study doesn't say.
pub const DEFAULT_RATE: f64 = 12.0;
/// The most frames one study solves (a long or fast study is sampled more coarsely).
pub const MAX_FRAMES: usize = 241;

/// One mate's value in a named position.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MateValue {
    pub mate: String,
    pub value: ValueExpr,
}

/// A named position of the mechanism (`position <name> = { values = [...] }`): the mate
/// controller's saved state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MatePosition {
    pub name: String,
    pub values: Vec<MateValue>,
}
