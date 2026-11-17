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
//! of [`crate::assembly::resolve`], warm-started from the previous sample), and solves each
//! pose's velocity and acceleration equations with the drive law's exact rates. Nothing is
//! interpolated between poses: every frame the Desktop plays is a solved pose.
//!
//! Named mate positions ([`MatePosition`], `position open = { values = [{ mate = hinge, value =
//! 90 deg }] }`) save values for a set of driving mates, so the mechanism can be stepped between
//! them.

pub mod drag;
pub mod dynamics;
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

    /// The rise's rate, `ds/dτ`.
    pub fn rate(self, tau: f64) -> f64 {
        let tau = tau.clamp(0.0, 1.0);
        match self {
            MotionProfile::Linear => 1.0,
            MotionProfile::Cycloidal => 1.0 - (2.0 * std::f64::consts::PI * tau).cos(),
        }
    }

    /// The rise's second derivative, `d²s/dτ²`.
    pub fn curvature(self, tau: f64) -> f64 {
        let tau = tau.clamp(0.0, 1.0);
        match self {
            MotionProfile::Linear => 0.0,
            MotionProfile::Cycloidal => {
                2.0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * tau).sin()
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

/// Whether a study drives its mates through set values (kinematics) or integrates the
/// mechanism's equations of motion under its loads (dynamics, [`dynamics`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotionKind {
    #[default]
    Kinematic,
    Dynamic,
}

/// A spring and damper acting along a mate's own value (`{ mate = hinge, stiffness = 0.5
/// N*m/rad, damping = 0.01 N*m*s/rad, free = 0 deg }`): on a revolute (or a cylindrical's
/// angle) a torque `−k (θ − θ₀) − c θ̇`, on a slider a force `−k (d − d₀) − c ḋ`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionSpring {
    pub mate: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stiffness: Option<ValueExpr>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub damping: Option<ValueExpr>,
    /// The value at which the spring is unloaded (default 0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub free: Option<ValueExpr>,
}

/// A motor on a mate: a constant torque about a revolute's axis or a force along a slider's.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionMotor {
    pub mate: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub torque: Option<ValueExpr>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force: Option<ValueExpr>,
}

/// A motion study of the assembly (`motion <name> = { … }`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionStudy {
    pub name: String,
    #[serde(default, skip_serializing_if = "is_kinematic")]
    pub kind: MotionKind,
    pub duration: ValueExpr,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drives: Vec<MotionDrive>,
    /// A dynamic study's gravity, three accelerations (default `[0, 0, −9.80665] m/s²`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gravity: Option<Vec<ValueExpr>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub springs: Vec<MotionSpring>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub motors: Vec<MotionMotor>,
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

fn is_kinematic(k: &MotionKind) -> bool {
    *k == MotionKind::Kinematic
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
