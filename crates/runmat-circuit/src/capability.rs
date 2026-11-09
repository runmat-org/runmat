//! Board-house capability rule sets (PCB-26): named sets of fabrication limits a layout loads
//! with `limits = { capability = advanced }`. Values written beside `capability` override the
//! set's value for that key; every other key takes the set's value.
//!
//! The three sets follow the classes board houses publish for rigid boards:
//! - `standard`: the lowest-cost process, 0.15 mm track and space and 0.3 mm drills;
//! - `advanced`: 0.1 mm track and space and 0.2 mm drills;
//! - `hdi`: laser microvias and 0.075 mm track and space.

use crate::geom::mm;
use crate::layout::FabLimits;

/// A named fabrication capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Capability {
    Standard,
    Advanced,
    Hdi,
}

impl Capability {
    pub const ALL: [Capability; 3] = [Capability::Standard, Capability::Advanced, Capability::Hdi];

    /// The word a layout writes: `standard`, `advanced` or `hdi`.
    pub fn word(self) -> &'static str {
        match self {
            Capability::Standard => "standard",
            Capability::Advanced => "advanced",
            Capability::Hdi => "hdi",
        }
    }

    pub fn from_word(w: &str) -> Option<Capability> {
        Capability::ALL.into_iter().find(|c| c.word() == w)
    }

    /// What the set is, for the command's choices and the rules table.
    pub fn label(self) -> &'static str {
        match self {
            Capability::Standard => "Standard (0.15 mm track and space, 0.3 mm drill)",
            Capability::Advanced => "Advanced (0.1 mm track and space, 0.2 mm drill)",
            Capability::Hdi => "HDI (0.075 mm track and space, laser microvias)",
        }
    }

    /// The set's limits, in the order of [`FabLimits`]' fields: clearance, track, via diameter,
    /// drill, annular ring, hole to hole, copper to edge, silk to pad, microvia drill (mm).
    fn values(self) -> [f64; 9] {
        match self {
            Capability::Standard => [0.15, 0.15, 0.6, 0.3, 0.15, 0.5, 0.3, 0.15, 0.1],
            Capability::Advanced => [0.1, 0.1, 0.45, 0.2, 0.1, 0.25, 0.25, 0.1, 0.1],
            Capability::Hdi => [0.075, 0.075, 0.3, 0.15, 0.075, 0.2, 0.2, 0.075, 0.075],
        }
    }

    /// The set as limits, with `capability` naming it and no written values.
    pub fn limits(self) -> FabLimits {
        let [clearance, track, via, drill, annular, hole, edge, silk, micro] =
            self.values().map(mm);
        FabLimits {
            min_clearance: clearance,
            min_track: track,
            min_via_diameter: via,
            min_drill: drill,
            min_annular: annular,
            min_hole_to_hole: hole,
            min_copper_to_edge: edge,
            min_silk_to_pad: silk,
            min_microvia_drill: micro,
            capability: Some(self),
            sources: Default::default(),
        }
    }
}
