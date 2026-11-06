//! Fit and mass checks a study reports before it meshes a design (CAD-38).
//!
//! A study that names a part or an assembly runs [`study_checks`] first:
//! - **mass:** every part occurrence has a density (its `density` property or its material's),
//!   so the study can give it a material and its loads from gravity and acceleration are right;
//! - **fit:** no two part occurrences share material (an overlap meshes as one region with
//!   doubled stiffness and mass), and pairs closer than the contact gap without touching are
//!   listed, since a bonded or contact pair that never meets transfers no load.
//!
//! An error stops the study; a warning is reported with its results. The Desktop runs the same
//! checks as `simulate.check` through [`crate::analyze`].

use crate::evaluate::{assembly_clearance, Interference};
use crate::mass::{assembly_mass, part_mass, MassReport};
use crate::runtime::Files;
use serde::Serialize;

/// Pairs closer than this without touching are reported as open contacts (m; 0.1 mm).
pub const CONTACT_GAP: f64 = 1e-4;

/// One finding: which check, how serious, what it says and the occurrences it names.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StudyCheck {
    /// `mass` or `fit`.
    pub check: &'static str,
    /// `error` stops the study; `warning` is reported with its results.
    pub severity: &'static str,
    pub message: String,
    /// Outline IDs of what the finding names (`occurrence:<path>`).
    pub items: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct StudyChecks {
    pub mass: MassReport,
    /// The fit check of an assembly; none for a part on its own.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fit: Option<Interference>,
    pub checks: Vec<StudyCheck>,
}

impl StudyChecks {
    /// Whether the study may run: no check found an error.
    pub fn ready(&self) -> bool {
        !self.checks.iter().any(|c| c.severity == "error")
    }
}

fn occurrence(path: &str) -> String {
    format!("occurrence:{path}")
}

/// Runs the fit and mass checks on the part or assembly `path` in `configuration`, with open
/// contacts reported below `gap` (m; [`CONTACT_GAP`] when `None`).
pub fn study_checks(
    path: &str,
    text: &str,
    files: &Files,
    configuration: Option<&str>,
    gap: Option<f64>,
) -> Result<StudyChecks, String> {
    let assembly = path.ends_with(".assembly");
    let mass = if assembly {
        assembly_mass(path, text, files, configuration)?
    } else {
        part_mass(path, text, configuration)?
    };
    let mut checks = Vec::new();
    for part in mass.parts.iter().filter(|p| p.mass.is_none()) {
        checks.push(StudyCheck {
            check: "mass",
            severity: "error",
            message: format!(
                "{} has no density: give it a material or a `density` property so the study can assign its material",
                part.path
            ),
            items: if assembly { vec![occurrence(&part.path)] } else { Vec::new() },
        });
    }
    if mass.parts.iter().any(|p| p.mass.is_some()) && mass.total.mass <= 0.0 {
        checks.push(StudyCheck {
            check: "mass",
            severity: "error",
            message: "the design has no volume to mesh".into(),
            items: Vec::new(),
        });
    }
    let fit = if assembly {
        let found = assembly_clearance(path, text, files, configuration, gap.unwrap_or(CONTACT_GAP))?;
        for o in &found.overlaps {
            checks.push(StudyCheck {
                check: "fit",
                severity: "error",
                message: format!(
                    "{} and {} overlap by {:.3} mm³; the mesh would merge them",
                    o.a,
                    o.b,
                    o.volume * 1e9
                ),
                items: vec![occurrence(&o.a), occurrence(&o.b)],
            });
        }
        for n in &found.near {
            checks.push(StudyCheck {
                check: "fit",
                severity: "warning",
                message: format!(
                    "{} and {} are {:.3} mm apart; a contact between them carries no load until the gap closes",
                    n.a,
                    n.b,
                    n.distance * 1000.0
                ),
                items: vec![occurrence(&n.a), occurrence(&n.b)],
            });
        }
        for e in &found.errors {
            checks.push(StudyCheck {
                check: "fit",
                severity: "warning",
                message: format!("a pair was not checked: {e}"),
                items: Vec::new(),
            });
        }
        Some(found)
    } else {
        None
    };
    Ok(StudyChecks { mass, fit, checks })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(material: &str) -> String {
        format!("runmat 0.7\nstd 1\n\npart block {{\n    sketch s1 on origin.xy {{\n        l1 = line(start = [0 mm, 0 mm], end = [10 mm, 0 mm])\n        l2 = line(start = [10 mm, 0 mm], end = [10 mm, 10 mm])\n        l3 = line(start = [10 mm, 10 mm], end = [0 mm, 10 mm])\n        l4 = line(start = [0 mm, 10 mm], end = [0 mm, 0 mm])\n    }}\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output block = {{ body = feature.body.body{material} }}\n}}\n")
    }

    #[test]
    fn a_part_with_a_material_is_ready_and_one_without_is_not() {
        let ready = study_checks("/p/block.part", &block(", material = \"material.aluminum\""), &Files::new(), None, None).unwrap();
        assert!(ready.ready(), "{:?}", ready.checks);
        assert!(ready.fit.is_none());
        let missing = study_checks("/p/block.part", &block(""), &Files::new(), None, None).unwrap();
        assert!(!missing.ready());
        assert_eq!(missing.checks[0].check, "mass");
        assert!(missing.checks[0].message.contains("has no density"), "{}", missing.checks[0].message);
    }
}
