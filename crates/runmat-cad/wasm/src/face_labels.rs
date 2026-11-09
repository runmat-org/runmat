//! Face labels for Evaluate tables: each face as an engineer reads it (`Hollow inner wall · body
//! side 1`). The display carries the label the name table gives each face
//! ([`runmat_cad::display::FaceDisplay::label`]); every Evaluate command that lists faces names
//! them through [`label`], so a face reads the same in each table, in forms and in the agent's
//! picks.

use runmat_cad::display::FaceDisplay;

/// Names face `face` of a body for people: its label, else `Face <n>`.
pub fn label(_body: u32, face: &FaceDisplay) -> String {
    face.label
        .clone()
        .unwrap_or_else(|| format!("Face {}", face.index + 1))
}
