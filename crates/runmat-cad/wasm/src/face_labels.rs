//! Face labels for Evaluate tables: each face of a regenerated part as an engineer reads it
//! (`Hollow inner wall · body side 1`), from the part's name table
//! ([`runmat_cad::refs::NameTable::label`]). Every Evaluate command that lists faces names them
//! through [`labeller`], so a face reads the same in each table.

use runmat_cad::display::FaceDisplay;
use runmat_cad::refs::{Reference, TopoKind};
use runmat_cad::syntax::refs::print_reference;

use crate::runtime::RegeneratedPart;

/// Names a face of body `body` (the display's body index) for people: the label of the name
/// whose text the display shows as the face's reference, else of the face's first name, else
/// `Face <n>`.
pub fn labeller(part: &RegeneratedPart) -> impl Fn(u32, &FaceDisplay) -> String + '_ {
    move |body, face| {
        let state = &part.result.state;
        let names = state.bodies.get(body as usize).and_then(|b| b.names.faces.get(face.index as usize));
        let text = |n: &runmat_cad::refs::EntityName| print_reference(&Reference::Name { kind: TopoKind::Face, name: n.clone() });
        let name = names.and_then(|set| {
            set.iter()
                .find(|n| face.reference.is_some() && text(n) == face.reference)
                .or_else(|| set.iter().next())
        });
        match name {
            Some(name) => state.names.label(name),
            None => format!("Face {}", face.index + 1),
        }
    }
}
