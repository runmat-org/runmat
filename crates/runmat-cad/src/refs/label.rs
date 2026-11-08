//! Names for people: a face's canonical name as an engineer reads it in a results table
//! (`Hollow inner wall · body side 1`), built from the name's recipe and the part's name
//! table. The canonical reference text stays the id; the label is only for reading.

use super::name::{recipe_of, EntityName, GenerationName, InstanceKey, NameTable, NameToken, PlaneSide, Role, SourceTag};

/// How deep a label spells out the names it is made from; deeper ones read `…`.
const MAX_LABEL_DEPTH: u32 = 3;

/// `front_left_seat` → `front left seat`.
fn words(id: &str) -> String {
    id.replace(['_', '-'], " ")
}

fn capitalized(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// A sketch curve as a side's name: a numbered curve (`l2`, `c10`) reads by its number,
/// a named one (`yaw`, `top_edge`) by its words.
fn curve(entity: &str) -> String {
    let digits = entity.trim_start_matches(|c: char| c.is_ascii_alphabetic());
    if !digits.is_empty() && digits.len() < entity.len() && entity.len() - digits.len() <= 2 && digits.chars().all(|c| c.is_ascii_digit()) {
        digits.to_string()
    } else {
        words(entity)
    }
}

fn role_words(role: Role) -> &'static str {
    match role {
        Role::Side => "side",
        Role::StartCap => "start face",
        Role::EndCap => "end face",
        Role::Blend => "round",
        Role::Chamfer => "chamfer",
        Role::ShellInner => "inner wall",
        Role::HoleWall => "hole wall",
        Role::HoleBottom => "hole bottom",
        Role::HoleCounterbore => "counterbore",
        Role::HoleCounterboreWall => "counterbore wall",
        Role::HoleCountersink => "countersink",
        Role::ShellRim => "rim",
        Role::ShellOuter => "outer wall",
        Role::ThinWall => "thin wall",
        Role::DraftStep => "draft step",
        Role::SplitFace => "split face",
        Role::Body => "body",
        Role::Replacement => "replaced face",
        Role::Boundary => "boundary surface",
        Role::SurfacePatch => "patch",
        Role::OffsetFace => "offset face",
        Role::Extension => "extension",
        Role::ThickenFace => "thickened face",
        Role::ThickenRim => "thickened rim",
        Role::MidSurface => "mid-surface",
        Role::Ruled => "ruled surface",
        Role::SheetFace => "sheet face",
        Role::SheetEdge => "sheet edge",
        Role::BendFace => "bend",
        Role::Relief => "relief",
        Role::FormFace => "form face",
        Role::MemberSide => "member side",
        Role::MemberEnd => "member end",
        Role::CapPlate => "cap plate",
        Role::Gusset => "gusset",
        Role::WeldBead => "weld bead",
        Role::PartingFace => "parting face",
        Role::ThreadFace => "thread",
        Role::EmbossTop => "emboss top",
        Role::Rib => "rib face",
    }
}

/// The feature and the face's role, with the role left out when the feature's name already
/// ends with it: a fillet feature named `round` reads `Round between …`, not `Round round`.
fn head(feature: &str, role: &str) -> String {
    let lower = feature.to_lowercase();
    if lower == role || lower.ends_with(&format!(" {role}")) {
        feature.to_string()
    } else {
        format!("{feature} {role}")
    }
}

fn instance(key: &InstanceKey) -> String {
    match key {
        InstanceKey::Linear { step } | InstanceKey::Circular { step } | InstanceKey::Curve { step } => format!("copy {step}"),
        InstanceKey::Grid { first, second } => format!("copy {first}, {second}"),
        InstanceKey::Lattice { i, j } => format!("copy {i}, {j}"),
        InstanceKey::Ring { ring, step } => format!("ring {ring} copy {step}"),
        InstanceKey::Row { key } => format!("row {}", words(key)),
        InstanceKey::Mirror => "mirror image".into(),
        InstanceKey::Point { point } => format!("copy at {}", words(point)),
    }
}

impl NameTable {
    /// The name as an engineer reads it: the feature, what the face is to it, and what it was
    /// made from (`Hollow inner wall · body side 1`, `Edge round between body side 1 and body
    /// end face`, `Row copy 2 · boss side 1`). Names the table doesn't hold read through the
    /// recipes their tags recorded, else as `a face`.
    pub fn label(&self, name: &EntityName) -> String {
        capitalized(&self.label_at(name, 0))
    }

    fn child(&self, token: &NameToken, depth: u32) -> String {
        if depth >= MAX_LABEL_DEPTH {
            return "…".into();
        }
        match self.get(token).cloned().or_else(|| recipe_of(token)) {
            Some(name) => self.label_at(&name, depth + 1),
            None => "a face".into(),
        }
    }

    fn label_at(&self, name: &EntityName, depth: u32) -> String {
        match name {
            EntityName::Generated(GenerationName { feature, role, tag, ordinal, step: _ }) => {
                let head = head(&words(&feature.0), role_words(*role));
                let mut text = match tag {
                    SourceTag::None => head,
                    SourceTag::Curve { entity, .. } => format!("{head} {}", curve(&entity.0)),
                    SourceTag::Key { key } => format!("{head} {}", words(key)),
                    SourceTag::Side { side } => format!(
                        "{head}, {} side",
                        match side {
                            PlaneSide::Negative => "negative",
                            PlaneSide::Positive => "positive",
                        }
                    ),
                    SourceTag::Face { face } => format!("{head} · {}", self.child(face, depth)),
                    SourceTag::Edge { between } => {
                        format!("{head} between {} and {}", self.child(&between[0], depth), self.child(&between[1], depth))
                    }
                    SourceTag::Vertex { faces } => {
                        let names: Vec<String> = faces.iter().map(|t| self.child(t, depth)).collect();
                        format!("{head} at {}", names.join(", "))
                    }
                };
                if let Some(ordinal) = ordinal {
                    text += &format!(", piece {}", ordinal + 1);
                }
                text
            }
            EntityName::Copy { base, pattern, instance: key, .. } => {
                format!("{} {} · {}", words(&pattern.0), instance(key), self.child(base, depth))
            }
            EntityName::CadFile { name, .. } => name.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{FeatureId, SketchEntityId};

    fn side(feature: &str, entity: &str) -> EntityName {
        EntityName::generated(feature, Role::Side, SourceTag::Curve { sketch: FeatureId::new("s1"), entity: SketchEntityId::new(entity) })
    }

    #[test]
    fn labels_read_as_the_feature_what_the_face_is_and_what_it_came_from() {
        let mut table = NameTable::new();
        let wall = side("body", "l2");
        let top = EntityName::generated("body", Role::EndCap, SourceTag::None);
        let inner = EntityName::generated("hollow", Role::ShellInner, SourceTag::face_of(&wall));
        let round = EntityName::generated("edge_round", Role::Blend, SourceTag::edge_between(&wall, &top));
        let copy = EntityName::copy(&inner, &FeatureId::new("row"), InstanceKey::Linear { step: 2 });
        for n in [&wall, &top, &inner, &round] {
            table.intern(n).unwrap();
        }
        assert_eq!(table.label(&wall), "Body side 2");
        assert_eq!(table.label(&side("body", "yaw_bore")), "Body side yaw bore");
        assert_eq!(table.label(&inner), "Hollow inner wall · body side 2");
        let r = table.label(&round);
        assert!(r.starts_with("Edge round between "), "{r}");
        // A feature named after its role says the word once.
        let named_round = EntityName::generated("round", Role::Blend, SourceTag::edge_between(&wall, &top));
        let n = table.label(&named_round);
        assert!(n.starts_with("Round between ") && n.contains("body side 2") && n.contains("body end face"), "{n}");
        assert_eq!(table.label(&EntityName::generated("chamfer", Role::Chamfer, SourceTag::None)), "Chamfer");
        assert!(r.contains("body side 2") && r.contains("body end face"), "{r}");
        assert_eq!(table.label(&copy), "Row copy 2 · hollow inner wall · body side 2");
    }
}
