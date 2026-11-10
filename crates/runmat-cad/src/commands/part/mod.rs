//! Part editor commands (`.part`): Sketch, Features, Direct, Surfaces, Sheet Metal, Weldments,
//! Mold, Evaluate, Annotate, Simulate and Manage, per the UI target §6.

mod features;
mod direct_surfaces;
mod sheet_metal;
mod weldments_mold;
mod analysis;
mod manage;
mod rows;
use analysis::*;
use direct_surfaces::*;
use features::*;
use manage::*;
use rows::*;
use sheet_metal::*;
use weldments_mold::*;

use super::view::cluster_3d;
use super::*;
use EditorKind::Part;
use SelectKind as K;

const P: &[EditorKind] = &[Part];

const END_CONDITIONS: &[(&str, &str)] = &[
    ("blind", "Blind"),
    ("through_all", "Through all"),
    ("up_to_next", "Up to next"),
    ("up_to_vertex", "Up to vertex"),
    ("up_to_face", "Up to face"),
    ("up_to_plane", "Up to plane"),
    ("offset_from_face", "Offset from face"),
    ("up_to_body", "Up to body"),
    ("mid_plane", "Mid-plane"),
];

const BODY_OPS: &[(&str, &str)] = &[
    ("add", "Add"),
    ("remove", "Remove"),
    ("intersect", "Intersect"),
    ("new", "New body"),
];

fn end_fields(prefix: &str, label: &str) -> Vec<Field> {
    let id = |s: &str| format!("{prefix}{s}");
    vec![
        choice(&id("end"), label, END_CONDITIONS),
        length(&id("distance"), "Depth", 10.0)
            .shown(&id("end"), &["blind", "mid_plane", "offset_from_face"])
            .drag(Manipulator::Arrow),
        pick(
            &id("up_to"),
            "Up to",
            &[K::Face, K::Vertex, K::Body, K::Plane],
        )
        .shown(
            &id("end"),
            &[
                "up_to_vertex",
                "up_to_face",
                "up_to_plane",
                "offset_from_face",
                "up_to_body",
            ],
        ),
        flag(&id("flip"), "Flip direction", false),
    ]
}

fn op_field() -> Field {
    radio("op", "Op", BODY_OPS)
}

fn merge_field() -> Field {
    pick("merge", "Merge into", &[K::Body])
        .optional()
        .shown("op", &["add", "remove", "intersect"])
}

fn profile() -> Field {
    picks("profile", "Profile", &[K::SketchRegion, K::Sketch, K::Face])
}

fn bodies_exist() -> Predicate {
    has(Facet::Bodies)
}

fn sheet_metal() -> Predicate {
    has(Facet::SheetMetal)
}

pub(super) fn add(catalog: &mut Catalog) {
    let mut commands = Vec::new();
    commands.extend(features());
    commands.extend(direct());
    commands.extend(surfaces());
    commands.extend(sheet_metal_commands());
    commands.extend(weldments());
    commands.extend(mold());
    commands.extend(evaluate());
    commands.extend(annotate());
    commands.extend(simulate());
    commands.extend(manage());
    commands.extend(tree_actions());
    catalog.add_commands(commands);
    catalog.rows.push(rows(catalog));
}
