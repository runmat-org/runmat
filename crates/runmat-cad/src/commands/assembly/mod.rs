//! Assembly editor commands (`.assembly`), per the UI target §7.

mod assemble;
mod evaluate;
mod layout;
mod manage;
mod rows;

use super::view::cluster_3d;
use super::*;
use EditorKind::Assembly;
use SelectKind as K;

const A: &[EditorKind] = &[Assembly];

/// The kinds Edit mate changes a mate to, on the connectors it already has.
pub(super) const EDITABLE_MATE_TYPES: &[(&str, &str)] = &[
    ("fasten", "Fasten"),
    ("revolute", "Revolute"),
    ("slider", "Slider"),
    ("cylindrical", "Cylindrical"),
    ("planar", "Planar"),
    ("ball", "Ball"),
    ("parallel", "Parallel"),
    ("distance", "Distance"),
    ("angle", "Angle"),
    ("tangent", "Tangent"),
    ("pin_slot", "Pin slot"),
    ("gear", "Gear"),
    ("rack_pinion", "Rack and pinion"),
    ("screw", "Screw"),
    ("cam", "Cam"),
];

pub(super) const MATE_TYPES: &[(&str, &str)] = &[
    ("coincident", "Coincident"),
    ("concentric", "Concentric"),
    ("distance", "Distance"),
    ("angle", "Angle"),
    ("parallel", "Parallel"),
    ("perpendicular", "Perpendicular"),
    ("tangent", "Tangent"),
    ("fasten", "Fasten"),
    ("revolute", "Revolute"),
    ("slider", "Slider"),
    ("cylindrical", "Cylindrical"),
    ("planar", "Planar"),
    ("ball", "Ball"),
    ("pin_slot", "Pin slot"),
    ("width", "Width"),
    ("symmetric", "Symmetric"),
    ("path", "Path"),
    ("linear_coupler", "Linear coupler"),
    ("gear", "Gear"),
    ("rack_pinion", "Rack and pinion"),
    ("screw", "Screw"),
    ("cam", "Cam"),
    ("hinge", "Hinge"),
    ("universal_joint", "Universal joint"),
];

fn occurrences() -> Predicate {
    has(Facet::Occurrences)
}

fn asm(id: &str, label: &str, icon: &str, desc: &str) -> CommandDescriptor {
    cmd(id, label, icon, A).desc(desc)
}

pub(super) fn add(catalog: &mut Catalog) {
    let conn = &[K::MateConnector, K::Face, K::Edge, K::Vertex, K::Occurrence];
    let mut commands = assemble::assemble(conn);
    commands.extend(layout::layout());
    commands.extend(evaluate::evaluate());
    commands.extend(manage::manage());
    catalog.add_commands(commands);
    rows::add_rows(catalog);
}
