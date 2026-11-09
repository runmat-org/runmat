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
    ("width", "Width"),
    ("symmetric", "Symmetric"),
    ("path", "Path"),
    ("linear_coupler", "Linear coupler"),
    ("hinge", "Hinge"),
    ("universal_joint", "Universal joint"),
];

/// The kinds Mate adds from its picks: the standard mates and every connector mate the solver
/// has. A cam's profile and a path mate's path are closed or open paths the part publishes
/// (Publish, Path); width and symmetric take their extra picks.
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
    ("gear", "Gear"),
    ("rack_pinion", "Rack and pinion"),
    ("screw", "Screw"),
    ("cam", "Cam"),
    ("width", "Width"),
    ("symmetric", "Symmetric"),
    ("path", "Path"),
    ("linear_coupler", "Linear coupler"),
    ("hinge", "Hinge"),
    ("universal_joint", "Universal joint"),
];

/// What a mate's connector can be picked on.
pub(super) const CONNECTOR_PICKS: &[SelectKind] = &[K::MateConnector, K::Face, K::Edge, K::Vertex, K::Occurrence];

/// The fields the advanced mates add to Mate and Edit mate: width's two other faces, the
/// symmetry plane, the cam's profile, the path's edges and the universal joint's point. In
/// Edit mate the picks are optional (a mate that already has them keeps them).
pub(super) fn advanced_mate_fields(conn: &[SelectKind], required: bool) -> Vec<Field> {
    let need = |f: Field| if required { f } else { f.optional() };
    vec![
        need(pick("first2", "Groove's other face", conn).shown("type", &["width"])),
        need(pick("second2", "Tab's other face", conn).shown("type", &["width"])),
        need(pick("plane", "Symmetry plane", conn).shown("type", &["symmetric"])),
        text("profile", "Profile", Some("lobe")).shown("type", &["cam"]),
        // The path: edges of the first pick's component, picked in the view or the tree; the
        // applier publishes them from its part.
        need(picks("path", "Path", &[K::Edge]).shown("type", &["path"])),
        flag("joint_point", "Joint point", true).shown("type", &["universal_joint"]),
    ]
}

fn occurrences() -> Predicate {
    has(Facet::Occurrences)
}

fn asm(id: &str, label: &str, icon: &str, desc: &str) -> CommandDescriptor {
    cmd(id, label, icon, A).desc(desc)
}

pub(super) fn add(catalog: &mut Catalog) {
    let conn = CONNECTOR_PICKS;
    let mut commands = assemble::assemble(conn);
    commands.extend(layout::layout());
    commands.extend(evaluate::evaluate());
    commands.extend(manage::manage());
    catalog.add_commands(commands);
    rows::add_rows(catalog);
}
