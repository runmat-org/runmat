//! The CAD view cluster and the view commands of the 3D and drawing editors: selection filter,
//! visibility, standard views, configuration picker. The commands every design editor shares
//! (rebuild, display, grid, section, measure, fit, new study) are `runmat_design_commands::common`.

use super::*;
use EditorKind::*;
use SelectKind as K;

pub(super) const CAD_3D: &[EditorKind] = &[Part, Assembly];

pub(super) fn add(catalog: &mut Catalog) {
    let mut commands = vec![
        cmd(
            "view.select-filter",
            "Selection filter",
            "select-filter",
            CAD_3D,
        )
        .desc("Choose what clicks in the view select.")
        .kw(&["filter", "pick"])
        .inputs(vec![choice(
            "filter",
            "Select",
            &[
                ("any", "Anything"),
                ("body", "Bodies"),
                ("face", "Faces"),
                ("edge", "Edges"),
                ("vertex", "Vertices"),
                ("feature", "Features"),
                ("sketch", "Sketches"),
            ],
        )])
        .edit(view_action("select_filter")),
        cmd("view.visibility", "Visibility", "visibility", CAD_3D)
            .desc("Show or hide planes, sketches, hidden bodies and annotations.")
            .kw(&["hide", "show", "planes"])
            .inputs(vec![
                flag("planes", "Planes", true),
                flag("sketches", "Sketches", true),
                flag("annotations", "Annotations", true),
                flag("hidden", "Hidden items", false),
            ])
            .edit(view_action("visibility")),
        cmd(
            "view.configuration",
            "Configuration",
            "configuration",
            &[Part, Assembly],
        )
        .desc("Switch the active configuration.")
        .kw(&["config", "variant", "design table"])
        .inputs(vec![text("configuration", "Configuration", None)])
        .edit(view_action("configuration")),
        cmd("view.front", "Front", "view-front", CAD_3D)
            .desc("Look at the model from the front.")
            .edit(view_action("front")),
        cmd("view.back", "Back", "view-back", CAD_3D)
            .desc("Look at the model from the back.")
            .edit(view_action("back")),
        cmd("view.top", "Top", "view-top", CAD_3D)
            .desc("Look at the model from the top.")
            .edit(view_action("top")),
        cmd("view.bottom", "Bottom", "view-bottom", CAD_3D)
            .desc("Look at the model from the bottom.")
            .edit(view_action("bottom")),
        cmd("view.left", "Left", "view-left", CAD_3D)
            .desc("Look at the model from the left.")
            .edit(view_action("left")),
        cmd("view.right", "Right", "view-right", CAD_3D)
            .desc("Look at the model from the right.")
            .edit(view_action("right")),
        cmd("view.isometric", "Isometric", "view-iso", CAD_3D)
            .desc("Look at the model from the isometric corner.")
            .kw(&["iso"])
            .edit(view_action("isometric")),
        cmd("view.perspective", "Perspective", "view-perspective", CAD_3D)
            .desc("Draw the view with perspective: nearer parts look larger.")
            .kw(&["projection", "camera"])
            .edit(view_action("perspective")),
        cmd("view.orthographic", "Orthographic", "view-orthographic", CAD_3D)
            .desc("Draw the view as a parallel projection, so sizes don't change with depth.")
            .kw(&["projection", "parallel", "camera"])
            .edit(view_action("orthographic")),
        cmd(
            "view.normal-to",
            "Normal to",
            "view-normal",
            &[Part, Assembly],
        )
        .desc("Look straight at the selected plane, face or sketch.")
        .when(any(vec![
            selected_one(&[K::Plane, K::Face, K::Sketch]),
            in_mode(&[EditMode::Sketch], "Select a plane or face"),
        ]))
        .edit(view_action("normal_to")),
        cmd(
            "view.representation",
            "Representation",
            "representation",
            &[Assembly],
        )
        .desc("Choose how components load and draw in this view.")
        .kw(&["rep", "lightweight", "envelope"])
        .inputs(vec![choice(
            "representation",
            "Representation",
            &[
                ("full", "Full"),
                ("display", "Display only"),
                ("envelope", "Envelope"),
                ("defeatured", "Defeatured"),
            ],
        )])
        .edit(view_action("representation")),
        cmd(
            "view.working-set",
            "Working set",
            "working-set",
            &[Assembly],
        )
        .desc("Load only the selected components and what they touch.")
        .kw(&["isolate", "load"])
        .inputs(vec![
            picks("components", "Components", &[K::Occurrence]),
            flag("neighbors", "Include touching components", true),
        ])
        .edit(view_action("working_set")),
        cmd("assembly.options", "Options", "options", &[Assembly])
            .desc("Pick the product options and effectivity the assembly shows.")
            .kw(&["effectivity", "variant", "serial", "msn"])
            .inputs(vec![
                text("options", "Options", Some("engine=b")),
                text("effectivity", "Effectivity", Some("MSN 42")).optional(),
            ])
            .edit(edit(EditOp::Set, Section::Options).item("active")),
    ];
    for command in &mut commands {
        if command.edit.op == EditOp::View && command.inputs.is_empty() {
            command.preview = false;
        }
    }
    catalog.add_commands(commands);
}

/// The 3D view cluster (part, assembly, circuit 3D): display, filter, visibility, grid, section,
/// standard views, configuration, measure.
pub(super) fn cluster_3d(extra: Vec<RowItem>) -> Vec<RowItem> {
    let mut items = vec![
        variants(
            "view.display.menu",
            "Shaded",
            "display",
            "view.display",
            "style",
            &[
                ("shaded", "Shaded"),
                ("shaded_edges", "Shaded with edges"),
                ("xray", "X-ray edges"),
                ("wireframe", "Wireframe"),
                ("hidden_lines", "Hidden lines removed"),
            ],
        ),
        variants(
            "view.select-filter.menu",
            "Select",
            "select-filter",
            "view.select-filter",
            "filter",
            &[
                ("any", "Anything"),
                ("body", "Bodies"),
                ("face", "Faces"),
                ("edge", "Edges"),
                ("vertex", "Vertices"),
                ("feature", "Features"),
                ("sketch", "Sketches"),
            ],
        ),
        c("view.visibility"),
        c("view.grid"),
        menu(
            "view.section.menu",
            "Section",
            "section",
            vec![e("view.section"), e("view.section-off")],
        ),
        menu(
            "view.orient.menu",
            "View",
            "view-iso",
            vec![
                e("view.isometric"),
                e("view.front"),
                e("view.back"),
                e("view.top"),
                e("view.bottom"),
                e("view.left"),
                e("view.right"),
                e("view.normal-to"),
                e("view.fit"),
                e("view.zoom-selection"),
                e("view.perspective"),
                e("view.orthographic"),
            ],
        ),
    ];
    items.extend(extra);
    items.push(c("view.measure"));
    items
}
