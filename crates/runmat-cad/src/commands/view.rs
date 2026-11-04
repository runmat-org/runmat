//! The view cluster and the commands every design editor shares: rebuild, measure, section,
//! display, selection filter, standard views, configuration picker, new study.

use super::*;
use EditorKind::*;
use SelectKind as K;

pub(super) const CAD_3D: &[EditorKind] = &[Part, Assembly];
pub(super) const DESIGN: &[EditorKind] = &[Part, Assembly, Drawing, Circuit, CircuitLayout];

pub(super) fn add(catalog: &mut Catalog) {
    let mut commands = vec![
        cmd("design.rebuild", "Rebuild", "rebuild", DESIGN)
            .desc("Rebuild the file and refresh its checks.")
            .kw(&["regenerate", "update", "recompute"])
            .edit(check("rebuild")),
        cmd(
            "view.display",
            "Display style",
            "display",
            &[Part, Assembly, Circuit, CircuitLayout],
        )
        .desc("Choose how the model is drawn.")
        .kw(&["shaded", "wireframe", "hidden lines", "x-ray"])
        .inputs(vec![
            choice(
                "style",
                "Style",
                &[
                    ("shaded", "Shaded"),
                    ("shaded_edges", "Shaded with edges"),
                    ("xray", "X-ray edges"),
                    ("wireframe", "Wireframe"),
                    ("hidden_lines", "Hidden lines removed"),
                ],
            ),
            flag("perspective", "Perspective", false),
        ])
        .edit(view_action("display")),
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
            "view.grid",
            "Grid",
            "grid",
            &[Part, Assembly, Drawing, Circuit, CircuitLayout],
        )
        .desc("Show the grid and set its spacing.")
        .inputs(vec![
            flag("show", "Show the grid", true),
            length("spacing", "Spacing", 10.0),
        ])
        .edit(view_action("grid")),
        cmd(
            "view.section",
            "Section view",
            "section",
            &[Part, Assembly, Circuit, CircuitLayout],
        )
        .desc("Cut the view with one or more section planes.")
        .kw(&["cut", "clip", "cross section"])
        .inputs(vec![
            choice(
                "plane",
                "Plane",
                &[
                    ("front", "Front"),
                    ("top", "Top"),
                    ("right", "Right"),
                    ("selection", "Selected plane or face"),
                ],
            ),
            pick("reference", "Reference", &[K::Plane, K::Face]).shown("plane", &["selection"]),
            offset("offset", "Offset", 0.0).drag(Manipulator::Arrow),
            flag("second", "Second plane", false),
            choice(
                "second_plane",
                "Second",
                &[("front", "Front"), ("top", "Top"), ("right", "Right")],
            )
            .if_checked("second"),
            flag("cap", "Cap the cut", true),
        ])
        .preview()
        .edit(view_action("section")),
        cmd(
            "view.section-off",
            "Section off",
            "section-off",
            &[Part, Assembly, Circuit, CircuitLayout],
        )
        .desc("Remove the section planes.")
        .edit(view_action("section_off")),
        cmd(
            "view.measure",
            "Measure",
            "measure",
            &[Part, Assembly, Drawing, Circuit, CircuitLayout],
        )
        .desc("Measure distance, angle, radius or path length; results stay pinned in the view.")
        .kw(&["distance", "angle", "radius", "dimension", "ruler"])
        .inputs(vec![
            radio(
                "mode",
                "Measure",
                &[
                    ("distance", "Distance"),
                    ("angle", "Angle"),
                    ("radius", "Radius"),
                    ("path", "Path length"),
                ],
            ),
            picks(
                "items",
                "Items",
                &[
                    K::Face,
                    K::Edge,
                    K::Vertex,
                    K::Body,
                    K::Occurrence,
                    K::Pad,
                    K::Track,
                    K::Via,
                ],
            ),
            flag("pin", "Pin the result", true),
        ])
        .preview()
        .edit(analyze("measure")),
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
        cmd("view.fit", "Fit", "fit", DESIGN)
            .desc("Fit the whole model in the view.")
            .kw(&["zoom", "extents"])
            .edit(view_action("fit")),
        cmd(
            "view.zoom-selection",
            "Zoom to selection",
            "zoom-selection",
            DESIGN,
        )
        .desc("Frame the selected items.")
        .when(selected(&[]))
        .edit(view_action("zoom_selection")),
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
        cmd(
            "study.new",
            "New study",
            "study",
            &[Part, Assembly, Circuit, CircuitLayout],
        )
        .desc("Create a .fea study from this design.")
        .kw(&["simulation", "fea", "analysis", "thermal"])
        .inputs(vec![
            choice(
                "physics",
                "Physics",
                &[
                    ("structural", "Structural"),
                    ("modal", "Modal"),
                    ("thermal", "Thermal"),
                    ("ir_drop", "IR drop"),
                    ("thermal_structural", "Thermal and structural"),
                ],
            ),
            text("name", "File name", Some("study.fea")),
        ])
        .edit(edit(EditOp::CreateFile, Section::Studies).item("fea")),
        cmd("design.exit-nested", "Exit", "exit", DESIGN)
            .desc("Leave the nested edit and return to the level above.")
            .when(in_mode(
                &[EditMode::Sketch, EditMode::Sketch3d, EditMode::InContext],
                "Nothing nested is being edited",
            ))
            .edit(edit(EditOp::ExitMode, Section::None)),
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
            ],
        ),
    ];
    items.extend(extra);
    items.push(c("view.configuration"));
    items.push(c("view.measure"));
    items
}
