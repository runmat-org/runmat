//! The commands every design editor shares, CAD and circuit alike: rebuild, display style, grid,
//! section, measure, fit, zoom to selection, new study and leaving a nested edit. Each domain's
//! catalog merges these in, so its rows can name them and its lint sees them.

use super::*;
use EditorKind::*;
use SelectKind as K;

const DESIGN: &[EditorKind] = &[Part, Assembly, Drawing, Circuit, CircuitLayout];

/// The shared commands, as a catalog to merge.
pub fn common() -> Catalog {
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
    let mut catalog = Catalog::empty();
    catalog.add_commands(commands);
    catalog.finish();
    catalog
}
