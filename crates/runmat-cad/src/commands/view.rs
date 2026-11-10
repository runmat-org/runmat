//! The CAD view cluster and the view commands of the 3D and drawing editors: selection filter,
//! visibility, standard views, configuration picker. The commands every design editor shares
//! (rebuild, display, grid, section, measure, fit, new study) are `runmat_design_commands::common`.

use super::*;
use EditorKind::*;
use SelectKind as K;

pub(super) const CAD_3D: &[EditorKind] = &[Part, Assembly];
/// The 3D editors plus the .fea study view, which shares their viewport toolbar.
const VIEW_3D: &[EditorKind] = &[Part, Assembly, Fea];

pub(super) fn add(catalog: &mut Catalog) {
    let mut commands = vec![
        cmd(
            "view.select-filter",
            "Selection filter",
            "select-filter",
            VIEW_3D,
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
        cmd("view.visibility", "Visibility", "visibility", VIEW_3D)
            .desc("Hide or show the selected bodies, components or regions, or show everything again.")
            .kw(&["hide", "show", "unhide"])
            .inputs(vec![choice(
                "action",
                "Action",
                &[
                    ("hide", "Hide selected"),
                    ("show", "Show selected"),
                    ("show_all", "Show all"),
                ],
            )])
            .edit(view_action("visibility")),
        cmd("view.configuration", "Configuration", "configuration", &[Part])
        .desc("Switch the active configuration.")
        .kw(&["config", "variant", "design table"])
        .inputs(vec![text("configuration", "Configuration", None)])
        .edit(view_action("configuration")),
        cmd("view.front", "Front", "view-front", VIEW_3D)
            .desc("Look at the model from the front.")
            .edit(view_action("front")),
        cmd("view.back", "Back", "view-back", VIEW_3D)
            .desc("Look at the model from the back.")
            .edit(view_action("back")),
        cmd("view.top", "Top", "view-top", VIEW_3D)
            .desc("Look at the model from the top.")
            .edit(view_action("top")),
        cmd("view.bottom", "Bottom", "view-bottom", VIEW_3D)
            .desc("Look at the model from the bottom.")
            .edit(view_action("bottom")),
        cmd("view.left", "Left", "view-left", VIEW_3D)
            .desc("Look at the model from the left.")
            .edit(view_action("left")),
        cmd("view.right", "Right", "view-right", VIEW_3D)
            .desc("Look at the model from the right.")
            .edit(view_action("right")),
        cmd("view.isometric", "Isometric", "view-iso", VIEW_3D)
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
        .when(unavailable(
            "Per-view representations aren't available yet. Set a component's representation in its Properties.",
        ))
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
    catalog.add_commands(fea_commands());
    catalog.rows.push(fea_rows());
}

/// The .fea study view's own commands: each opens Source at the declaration it names, shows or
/// hides a kind of setup symbol, runs the study or shows its results. Loads, supports and the
/// mesh are written in Source; the study view draws what the checked study holds.
fn fea_commands() -> Vec<CommandDescriptor> {
    let source = |id: &str, label: &str, icon: &str, item: &str, desc: &str| {
        cmd(id, label, icon, &[Fea]).desc(desc).edit(view_action(item))
    };
    let show = |id: &str, label: &str, icon: &str, item: &str, desc: &str| {
        cmd(id, label, icon, &[Fea])
            .desc(desc)
            .kw(&["show", "hide", "visibility", "symbols"])
            .inputs(vec![flag("show", label, true)])
            .edit(view_action(item))
    };
    vec![
        source("fea.geometry", "Geometry", "body", "source_geometry", "Open the geometry declaration: the CAD file and its named regions."),
        source("fea.materials", "Materials", "material", "source_material", "Open the material declarations and their assignments."),
        source("fea.source", "Edit source", "source", "source", "Open the selected item's declaration in Source, or the study's."),
        source("fea.mesh", "Mesh settings", "settings", "source_mesh", "Open the mesh declaration: element order, target size and deviation."),
        source("fea.loads", "Edit loads", "edit", "source_loads", "Open the model's loads, supports and conditions in Source."),
        show("fea.show-loads", "Loads", "pull", "show_loads", "Show the load arrows and their values on the model."),
        show("fea.show-supports", "Supports", "fix", "show_supports", "Show the ground symbols on held faces."),
        show("fea.show-conditions", "Other conditions", "heat", "show_conditions", "Show thermal, electrical and other boundary symbols."),
        show("fea.show-tints", "Region tints", "highlight", "show_tints", "Tint each loaded, held or bonded face by its role."),
        source("fea.analysis", "Analysis", "study", "source_analysis", "Open the study's analysis, outputs and assessments."),
        cmd("fea.run", "Run study", "simulate", &[Fea])
            .desc("Mesh and solve the study.")
            .kw(&["solve", "simulate", "start"])
            .edit(view_action("run")),
        cmd("fea.results", "Results", "visualization", &[Fea])
            .desc("Show the last run's result fields.")
            .kw(&["fields", "stress", "displacement"])
            .edit(view_action("results")),
    ]
}

/// The .fea study view's rows, built like the part editor's: tabs for setup, mesh, loads and
/// supports, the study and its results, and the 3D view cluster with the items a meshed study
/// supports. Visibility hides picked regions, or bodies when the filter selects bodies, and shows
/// or hides each kind of setup symbol. Results-only controls stay in the results view.
fn fea_rows() -> EditorRows {
    let mut view_cluster = vec![
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
            ],
        ),
        variants(
            "view.select-filter.menu",
            "Select",
            "select-filter",
            "view.select-filter",
            "filter",
            &[("face", "Faces"), ("edge", "Edges"), ("body", "Bodies")],
        ),
        variants(
            "view.visibility.menu",
            "Visibility",
            "visibility",
            "view.visibility",
            "action",
            &[
                ("hide", "Hide selected"),
                ("show", "Show selected"),
                ("show_all", "Show all"),
            ],
        ),
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
                e("view.fit"),
            ],
        ),
        c("view.measure"),
    ];
    // Visibility also shows or hides each kind of setup symbol.
    if let Some(RowItem::Menu { entries, .. }) = view_cluster
        .iter_mut()
        .find(|item| matches!(item, RowItem::Menu { id, .. } if id == "view.visibility.menu"))
    {
        entries.extend(["fea.show-loads", "fea.show-supports", "fea.show-conditions", "fea.show-tints"].map(e));
    }
    let tabs = vec![
        tab("setup", "Setup", vec![group("declarations", "Declarations", vec![c("fea.geometry"), c("fea.materials"), c("fea.source")])]),
        tab("mesh", "Mesh", vec![group("mesh", "Mesh", vec![c("fea.mesh")])]),
        tab(
            "loads",
            "Loads and supports",
            vec![
                group("show", "Show", vec![c("fea.show-loads"), c("fea.show-supports"), c("fea.show-conditions"), c("fea.show-tints")]),
                group("edit", "Edit", vec![c("fea.loads")]),
            ],
        ),
        tab("study", "Study", vec![group("study", "Study", vec![c("fea.analysis"), c("fea.run")])]),
        tab("results", "Results", vec![group("results", "Results", vec![c("fea.results")])]),
    ];
    EditorRows {
        editor: Fea,
        mode: EditMode::Model,
        tabs,
        default_tab: "setup".to_owned(),
        view_cluster,
        document_bar: Vec::new(),
        shortcut_box: Vec::new(),
        mode_actions: Vec::new(),
    }
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
        variants(
            "view.visibility.menu",
            "Visibility",
            "visibility",
            "view.visibility",
            "action",
            &[
                ("hide", "Hide selected"),
                ("show", "Show selected"),
                ("show_all", "Show all"),
            ],
        ),
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
