//! Right-click menus for part, assembly and drawing items (UI target decision 12), and the
//! commands only those menus reach: the best of SolidWorks, Onshape and Altium per item kind.
//! Each menu is quick icon actions, grouped commands with `▸` submenus, then Agent ▸.

use super::*;
use EditorKind::*;
use SelectKind as K;

const P: &[EditorKind] = &[Part];
const A: &[EditorKind] = &[Assembly];
const D: &[EditorKind] = &[Drawing];

/// A submenu entry that runs `command` as is under its own label.
fn labeled(command: &str, label: &str) -> MenuEntry {
    MenuEntry {
        command: command.to_owned(),
        label: Some(label.to_owned()),
        values: Default::default(),
    }
}

pub(super) fn add(catalog: &mut Catalog) {
    let tree = &[K::Feature, K::Sketch];
    let shown = &[K::Feature, K::Sketch, K::Body];
    catalog.add_commands(vec![
        // Part.
        cmd("part.move-up", "Move up", "move-up", P)
            .desc("Move the feature one place earlier in the tree.")
            .when(selected_one(tree))
            .edit(edit(EditOp::Reorder, Section::Features).variant("direction", "up")),
        cmd("part.move-down", "Move down", "move-down", P)
            .desc("Move the feature one place later in the tree.")
            .when(selected_one(tree))
            .edit(edit(EditOp::Reorder, Section::Features).variant("direction", "down")),
        cmd("part.parents-children", "Parents and children", "references", P)
            .desc("List what the item is built on and what is built on it.")
            .kw(&["dependencies", "references"])
            .when(selected_one(&[K::Feature, K::Sketch, K::Body]))
            .edit(analyze("parents_children")),
        cmd("part.where-used", "Where used", "where-used", P)
            .desc("Find the files and features that use this item, through references and publishes.")
            .when(selected_one(&[K::Feature, K::Sketch, K::Body, K::Input]))
            .edit(analyze("where_used")),
        cmd("part.hide", "Hide/show", "hide", P)
            .desc("Hide the selected items in the view, or show them again.")
            .kw(&["visibility", "show"])
            .when(selected(shown))
            .edit(view_action("hide")),
        cmd("part.isolate", "Isolate", "isolate", P)
            .desc("Show only the selected items until isolate ends.")
            .when(selected(shown))
            .edit(view_action("isolate")),
        cmd("part.edit-sketch-plane", "Edit plane", "plane", P)
            .desc("Move the sketch to another plane or planar face.")
            .inputs(vec![pick("plane", "Plane", &[K::Plane, K::Face])])
            .when(selected_one(&[K::Sketch]))
            .edit(edit(EditOp::Edit, Section::Sketch).item("plane")),
        cmd("part.section-here", "Section here", "section", P)
            .desc("Cut the view through the selected body, face or plane.")
            .when(selected_one(&[K::Body, K::Face, K::Plane]))
            .edit(view_action("section")),
        cmd("part.export-body", "Export body", "export", P)
            .desc("Write the selected bodies to a STEP, Parasolid or STL file.")
            .inputs(vec![choice(
                "format",
                "Format",
                &[("step", "STEP"), ("x_t", "Parasolid"), ("stl", "STL")],
            )])
            .when(selected(&[K::Body]))
            .edit(export("body")),
        // Assembly.
        cmd("assembly.component-configuration", "Configuration", "configuration", A)
            .desc("Use another configuration of the component's file.")
            .inputs(vec![text("configuration", "Configuration", Some("default"))])
            .when(selected(&[K::Occurrence]))
            .edit(edit(EditOp::Set, Section::Occurrences).item("configuration")),
        cmd("assembly.show-mates", "Show mates", "mate", A)
            .desc("Highlight the component's mates and list them in Properties.")
            .when(selected_one(&[K::Occurrence]))
            .edit(view_action("show_mates")),
        cmd("assembly.flip-mate", "Flip", "flip", A)
            .desc("Flip the mate's alignment.")
            .when(selected(&[K::Mate]))
            .edit(edit(EditOp::Set, Section::Mates).item("flip")),
        cmd("assembly.show-mate-entities", "Show entities", "highlight", A)
            .desc("Highlight the faces, edges and connectors the mate uses.")
            .when(selected_one(&[K::Mate]))
            .edit(view_action("show_mate_entities")),
        // Drawing.
        cmd("drawing.view-scale", "Scale", "scale", D)
            .desc("Use the sheet scale or a custom scale for the view.")
            .inputs(vec![choice(
                "scale",
                "Scale",
                &[
                    ("sheet", "Sheet scale"),
                    ("1:1", "1:1"),
                    ("1:2", "1:2"),
                    ("1:5", "1:5"),
                    ("1:10", "1:10"),
                    ("2:1", "2:1"),
                ],
            )])
            .when(selected(&[K::View]))
            .edit(edit(EditOp::Set, Section::Views).item("scale")),
        cmd("drawing.view-display", "Display style", "display", D)
            .desc("Draw the view with hidden lines removed or visible, or shaded.")
            .inputs(vec![choice(
                "style",
                "Style",
                &[
                    ("hidden_removed", "Hidden lines removed"),
                    ("hidden_visible", "Hidden lines visible"),
                    ("wireframe", "Wireframe"),
                    ("shaded_edges", "Shaded with edges"),
                    ("shaded", "Shaded"),
                ],
            )])
            .when(selected(&[K::View]))
            .edit(edit(EditOp::Set, Section::Views).item("display")),
        cmd("drawing.hide-view", "Hide", "hide", D)
            .desc("Hide the view on the sheet; its dimensions stay with it.")
            .when(selected(&[K::View]))
            .edit(edit(EditOp::Set, Section::Views).item("hidden")),
        cmd("drawing.edit-annotation", "Edit", "edit", D)
            .desc("Open the annotation's text and settings in Properties.")
            .when(selected_one(&[K::Annotation, K::DrawingDimension]))
            .edit(edit(EditOp::Edit, Section::Annotations)),
        cmd("drawing.reattach", "Reattach", "reference", D)
            .desc("Attach the annotation to another edge, face or point.")
            .inputs(vec![pick("target", "Attach to", &[K::Edge, K::Face, K::Vertex, K::Point])])
            .when(selected_one(&[K::Annotation, K::DrawingDimension]))
            .edit(edit(EditOp::Set, Section::Annotations).item("attachment")),
        cmd("drawing.show-in-model", "Show in model", "open", D)
            .desc("Open the model with the annotated geometry selected.")
            .when(selected_one(&[K::Annotation, K::DrawingDimension, K::View]))
            .edit(view_action("show_in_model")),
    ]);

    let appearance = menu(
        "appearance",
        "Appearance",
        "appearance",
        choice_pairs(catalog.command("manage.appearance").expect("appearance"), "finish")
            .iter()
            .map(|(value, label)| preset("manage.appearance", label, "finish", value))
            .chain([labeled("manage.appearance", "Edit appearance…")])
            .collect(),
    );
    let material = menu(
        "material",
        "Material",
        "material",
        vec![
            preset("manage.material", "Aluminum 6061-T6", "material", "Al 6061-T6"),
            preset("manage.material", "Steel 1018", "material", "Steel 1018"),
            preset("manage.material", "Stainless 304", "material", "SS 304"),
            preset("manage.material", "ABS", "material", "ABS"),
            labeled("manage.material", "Edit material…"),
        ],
    );
    let configuration = menu(
        "component-configuration",
        "Configuration",
        "configuration",
        vec![
            preset("assembly.component-configuration", "Default", "configuration", "default"),
            labeled("assembly.component-configuration", "Choose…"),
        ],
    );
    let representation = choice_submenu(
        catalog,
        "component-representation",
        "Representation",
        "representation",
        "assembly.representation",
        "representation",
    );
    let scale = choice_submenu(catalog, "view-scale", "Scale", "scale", "drawing.view-scale", "scale");
    let display = choice_submenu(
        catalog,
        "view-display",
        "Display style",
        "display",
        "drawing.view-display",
        "style",
    );

    catalog.context_menus.extend([
        context_menu(
            K::Feature,
            P,
            &[
                "part.edit-feature",
                "part.edit-sketch",
                "part.suppress",
                "part.rollback-here",
                "part.isolate",
                "design.show-source",
            ],
            vec![
                vec![c("part.edit-feature"), c("part.edit-sketch")],
                vec![
                    c("part.suppress"),
                    c("part.unsuppress"),
                    c("part.rollback-here"),
                    c("part.rollback-end"),
                ],
                vec![
                    c("part.rename"),
                    c("design.show-source"),
                    c("part.move-up"),
                    c("part.move-down"),
                ],
                vec![c("part.parents-children"), c("part.where-used")],
                vec![c("part.hide"), c("part.isolate"), appearance.clone()],
                vec![c("design.copy-reference")],
                vec![c("part.delete")],
            ],
        ),
        context_menu(
            K::Sketch,
            P,
            &[
                "part.edit-sketch",
                "part.edit-sketch-plane",
                "part.hide",
                "sketch.fully-define",
                "sketch.check",
            ],
            vec![
                vec![c("part.edit-sketch"), c("part.edit-sketch-plane")],
                vec![c("part.hide")],
                vec![c("sketch.fully-define"), c("sketch.check")],
                vec![c("part.rename"), c("design.show-source"), c("design.copy-reference")],
                vec![c("part.delete")],
            ],
        ),
        context_menu(
            K::Body,
            P,
            &[
                "part.hide",
                "part.isolate",
                "manage.appearance",
                "part.section-here",
                "evaluate.mass-properties",
            ],
            vec![
                vec![c("part.hide"), c("part.isolate")],
                vec![appearance, material],
                vec![c("part.body-move"), c("part.section-here")],
                vec![c("evaluate.mass-properties")],
                vec![c("part.body-promote"), c("part.export-body")],
                vec![c("part.rename"), c("design.copy-reference")],
                vec![c("part.body-delete")],
            ],
        ),
        context_menu(
            K::Occurrence,
            A,
            &[
                "assembly.open",
                "assembly.edit-in-context",
                "assembly.hide",
                "assembly.isolate",
                "assembly.fix",
                "assembly.mate",
            ],
            vec![
                vec![c("assembly.open"), c("assembly.edit-in-context")],
                vec![c("assembly.hide"), c("assembly.isolate"), c("assembly.transparency")],
                vec![c("assembly.fix"), configuration, representation],
                vec![c("assembly.replace"), c("assembly.pattern")],
                vec![c("assembly.mate"), c("assembly.show-mates")],
                vec![c("assembly.where-used"), c("design.copy-reference")],
                vec![c("assembly.delete")],
            ],
        ),
        context_menu(
            K::Mate,
            A,
            &[
                "assembly.edit-mate",
                "assembly.suppress",
                "assembly.flip-mate",
                "assembly.show-mate-entities",
            ],
            vec![
                vec![c("assembly.edit-mate"), c("assembly.suppress"), c("assembly.flip-mate")],
                vec![c("assembly.show-mate-entities"), c("design.show-source")],
                vec![c("assembly.delete")],
            ],
        ),
        context_menu(
            K::View,
            D,
            &[
                "drawing.edit-view",
                "drawing.projected",
                "drawing.section",
                "drawing.detail",
                "drawing.model-items",
                "drawing.hide-view",
            ],
            vec![
                vec![c("drawing.edit-view")],
                vec![c("drawing.projected"), c("drawing.section"), c("drawing.detail")],
                vec![scale, display],
                vec![c("drawing.model-items"), c("drawing.show-in-model")],
                vec![c("drawing.hide-view"), c("design.copy-reference")],
                vec![c("drawing.delete")],
            ],
        ),
        context_menu(
            K::Annotation,
            D,
            &[
                "drawing.edit-annotation",
                "drawing.reattach",
                "drawing.show-in-model",
                "design.show-source",
            ],
            vec![
                vec![c("drawing.edit-annotation"), c("drawing.reattach")],
                vec![c("drawing.show-in-model"), c("design.show-source")],
                vec![c("drawing.delete")],
            ],
        ),
    ]);
}
