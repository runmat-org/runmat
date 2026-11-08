//! CAD design commands: part, sketch, assembly and drawing commands, rows and item schemas,
//! described with the shared descriptor schema of `runmat_design_commands` (re-exported here).
//!
//! [`catalog`] is the CAD catalog with the shared commands merged in; `runmat-circuit` builds
//! its own the same way, and the `runmat-design-commands` crate's tests merge both into the
//! Desktop's `design-commands.json`.

mod assembly;
mod context;
mod drawing;
mod items;
mod part;
mod sketch;
mod view;

pub use runmat_design_commands::*;

/// The CAD catalog: the shared commands plus part, sketch, assembly and drawing commands.
pub fn catalog() -> Catalog {
    let mut c = runmat_design_commands::common();
    view::add(&mut c);
    part::add(&mut c);
    sketch::add(&mut c);
    assembly::add(&mut c);
    drawing::add(&mut c);
    items::add(&mut c);
    context::add(&mut c);
    c.withdraw(UNBUILT);
    c.finish();
    c
}

/// Commands described here whose appliers aren't built yet. They stay out of the catalog, so
/// no toolbar, menu or search offers a button that does nothing; an id leaves this list in the
/// commit that builds its applier, with its test.
pub const UNBUILT: &[&str] = &[
    // Sketch.
    "sketch.fully-define",
    "sketch.use-edges",
    "sketch.stretch",
    "sketch.block",
    "sketch.image",
    "sketch.dxf-import",
    "sketch.repair",
    // Part: reference curves, wrap, bodies to parts and where-used.
    "part.curve-helix",
    "part.curve-projected",
    "part.curve-composite",
    "part.curve-split-line",
    "part.curve-intersection",
    "part.curve-through-points",
    "part.curve-equation",
    "part.wrap",
    "part.body-promote",
    "part.where-used",
    // Surfaces: untrim has no kernel operation yet; replacing solid faces with a surface is
    // Direct's Replace face.
    "surface.untrim",
    "surface.replace-face",
    // Sheet metal the regeneration doesn't build yet: lofted bends, converting a solid, tabs
    // and slots.
    "sheet-metal.lofted-bend",
    "sheet-metal.convert",
    "sheet-metal.tab-and-slot",
    // Assembly fasteners: the runtime publishes no edge paths for a rule to run along and has
    // no catalog of a fastener family's sizes, so rules, smart fasteners, the library and hole
    // series are written in Source until it does.
    "assembly.fastener-rule",
    "assembly.smart-fasteners",
    "assembly.fastener-library",
    "assembly.hole-series",
    // Direct: relations kept while a face moves.
    "direct.live-relations",
    // Manage.
    "manage.equations",
    "manage.find-references",
    "manage.freeze-bar",
    "manage.material",
    "manage.appearance",
    "manage.decal",
    "manage.import",
    "manage.heal",
    "manage.split-to-files",
    "manage.pack-and-share",
    // Drawing: views the model doesn't draw yet (break, crop, alternate position), annotation
    // tools with no model (hatch, revision cloud, sheet sketch, stacked balloons), and settings
    // with no applier.
    "drawing.break",
    "drawing.crop",
    "drawing.alternate-position",
    "drawing.model-items",
    "drawing.auto-balloon",
    "drawing.stacked-balloon",
    "drawing.area-hatch",
    "drawing.revision-cloud",
    "drawing.sketch",
    "table.weld",
    "table.general",
    "sheet.layers",
    "sheet.line-styles",
    "sheet.format",
    "drawing.document-settings",
    "drawing.hide-view",
    "drawing.edit-annotation",
    "drawing.reattach",
    "drawing.print",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FeatureKind;

    #[test]
    fn catalog_is_sound() {
        let c = catalog();
        let problems = c.lint();
        assert!(problems.is_empty(), "{problems:#?}");
    }

    #[test]
    fn every_editor_has_the_ui_target_tabs() {
        let c = catalog();
        let tabs = |editor, mode| -> Vec<String> {
            c.rows
                .iter()
                .find(|r| r.editor == editor && r.mode == mode)
                .map(|r| r.tabs.iter().map(|t| t.label.clone()).collect())
                .unwrap_or_default()
        };
        assert_eq!(
            tabs(EditorKind::Part, EditMode::Model),
            [
                "Features",
                "Sketch",
                "Direct",
                "Surfaces",
                "Sheet Metal",
                "Weldments",
                "Mold",
                "Evaluate",
                "Annotate",
                "Simulate",
                "Manage"
            ]
        );
        assert_eq!(
            tabs(EditorKind::Assembly, EditMode::Model),
            [
                "Assemble",
                "Layout",
                "Fasteners",
                "Motion",
                "Evaluate",
                "Explode",
                "Simulate",
                "Manage"
            ]
        );
        assert_eq!(
            tabs(EditorKind::Drawing, EditMode::Model),
            ["Views", "Annotate", "Tables", "Sheet", "Manage"]
        );
        assert_eq!(tabs(EditorKind::Part, EditMode::Sketch), ["Sketch"]);
    }

    #[test]
    fn every_command_is_placed_or_contextual() {
        // Reached by keys and command search only (Ctrl+B rebuilds; the rollback bar rolls to end).
        const KEYS_ONLY: &[&str] = &["design.rebuild", "part.rollback-end"];
        let c = catalog();
        let reachable = |id: &str| {
            c.rows.iter().any(|r| {
                r.view_cluster.iter().any(|i| i.command_ids().contains(&id))
                    || r.document_bar.iter().any(|i| i.command_ids().contains(&id))
                    || r.shortcut_box.iter().any(|s| s == id)
                    || r.mode_actions.iter().any(|s| s == id)
            })
        };
        let unreachable: Vec<&str> = c
            .commands
            .iter()
            .filter(|command| {
                command.placements.is_empty()
                    && command.context.is_empty()
                    && !reachable(&command.id)
                    && !KEYS_ONLY.contains(&command.id.as_str())
            })
            .map(|command| command.id.as_str())
            .collect();
        assert!(unreachable.is_empty(), "never reachable: {unreachable:?}");
    }

    /// Every feature a command adds is a real `FeatureKind` tag (and sub-kind tag), so the
    /// descriptors can't drift from the model.
    #[test]
    fn feature_edits_name_real_model_kinds() {
        let c = catalog();
        for command in &c.commands {
            let edit = &command.edit;
            if edit.section != Section::Features || edit.op != EditOp::Add {
                continue;
            }
            let item = edit
                .item
                .as_deref()
                .unwrap_or_else(|| panic!("{} adds no item", command.id));
            let mut probe = serde_json::Map::new();
            probe.insert("feature".into(), item.into());
            if let Some(v) = &edit.variant {
                probe.insert(v.tag.clone(), v.value.clone().into());
            }
            let err = serde_json::from_value::<FeatureKind>(serde_json::Value::Object(probe))
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            assert!(
                !err.contains("unknown variant"),
                "{}: {item} {:?} is not a model kind: {err}",
                command.id,
                edit.variant
            );
        }
    }

    #[test]
    fn unbuilt_commands_are_offered_nowhere() {
        let c = catalog();
        let json = c.to_json();
        for id in UNBUILT {
            assert!(c.command(id).is_none(), "{id}");
            assert!(!json.contains(&format!("\"{id}\"")), "{id} is still named in the catalog");
        }
    }

    #[test]
    fn json_round_trips_and_is_deterministic() {
        let a = catalog().to_json();
        let b = catalog().to_json();
        assert_eq!(a, b);
        let back: Catalog = serde_json::from_str(&a).unwrap();
        assert_eq!(back.to_json(), a);
        assert!(a.contains("\"format_version\": 2"));
    }
}
