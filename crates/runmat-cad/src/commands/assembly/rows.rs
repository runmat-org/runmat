//! The assembly editor's ribbon rows, tabs and menus.

use super::*;
use super::cluster_3d;
use EditorKind::Assembly;

pub(super) fn add_rows(catalog: &mut Catalog) {
    let mate_menu = variants(
        "assembly.mate.menu",
        "Mate",
        "mate",
        "assembly.mate",
        "type",
        MATE_TYPES,
    );
    catalog.rows.push(EditorRows {
        editor: Assembly,
        mode: EditMode::Model,
        default_tab: "assemble".into(),
        document_bar: vec![
            c("view.configuration"),
            c("assembly.options"),
            c("view.representation"),
            c("view.working-set"),
        ],
        view_cluster: cluster_3d(vec![]),
        shortcut_box: [
            "assembly.insert",
            "assembly.mate",
            "assembly.move",
            "assembly.pattern",
            "assembly.interference",
            "view.measure",
            "view.section",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect(),
        mode_actions: Vec::new(),
        tabs: vec![
            tab(
                "assemble",
                "Assemble",
                vec![
                    group(
                        "components",
                        "Components",
                        vec![
                            variants(
                                "assembly.insert.menu",
                                "Insert",
                                "insert",
                                "assembly.insert",
                                "source",
                                &[
                                    ("part", "Part"),
                                    ("assembly", "Assembly"),
                                    ("circuit", "Circuit board"),
                                    ("step", "STEP file"),
                                    ("library", "Library part"),
                                ],
                            ),
                            c("assembly.new-part"),
                            c("assembly.replace"),
                        ],
                    ),
                    group(
                        "mates",
                        "Mates",
                        vec![
                            mate_menu,
                            c("assembly.mate-connector"),
                            c("assembly.smart-mate"),
                        ],
                    ),
                    group(
                        "motion",
                        "Motion",
                        vec![
                            variants(
                                "assembly.move.menu",
                                "Move",
                                "move",
                                "assembly.move",
                                "mode",
                                &[
                                    ("drag", "Free drag"),
                                    ("triad", "Move with triad"),
                                    ("align", "Move along an entity"),
                                ],
                            ),
                            c("assembly.fix"),
                        ],
                    ),
                    group(
                        "repeat",
                        "Repeat",
                        vec![
                            variants(
                                "assembly.pattern.menu",
                                "Pattern",
                                "pattern",
                                "assembly.pattern",
                                "layout",
                                &[
                                    ("linear", "Linear pattern"),
                                    ("circular", "Circular pattern"),
                                    ("feature", "Follow feature pattern"),
                                ],
                            ),
                            c("assembly.mirror"),
                            c("assembly.copy-with-mates"),
                        ],
                    ),
                    group(
                        "structure",
                        "Structure",
                        vec![
                            variants(
                                "assembly.subassembly.menu",
                                "Subassembly",
                                "subassembly",
                                "assembly.subassembly",
                                "action",
                                &[
                                    ("form", "Form new subassembly"),
                                    ("dissolve", "Dissolve"),
                                    ("flexible", "Make flexible"),
                                    ("rigid", "Make rigid"),
                                ],
                            ),
                            c("assembly.edit-in-context"),
                        ],
                    ),
                ],
            ),
            tab(
                "layout",
                "Layout",
                vec![
                    group(
                        "skeleton",
                        "Skeleton",
                        vec![
                            c("assembly.skeleton"),
                            c("assembly.publish"),
                            c("assembly.envelope"),
                            c("assembly.interface-check"),
                        ],
                    ),
                    group(
                        "features",
                        "Features",
                        vec![c("assembly.feature"), c("assembly.hole-series")],
                    ),
                ],
            ),
            tab(
                "fasteners",
                "Fasteners",
                vec![group(
                    "fasteners",
                    "Fasteners",
                    vec![
                        c("assembly.smart-fasteners"),
                        c("assembly.fastener-rule"),
                        menu(
                            "assembly.fastener-library.menu",
                            "Library",
                            "library",
                            vec![e("assembly.fastener-library")],
                        ),
                        c("assembly.hole-check"),
                    ],
                )],
            ),
            tab(
                "motion",
                "Motion",
                vec![group(
                    "motion",
                    "Motion",
                    vec![
                        c("motion.drag"),
                        c("motion.collision"),
                        c("motion.mate-controller"),
                        c("motion.animation"),
                        c("motion.basic"),
                        c("motion.to-fea"),
                    ],
                )],
            ),
            tab(
                "evaluate",
                "Evaluate",
                vec![
                    group(
                        "check",
                        "Check",
                        vec![
                            c("assembly.interference"),
                            c("assembly.clearance"),
                            c("assembly.hole-alignment"),
                        ],
                    ),
                    group(
                        "properties",
                        "Properties",
                        vec![
                            c("assembly.mass-properties"),
                            c("assembly.bom"),
                            c("assembly.where-used"),
                            c("assembly.visualization"),
                        ],
                    ),
                ],
            ),
            tab(
                "explode",
                "Explode",
                vec![group(
                    "explode",
                    "Explode",
                    vec![
                        c("explode.new"),
                        c("explode.step"),
                        c("explode.radial"),
                        c("explode.auto"),
                        c("explode.lines"),
                    ],
                )],
            ),
            tab(
                "simulate",
                "Simulate",
                vec![group(
                    "simulate",
                    "Simulate",
                    vec![
                        c("assembly.contacts"),
                        c("assembly.representation"),
                        c("study.new"),
                    ],
                )],
            ),
            tab(
                "manage",
                "Manage",
                vec![
                    group(
                        "parameters",
                        "Parameters",
                        vec![
                            c("assembly.options-effectivity"),
                            c("assembly.configurations"),
                            c("assembly.document-settings"),
                        ],
                    ),
                    group("references", "References", vec![c("assembly.update-pins")]),
                    group(
                        "display",
                        "Display",
                        vec![
                            c("assembly.isolate"),
                            c("assembly.hide"),
                            c("assembly.transparency"),
                        ],
                    ),
                    group(
                        "files",
                        "Files",
                        vec![
                            c("assembly.pack-and-share"),
                            variants(
                                "assembly.export.menu",
                                "Export",
                                "export",
                                "assembly.export",
                                "format",
                                &[("step", "STEP assembly"), ("gltf", "glTF"), ("usd", "USD")],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    });
}
