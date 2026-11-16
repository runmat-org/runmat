//! Assembly commands on the Manage tab: configurations, publishing, export and files.

use super::*;
use SelectKind as K;

pub(super) fn manage() -> Vec<CommandDescriptor> {
    vec![
        // Manage.
        asm(
            "assembly.options-effectivity",
            "Options and effectivity",
            "options",
            "Define product options and the serial ranges they apply to.",
        )
        .inputs(vec![
            text("option", "Option", Some("engine")).help(
                "An existing option's name changes it.",
            ),
            text("values", "Values", Some("a, b")).help(
                "The values the option takes, the first one standard; `1-99` makes a serial-number range.",
            ),
            text("effectivity", "Effectivity", Some("MSN 1-99")).optional().help(
                "The serial range the option's other values apply in, as `MSN 1-99` (or `LOT 3-5`); left empty, they apply everywhere.",
            ),
        ])
        .on(&[K::ProductOption])
        .edit(edit(EditOp::Add, Section::Options).item("option")),
        asm(
            "assembly.configurations",
            "Configurations and display states",
            "configuration",
            "Add or change a configuration (input values and suppressed components) or a display state (hidden components).",
        )
        .inputs(vec![
            radio(
                "kind",
                "Add",
                &[
                    ("configuration", "Configuration"),
                    ("display_state", "Display state"),
                ],
            ),
            text("name", "Name", Some("open"))
                .help("An existing configuration's or display state's name changes it."),
            text("values", "Input values", None)
                .optional()
                .shown("kind", &["configuration"])
                .help("The assembly's inputs this configuration sets, as `width = 40 mm, depth = 20 mm`."),
            picks("suppress", "Suppressed components", &[K::Occurrence])
                .optional()
                .shown("kind", &["configuration"]),
            picks("hidden", "Hidden components", &[K::Occurrence])
                .optional()
                .shown("kind", &["display_state"])
                .help("The components the view leaves out while this display state is applied."),
        ])
        .on(&[K::Configuration, K::DisplayState])
        .edit(edit(EditOp::Add, Section::Configurations).item("configuration")),
        asm(
            "assembly.document-settings",
            "Document settings",
            "settings",
            "Units, standard and default tolerances.",
        )
        .inputs(vec![choice(
            "units",
            "Units",
            &[
                ("mmgs", "Millimeter, gram, second"),
                ("mks", "Meter, kilogram, second"),
                ("ips", "Inch, pound, second"),
            ],
        )])
        .edit(edit(EditOp::Set, Section::Settings).item("document")),
        asm(
            "assembly.release",
            "Release",
            "release",
            "Save the selected components' files, or this assembly, as their next released revision.",
        )
        .kw(&["revision", "release", "freeze", "version"])
        .inputs(vec![picks("components", "Components", &[K::Occurrence])
            .optional()
            .help("Leave empty to release this assembly.")])
        .on(&[K::Occurrence])
        .edit(edit(EditOp::Set, Section::Occurrences).item("release")),
        asm(
            "assembly.update-pins",
            "References and pins",
            "pin",
            "Pin components to their file's latest released revision, or set them to follow the file live.",
        )
        .kw(&["revision", "update", "version"])
        .inputs(vec![
            picks("components", "Components", &[K::Occurrence])
                .optional()
                .help("Leave empty to update every component."),
            radio(
                "to",
                "Update to",
                &[("latest", "Latest"), ("released", "Latest released")],
            ),
        ])
        .when(occurrences())
        .edit(edit(EditOp::Set, Section::Occurrences).item("revision")),
        asm(
            "assembly.isolate",
            "Isolate",
            "isolate",
            "Show only the selected components.",
        )
        .when(selected(&[K::Occurrence]))
        .on(&[K::Occurrence])
        .edit(view_action("isolate")),
        asm(
            "assembly.hide",
            "Hide/show",
            "visibility",
            "Hide or show components.",
        )
        .when(selected(&[K::Occurrence]))
        .on(&[K::Occurrence])
        .edit(view_action("hide")),
        asm(
            "assembly.transparency",
            "Transparency",
            "transparency",
            "Draw components see-through.",
        )
        .when(selected(&[K::Occurrence]))
        .on(&[K::Occurrence])
        .edit(view_action("transparency")),
        asm(
            "assembly.suppress",
            "Suppress",
            "suppress",
            "Leave components out of this configuration, or turn a sketch or assembly feature off.",
        )
        .when(selected(&[K::Occurrence, K::Mate, K::Sketch, K::Feature]))
        .on(&[K::Occurrence, K::Mate, K::Sketch, K::Feature])
        .edit(edit(EditOp::Suppress, Section::Occurrences)),
        asm(
            "assembly.pack-and-share",
            "Pack and share",
            "pack",
            "Collect the assembly and every file it references.",
        )
        .inputs(vec![text("folder", "Folder", Some("rig-pack"))])
        .edit(export("pack")),
        asm(
            "assembly.export",
            "Export",
            "export",
            "Export as a STEP assembly, glTF or USD.",
        )
        .inputs(vec![
            choice(
                "format",
                "Format",
                &[("step", "STEP assembly"), ("gltf", "glTF"), ("usd", "USD")],
            ),
            flag("hidden", "Include hidden components", false),
        ])
        .edit(export("selected")),
        asm(
            "assembly.delete",
            "Delete",
            "delete",
            "Delete components, mates, mate connectors, publishes, configurations, display states, envelopes, contact regions, options, sketches, assembly features or sketch curves.",
        )
        .when(selected(&[
            K::Occurrence,
            K::Mate,
            K::MateConnector,
            K::Publish,
            K::Configuration,
            K::DisplayState,
            K::Envelope,
            K::Contact,
            K::ProductOption,
            K::MotionStudy,
            K::MatePosition,
            K::Sketch,
            K::Feature,
            K::SketchEntity,
            K::SketchPoint,
        ]))
        .on(&[
            K::Occurrence,
            K::Mate,
            K::MateConnector,
            K::Publish,
            K::Configuration,
            K::DisplayState,
            K::Envelope,
            K::Contact,
            K::ProductOption,
            K::MotionStudy,
            K::MatePosition,
            K::Sketch,
            K::Feature,
            K::SketchEntity,
            K::SketchPoint,
        ])
        .edit(edit(EditOp::Remove, Section::Occurrences)),
        asm(
            "assembly.edit-mate",
            "Edit mate",
            "edit",
            "Change the mate's type, distance, angle, limits or alignment.",
        )
        .inputs(
            vec![
                choice("type", "Type", EDITABLE_MATE_TYPES),
                offset("offset", "Distance", 0.0).shown("type", &["distance"]),
                angle("angle", "Angle", 0.0).shown("type", &["angle"]),
                flag("angle_limits", "Angle limits", false)
                    .shown("type", &["revolute", "cylindrical", "hinge"]),
                qty("min", "Min angle", "angle", Some(-30.0)).if_checked("angle_limits"),
                qty("max", "Max angle", "angle", Some(45.0)).if_checked("angle_limits"),
                flag("distance_limits", "Travel limits", false)
                    .shown("type", &["slider", "cylindrical"]),
                offset("near", "Min travel", 0.0).if_checked("distance_limits"),
                offset("far", "Max travel", 10.0).if_checked("distance_limits"),
                qty("ratio", "Ratio", "ratio", Some(1.0))
                    .shown("type", &["gear", "linear_coupler"]),
                length("pitch", "Pitch", 1.0).shown("type", &["screw", "rack_pinion"]),
                length("radius", "Radius", 5.0).shown("type", &["tangent"]),
                flag("flip", "Flip", false),
            ]
            .into_iter()
            .chain(advanced_mate_fields(CONNECTOR_PICKS, false))
            .collect(),
        )
        .when(selected_one(&[K::Mate]))
        .on(&[K::Mate])
        .edit(edit(EditOp::Edit, Section::Mates)),
        asm(
            "assembly.open",
            "Open part",
            "open",
            "Open the component's file in its own tab.",
        )
        .when(selected_one(&[K::Occurrence]))
        .on(&[K::Occurrence])
        .edit(view_action("open")),
    ]
}
