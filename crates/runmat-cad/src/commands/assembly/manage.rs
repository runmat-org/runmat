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
            text("option", "Option", Some("engine")),
            text("values", "Values", Some("a, b")),
            text("effectivity", "Effectivity", Some("MSN 1-99")).optional(),
        ])
        .edit(edit(EditOp::Add, Section::Options).item("option")),
        asm(
            "assembly.configurations",
            "Configurations and display states",
            "configuration",
            "Add configurations and display states.",
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
            text("name", "Name", Some("open")),
        ])
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
            "assembly.update-pins",
            "References and pins",
            "pin",
            "Review pinned revisions of components and update them.",
        )
        .kw(&["revision", "update", "version"])
        .inputs(vec![
            picks("components", "Components", &[K::Occurrence]).optional(),
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
            "Leave components out of this configuration.",
        )
        .when(selected(&[K::Occurrence, K::Mate]))
        .on(&[K::Occurrence, K::Mate])
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
            "Delete components or mates.",
        )
        .when(selected(&[K::Occurrence, K::Mate]))
        .on(&[K::Occurrence, K::Mate])
        .edit(edit(EditOp::Remove, Section::Occurrences)),
        asm(
            "assembly.edit-mate",
            "Edit mate",
            "edit",
            "Change the mate's type, distance, angle, limits or alignment.",
        )
        .inputs(vec![
            choice("type", "Type", EDITABLE_MATE_TYPES),
            offset("offset", "Distance", 0.0).shown("type", &["distance"]),
            angle("angle", "Angle", 0.0).shown("type", &["angle"]),
            flag("angle_limits", "Angle limits", false).shown("type", &["revolute", "cylindrical"]),
            qty("min", "Min angle", "angle", Some(-30.0)).if_checked("angle_limits"),
            qty("max", "Max angle", "angle", Some(45.0)).if_checked("angle_limits"),
            flag("distance_limits", "Travel limits", false).shown("type", &["slider", "cylindrical"]),
            offset("near", "Min travel", 0.0).if_checked("distance_limits"),
            offset("far", "Max travel", 10.0).if_checked("distance_limits"),
            qty("ratio", "Ratio", "ratio", Some(1.0)).shown("type", &["gear"]),
            length("pitch", "Pitch", 1.0).shown("type", &["screw", "rack_pinion"]),
            flag("flip", "Flip", false),
        ])
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
