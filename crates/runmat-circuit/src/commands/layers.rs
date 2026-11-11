//! Board layer view commands: the active layer, which layers draw, single-layer mode and each
//! layer's color and opacity. The board's layer strip, its right-click menu and the keymap run
//! these, and the agent can call them. They change only the view, never the file.

use super::CIRCUIT;
use runmat_design_commands::*;

fn on_board(c: CommandDescriptor) -> CommandDescriptor {
    c.when(has(Facet::Board))
}

fn layer_field() -> Field {
    text("layer", "Layer", Some("F.Cu"))
}

pub(super) fn add(c: &mut Catalog) {
    c.add_commands(vec![
        on_board(cmd("circuit.active-layer", "Active layer", "layers", CIRCUIT))
            .desc("Make a layer the one new tracks and pours go on; it draws over the others.")
            .kw(&["board", "pcb", "current layer"])
            .inputs(vec![layer_field()])
            .edit(view_action("active_layer")),
        on_board(cmd("circuit.next-layer", "Next copper layer", "layers", CIRCUIT))
            .desc("Make the next copper layer down the stack active.")
            .kw(&["board", "pcb", "layer down"])
            .edit(view_action("next_layer")),
        on_board(cmd(
            "circuit.previous-layer",
            "Previous copper layer",
            "layers",
            CIRCUIT,
        ))
        .desc("Make the previous copper layer up the stack active.")
        .kw(&["board", "pcb", "layer up"])
        .edit(view_action("previous_layer")),
        on_board(cmd(
            "circuit.layer-visibility",
            "Layer visibility",
            "visibility",
            CIRCUIT,
        ))
        .desc("Show or hide board layers: one layer, only one, all, copper only, or one side.")
        .kw(&["board", "pcb", "show", "hide", "layers"])
        .inputs(vec![
            choice(
                "action",
                "Action",
                &[
                    ("toggle", "Show or hide the layer"),
                    ("show", "Show the layer"),
                    ("hide", "Hide the layer"),
                    ("only", "Show only this layer"),
                    ("all", "Show all layers"),
                    ("hide_others", "Hide all others"),
                    ("copper", "Show only copper"),
                    ("top", "Show only the top side"),
                    ("bottom", "Show only the bottom side"),
                ],
            ),
            layer_field().optional(),
        ])
        .edit(view_action("layer_visibility")),
        on_board(cmd(
            "circuit.single-layer",
            "Single-layer mode",
            "display",
            CIRCUIT,
        ))
        .desc("Dim every layer but the active one; run again to draw them all.")
        .kw(&["board", "pcb", "dim", "focus layer"])
        .edit(view_action("single_layer")),
        on_board(cmd("circuit.layer-color", "Layer color", "display", CIRCUIT))
            .desc("Draw a layer in another color on this board.")
            .kw(&["board", "pcb", "colour"])
            .inputs(vec![layer_field(), text("color", "Color", Some("#e0302a"))])
            .edit(view_action("layer_color")),
        on_board(cmd(
            "circuit.layer-opacity",
            "Layer opacity",
            "display",
            CIRCUIT,
        ))
        .desc("Draw a layer more or less see-through on this board.")
        .kw(&["board", "pcb", "transparency"])
        .inputs(vec![
            layer_field(),
            qty("opacity", "Opacity", "ratio", Some(1.0))
                .min(0.0)
                .max(1.0),
        ])
        .edit(view_action("layer_opacity")),
    ]);
}
