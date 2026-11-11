//! The agent panel's suggestions: per editor, file state and selection, the composer's
//! placeholder first and then the empty panel's suggestions (at most [`MAX_PANEL_SUGGESTIONS`]).
//! The Desktop reads them from `design-guides.json` with the guides.

use super::PickEntity;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A suggestion for the agent panel. For the open editor, the file's state and the selection,
/// the first suggestion that matches is the composer's placeholder; the next ones that need no
/// selection, at most [`MAX_PANEL_SUGGESTIONS`], are the empty panel's suggestions. Choosing one
/// sends its text as the prompt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Suggestion {
    /// The editor it applies in: part, assembly, drawing, circuit, layout, fea, geometry, m, or
    /// none (nothing open).
    pub editor: String,
    /// The selection it needs, when it needs one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<PickEntity>,
    /// The file state it needs, when it needs one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<FileState>,
    /// The prompt. With `names`, `{1}`, `{2}`, … stand for the open file's first, second, …
    /// names of that kind, so the text names what the file holds.
    pub text: String,
    /// What the text's `{1}`, `{2}`, … name. A file with fewer such names isn't offered it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub names: Option<SuggestionNames>,
    /// The catalog command that does it, so the suggestion never names a missing feature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
}

/// The open file's content, as a suggestion needs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FileState {
    /// Nothing but whitespace: a new file.
    Empty,
    /// Some content.
    Written,
    /// A part with sheet-metal features: its own checks (flat pattern, bend reliefs) replace
    /// a machined part's.
    SheetMetal,
}

impl FileState {
    /// Every file state, for checks that walk them all.
    pub const ALL: [FileState; 3] = [FileState::Empty, FileState::Written, FileState::SheetMetal];
}

/// The names a suggestion's text takes from the open file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SuggestionNames {
    /// An assembly's occurrences, by their names (`use part … as <name>`), in file order.
    Occurrences,
}

/// The empty agent panel shows at most this many suggestions besides the placeholder.
pub const MAX_PANEL_SUGGESTIONS: usize = 3;

fn suggestion(
    editor: &str,
    selection: Option<PickEntity>,
    text: &str,
    cmd: Option<&str>,
) -> Suggestion {
    Suggestion {
        editor: editor.into(),
        selection,
        state: None,
        text: text.into(),
        names: None,
        command: cmd.map(str::to_owned),
    }
}

/// A suggestion whose text names what the open file holds.
fn naming(names: SuggestionNames, s: Suggestion) -> Suggestion {
    Suggestion {
        names: Some(names),
        ..s
    }
}

/// A suggestion for one file state.
fn when(state: FileState, s: Suggestion) -> Suggestion {
    Suggestion {
        state: Some(state),
        ..s
    }
}

/// Every suggestion, per editor and file state: the placeholder first, then the panel's.
pub(super) fn catalog() -> Vec<Suggestion> {
    use FileState::{Empty, SheetMetal, Written};
    use PickEntity::{Edge, Face, Net};
    use SuggestionNames::Occurrences;
    vec![
        suggestion("none", None, "Make a bracket that holds 2 kN", None),
        suggestion("none", None, "Capture a schematic for a sensor node", None),
        suggestion(
            "none",
            None,
            "Write a script that plots a damped sine wave",
            None,
        ),
        suggestion("none", None, "Show me an example in this project", None),
        when(
            Empty,
            suggestion("m", None, "Describe what this script should do", None),
        ),
        when(
            Empty,
            suggestion("m", None, "Plot a damped sine wave", None),
        ),
        when(
            Empty,
            suggestion("m", None, "Load a CSV file and plot each column", None),
        ),
        when(
            Empty,
            suggestion("m", None, "Simulate a spring and mass for 10 s", None),
        ),
        when(
            Written,
            suggestion("m", None, "Plot this result against time", None),
        ),
        when(
            Written,
            suggestion("m", None, "Run this script and explain the output", None),
        ),
        when(
            Written,
            suggestion("m", None, "Explain what this script does", None),
        ),
        when(
            Written,
            suggestion("m", None, "Check this script runs in RunMat", None),
        ),
        suggestion(
            "part",
            Some(Edge),
            "Fillet these edges 2 mm",
            Some("part.fillet"),
        ),
        suggestion(
            "part",
            Some(Face),
            "Draft these faces 2°",
            Some("part.draft"),
        ),
        when(
            Empty,
            suggestion(
                "part",
                None,
                "Make a 60 × 40 × 8 mm plate with four M5 holes",
                None,
            ),
        ),
        when(
            Empty,
            suggestion("part", None, "Model a bracket that holds 2 kN", None),
        ),
        when(
            Empty,
            suggestion("part", None, "Make a 20 mm shaft with a keyway", None),
        ),
        when(
            Written,
            suggestion(
                "part",
                None,
                "Add a 2 mm fillet to the top edges",
                Some("part.fillet"),
            ),
        ),
        when(
            Written,
            suggestion(
                "part",
                None,
                "Check this part under a 2 kN load",
                Some("study.new"),
            ),
        ),
        when(
            Written,
            suggestion("part", None, "Explain how this part is built", None),
        ),
        when(
            Written,
            suggestion("part", None, "Check this part can be machined", None),
        ),
        when(
            SheetMetal,
            suggestion(
                "part",
                None,
                "Add a 15 mm flange to the open edges",
                Some("sheet-metal.edge-flange"),
            ),
        ),
        when(
            SheetMetal,
            suggestion(
                "part",
                None,
                "Check this part under a 2 kN load",
                Some("study.new"),
            ),
        ),
        when(
            SheetMetal,
            suggestion("part", None, "Explain how this part is built", None),
        ),
        when(
            SheetMetal,
            suggestion(
                "part",
                None,
                "Check the flat pattern and bend reliefs",
                None,
            ),
        ),
        when(
            Empty,
            suggestion(
                "assembly",
                None,
                "Insert the parts this assembly brings together",
                Some("assembly.insert"),
            ),
        ),
        when(
            Empty,
            suggestion(
                "assembly",
                None,
                "Make a hinge from two plates and a pin",
                None,
            ),
        ),
        when(
            Empty,
            suggestion(
                "assembly",
                None,
                "Mount a motor on a bracket with four bolts",
                None,
            ),
        ),
        when(
            Written,
            naming(
                Occurrences,
                suggestion("assembly", None, "Mate {1} to {2}", Some("assembly.mate")),
            ),
        ),
        when(
            Written,
            suggestion(
                "assembly",
                None,
                "Explain how this assembly fits together",
                None,
            ),
        ),
        when(
            Written,
            suggestion("assembly", None, "Check the parts for interference", None),
        ),
        suggestion(
            "drawing",
            None,
            "Add the hole callouts",
            Some("drawing.hole-callout"),
        ),
        suggestion("drawing", None, "Add the critical dimensions", None),
        suggestion(
            "drawing",
            None,
            "Check this drawing is ready to release",
            None,
        ),
        when(
            Empty,
            suggestion(
                "circuit",
                None,
                "Describe the circuit: its blocks, power and interfaces",
                None,
            ),
        ),
        when(
            Empty,
            suggestion(
                "circuit",
                None,
                "Capture a schematic for a GPS sensor node",
                None,
            ),
        ),
        when(
            Empty,
            suggestion(
                "circuit",
                None,
                "Add a USB-C power input with ESD protection",
                None,
            ),
        ),
        when(
            Written,
            suggestion(
                "circuit",
                None,
                "Run ERC and fix what it finds",
                Some("schematic.erc"),
            ),
        ),
        when(
            Written,
            suggestion("circuit", None, "Explain how this circuit works", None),
        ),
        when(
            Written,
            suggestion(
                "circuit",
                None,
                "Update the board from this schematic",
                Some("schematic.update-board"),
            ),
        ),
        suggestion(
            "layout",
            None,
            "Update the board from the schematic",
            Some("board.update-schematic"),
        ),
        suggestion(
            "layout",
            Some(Net),
            "Run DRC on this net",
            Some("board.drc"),
        ),
        suggestion(
            "layout",
            None,
            "Run DRC and fix what it finds",
            Some("board.drc"),
        ),
        suggestion("layout", None, "Explain this board's stack-up", None),
        suggestion(
            "geometry",
            None,
            "Set up a simulation of this part",
            Some("study.new"),
        ),
        suggestion(
            "geometry",
            None,
            "Explain this model's bodies and features",
            None,
        ),
        suggestion(
            "geometry",
            None,
            "Recommend an analysis for this part",
            None,
        ),
        when(
            Empty,
            suggestion(
                "fea",
                None,
                "Describe the study: loads, supports and material",
                None,
            ),
        ),
        when(
            Empty,
            suggestion("fea", None, "Set up a stress study under a 2 kN load", None),
        ),
        when(
            Empty,
            suggestion("fea", None, "Set up a modal study of this part", None),
        ),
        when(
            Written,
            suggestion("fea", None, "Explain the margin on this result", None),
        ),
        when(
            Written,
            suggestion(
                "fea",
                None,
                "Run this study and summarize the results",
                None,
            ),
        ),
        when(
            Written,
            suggestion("fea", None, "Check the supports and loads", None),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The empty panel: each editor and file state has a placeholder and at most three
    /// suggestions, and an empty script or circuit isn't offered edits to content it lacks.
    #[test]
    fn every_editor_has_a_placeholder_and_at_most_three_suggestions() {
        let c = super::super::catalog();
        let shown = |editor: &str, state: FileState| -> Vec<String> {
            c.suggestions
                .iter()
                .filter(|s| s.editor == editor && s.selection.is_none())
                .filter(|s| s.state.is_none_or(|st| st == state))
                .map(|s| s.text.clone())
                .collect()
        };
        for editor in [
            "none", "m", "part", "assembly", "drawing", "circuit", "layout", "geometry", "fea",
        ] {
            for state in [FileState::Empty, FileState::Written] {
                let list = shown(editor, state);
                assert!(!list.is_empty(), "{editor} {state:?} has no placeholder");
                assert!(
                    list.len() <= MAX_PANEL_SUGGESTIONS + 1,
                    "{editor} {state:?}: {list:?}"
                );
            }
        }
        assert_eq!(
            shown("m", FileState::Empty)[0],
            "Describe what this script should do"
        );
        assert_eq!(
            shown("m", FileState::Written)[0],
            "Plot this result against time"
        );
        assert_eq!(
            shown("part", FileState::Written)[0],
            "Add a 2 mm fillet to the top edges"
        );
        assert_eq!(
            shown("circuit", FileState::Written)[0],
            "Run ERC and fix what it finds"
        );
        assert_eq!(shown("assembly", FileState::Written)[0], "Mate {1} to {2}");
        // A sheet-metal part is offered its own checks, never machining.
        let sheet = shown("part", FileState::SheetMetal);
        assert!(
            sheet.len() > 1 && sheet.len() <= MAX_PANEL_SUGGESTIONS + 1,
            "{sheet:?}"
        );
        assert!(
            sheet.contains(&"Check the flat pattern and bend reliefs".to_owned()),
            "{sheet:?}"
        );
        assert!(!sheet.iter().any(|t| t.contains("machined")), "{sheet:?}");
        assert_eq!(
            shown("assembly", FileState::Empty)[0],
            "Insert the parts this assembly brings together"
        );
        assert!(c
            .lint(&[])
            .iter()
            .all(|p| !p.contains("more than a placeholder")));
    }
}
