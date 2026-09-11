use serde::Serialize;

use super::{BuiltinExampleFixtureId, BuiltinFilesystemEntry};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinDesktopHostFixture {
    pub id: BuiltinExampleFixtureId,
    pub entries: &'static [BuiltinFilesystemEntry],
    pub interactions: &'static [BuiltinDesktopHostInteraction],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinDesktopHostInteraction {
    OpenFileDialog(BuiltinOpenFileDialogInteraction),
    SaveFileDialog(BuiltinSaveFileDialogInteraction),
    DirectoryDialog(BuiltinDirectoryDialogInteraction),
    LineInput(BuiltinLineInputInteraction),
    KeyInput(BuiltinKeyInputInteraction),
    FigurePresentation(BuiltinFigurePresentationInteraction),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinDialogFilterExpectation {
    pub patterns: &'static [&'static str],
    pub description: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinOpenFileDialogInteraction {
    pub request: BuiltinOpenFileDialogExpectation,
    pub outcome: BuiltinOpenFileDialogOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinOpenFileDialogExpectation {
    pub title: Option<&'static str>,
    pub default_path: Option<&'static str>,
    pub filters: &'static [BuiltinDialogFilterExpectation],
    pub multiselect: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinOpenFileDialogOutcome {
    Selection {
        paths: &'static [&'static str],
        /// Zero-based index into the declared filter list.
        filter_index: Option<usize>,
    },
    Cancel,
    Error(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinSaveFileDialogInteraction {
    pub request: BuiltinSaveFileDialogExpectation,
    pub outcome: BuiltinSaveFileDialogOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinSaveFileDialogExpectation {
    pub title: Option<&'static str>,
    pub default_path: Option<&'static str>,
    pub filters: &'static [BuiltinDialogFilterExpectation],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinSaveFileDialogOutcome {
    Selection {
        path: &'static str,
        /// Zero-based index into the declared filter list.
        filter_index: Option<usize>,
    },
    Cancel,
    Error(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinDirectoryDialogInteraction {
    pub request: BuiltinDirectoryDialogExpectation,
    pub outcome: BuiltinDirectoryDialogOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinDirectoryDialogExpectation {
    pub title: Option<&'static str>,
    pub default_path: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinDirectoryDialogOutcome {
    Selection { path: &'static str },
    Cancel,
    Error(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinLineInputInteraction {
    pub prompt: &'static str,
    pub echo: bool,
    pub outcome: BuiltinLineInputOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinLineInputOutcome {
    Line(&'static str),
    Error(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinKeyInputInteraction {
    pub prompt: &'static str,
    pub outcome: BuiltinKeyInputOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinKeyInputOutcome {
    KeyPress,
    Error(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinFigurePresentationInteraction {
    /// Stable one-based ordinal assigned by first observation in this case.
    pub figure_ordinal: usize,
    pub event: BuiltinFigurePresentationEvent,
    pub snapshot: Option<BuiltinFigureSnapshotExpectation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinFigurePresentationEvent {
    Created,
    Updated,
    Cleared,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinFigureSnapshotExpectation {
    pub title: Option<&'static str>,
    pub axes_rows: usize,
    pub axes_cols: usize,
}
