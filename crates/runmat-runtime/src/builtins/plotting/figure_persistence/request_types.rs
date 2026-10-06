use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SaveFormat {
    Fig,
    Png,
}

impl SaveFormat {
    pub(super) fn standard_extension(self) -> &'static str {
        match self {
            SaveFormat::Fig => "fig",
            SaveFormat::Png => "png",
        }
    }
}

#[derive(Debug)]
pub(super) struct SaveAsRequest {
    pub(super) figure: FigureHandle,
    pub(super) output_path: PathBuf,
    pub(super) format: SaveFormat,
}

#[derive(Debug)]
pub(super) struct SaveFigRequest {
    pub(super) selection: FigureSaveSelection,
    pub(super) output_path: PathBuf,
    pub(super) profile: FigureArchiveProfile,
}

#[derive(Debug, Clone)]
pub(super) enum FigureSaveSelection {
    Figures(Vec<FigureHandle>),
    Hierarchy(Vec<PlotHandle>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FigureArchiveProfile {
    Default,
    V6,
    V7,
    V73,
    Compact,
}

#[derive(Debug)]
pub(super) struct OpenFigRequest {
    pub(super) path: PathBuf,
    pub(super) copies: OpenFigCopies,
    pub(super) visibility: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OpenFigCopies {
    New,
    Reuse,
}
