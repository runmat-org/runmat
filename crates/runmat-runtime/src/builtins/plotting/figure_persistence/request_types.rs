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

    pub(super) fn label(self) -> &'static str {
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
    pub(super) figures: Vec<FigureHandle>,
    pub(super) output_path: PathBuf,
}

#[derive(Debug)]
pub(super) struct OpenFigRequest {
    pub(super) path: PathBuf,
    pub(super) visibility: Option<bool>,
}
