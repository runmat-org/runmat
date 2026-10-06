use std::fmt;

use super::super::Figure;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FigureDecoration {
    Title,
    Subtitle,
    XLabel,
    YLabel,
    ZLabel,
    Legend,
    ColorBar,
    XRuler,
    YRuler,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FigureAxesContainerSelection {
    pub decorations: Vec<FigureDecoration>,
    pub annotation_indices: Vec<usize>,
    pub plot_indices: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FigureAxesSelectionKind {
    FullAxes,
    Container(FigureAxesContainerSelection),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FigureAxesPersistenceSelection {
    pub source_axes_index: usize,
    pub kind: FigureAxesSelectionKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FigurePersistenceSelection {
    FullFigure,
    Hierarchy {
        include_super_title: bool,
        axes: Vec<FigureAxesPersistenceSelection>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FigureAxesRemap {
    pub source: usize,
    pub destination: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FigurePlotRemap {
    pub source: usize,
    pub destination: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FigureAnnotationRemap {
    pub source_axes: usize,
    pub source_annotation: usize,
    pub destination_axes: usize,
    pub destination_annotation: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FigureDecorationRemap {
    pub source_axes: usize,
    pub destination_axes: usize,
    pub decoration: FigureDecoration,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FigureProjectionRemap {
    pub axes: Vec<FigureAxesRemap>,
    pub plots: Vec<FigurePlotRemap>,
    pub annotations: Vec<FigureAnnotationRemap>,
    pub decorations: Vec<FigureDecorationRemap>,
}

#[derive(Clone, Debug)]
pub struct FigurePersistenceProjection {
    pub figure: Figure,
    pub remap: FigureProjectionRemap,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FigurePersistenceProjectionError {
    EmptySelection,
    AxesOutOfBounds {
        axes: usize,
    },
    PlotOutOfBounds {
        plot: usize,
    },
    PlotOwnedByDifferentAxes {
        plot: usize,
        selected_axes: usize,
        actual_axes: usize,
    },
    AnnotationOutOfBounds {
        axes: usize,
        annotation: usize,
    },
    MissingOverlayParent {
        axes: usize,
        parent: usize,
    },
    MissingGridPosition {
        axes: usize,
        rows: usize,
        columns: usize,
    },
}

impl fmt::Display for FigurePersistenceProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySelection => formatter.write_str("figure persistence selection is empty"),
            Self::AxesOutOfBounds { axes } => {
                write!(formatter, "selected axes index {axes} does not exist")
            }
            Self::PlotOutOfBounds { plot } => {
                write!(formatter, "selected plot index {plot} does not exist")
            }
            Self::PlotOwnedByDifferentAxes {
                plot,
                selected_axes,
                actual_axes,
            } => write!(
                formatter,
                "selected plot {plot} belongs to axes {actual_axes}, not axes {selected_axes}"
            ),
            Self::AnnotationOutOfBounds { axes, annotation } => write!(
                formatter,
                "selected annotation {annotation} does not exist on axes {axes}"
            ),
            Self::MissingOverlayParent { axes, parent } => write!(
                formatter,
                "selected overlay axes {axes} requires selected parent axes {parent}"
            ),
            Self::MissingGridPosition {
                axes,
                rows,
                columns,
            } => write!(
                formatter,
                "selected axes {axes} has no position in its {rows}-by-{columns} grid"
            ),
        }
    }
}

impl std::error::Error for FigurePersistenceProjectionError {}
