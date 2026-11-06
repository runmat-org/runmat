//! Where a sheet's title block draws: one layout, in sheet millimetres, that the Desktop's sheet
//! and every export (PDF, SVG, DXF, DWG) draw as given.
//!
//! The block follows ISO 7200: 180 mm wide at the frame's lower right corner, 10 mm rows, each
//! field's label 2.5 mm high above its value 3.5 mm high. More than four fields split into two
//! columns. A logo field (`source = image`) draws the image's outline in its cell, inset
//! [`LOGO_INSET`] on every side and under its label; a logo field with an empty label fills
//! its cell. The projection symbol takes its own cell left of the block
//! (ISO 5456-2).

use super::logo::Outline;
use super::{DrawingDocument, FieldSource, SheetSize};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The block's width (ISO 7200).
pub const BLOCK_WIDTH: f64 = 180.0;
/// One field's row height.
pub const ROW_HEIGHT: f64 = 10.0;
/// The frame's distance from the sheet edge.
pub const FRAME_MARGIN: f64 = 10.0;
/// The projection symbol cell's width.
pub const PROJECTION_CELL_WIDTH: f64 = 26.0;
/// Label and value text heights (ISO 3098).
pub const LABEL_HEIGHT: f64 = 2.5;
pub const VALUE_HEIGHT: f64 = 3.5;
/// A logo's inset in its cell.
pub const LOGO_INSET: f64 = 1.5;

/// A sheet's size in millimetres, landscape, by its ISO or ANSI name (A3 when unknown).
pub fn sheet_size(name: &str) -> [f64; 2] {
    match name.to_ascii_uppercase().as_str() {
        "A0" => [1189.0, 841.0],
        "A1" => [841.0, 594.0],
        "A2" => [594.0, 420.0],
        "A4" => [297.0, 210.0],
        "B" => [431.8, 279.4],
        "C" => [558.8, 431.8],
        "D" => [863.6, 558.8],
        _ => [420.0, 297.0],
    }
}

/// A text line placed on the sheet: its baseline's left end and its height.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlacedText {
    pub at: [f64; 2],
    pub height: f64,
    pub text: String,
}

/// One title-block cell: its rectangle (lower left corner, size), its label, and its value as
/// text or as a logo's closed outline loops.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TitleCell {
    pub at: [f64; 2],
    pub size: [f64; 2],
    pub label: PlacedText,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<PlacedText>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub logo: Vec<Vec<[f64; 2]>>,
}

/// A sheet's title block as drawn.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TitleLayout {
    pub cells: Vec<TitleCell>,
    /// The projection symbol's cell (lower left corner, size); none without fields.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_cell: Option<([f64; 2], [f64; 2])>,
}

/// One field to lay out: its label and its value text, or the outline of its logo.
pub enum TitleValue<'a> {
    Text(&'a str),
    Logo(&'a Outline),
}

/// The title block of a `sheet` mm wide sheet with `fields` in order.
pub fn layout(sheet: [f64; 2], fields: &[(&str, TitleValue<'_>)]) -> TitleLayout {
    if fields.is_empty() {
        return TitleLayout::default();
    }
    let x0 = sheet[0] - FRAME_MARGIN - BLOCK_WIDTH;
    let columns = if fields.len() > 4 { 2 } else { 1 };
    let per_column = fields.len().div_ceil(columns);
    let width = BLOCK_WIDTH / columns as f64;
    let cells = fields
        .iter()
        .enumerate()
        .map(|(i, (label, value))| {
            let x = x0 + (i / per_column) as f64 * width;
            let y = FRAME_MARGIN + (per_column - 1 - i % per_column) as f64 * ROW_HEIGHT;
            let label = PlacedText {
                at: [x + 1.5, y + ROW_HEIGHT - 1.5 - LABEL_HEIGHT],
                height: LABEL_HEIGHT,
                text: (*label).to_owned(),
            };
            let (value, logo) = match value {
                TitleValue::Text(text) => (
                    Some(PlacedText { at: [x + 1.5, y + 1.6], height: VALUE_HEIGHT, text: (*text).to_owned() }),
                    Vec::new(),
                ),
                // Under the label, the logo fills the rest of the cell; without a label, the
                // whole cell.
                TitleValue::Logo(outline) => {
                    let top = if label.text.is_empty() { y + ROW_HEIGHT - LOGO_INSET } else { label.at[1] - 0.5 };
                    let bottom = y + LOGO_INSET;
                    let center = [x + width / 2.0, (top + bottom) / 2.0];
                    (None, outline.fitted(center, width - 2.0 * LOGO_INSET, top - bottom))
                }
            };
            TitleCell { at: [x, y], size: [width, ROW_HEIGHT], label, value, logo }
        })
        .collect();
    let height = per_column as f64 * ROW_HEIGHT;
    TitleLayout {
        cells,
        projection_cell: Some(([x0 - PROJECTION_CELL_WIDTH, FRAME_MARGIN], [PROJECTION_CELL_WIDTH, height])),
    }
}

/// Every sheet's title block: `rows` holds each sheet's field text in order (the drawing
/// report's `title_blocks`), and `logo` gives the outline a logo field draws (`None` leaves its
/// cell empty).
pub fn sheet_layouts<'a>(
    document: &DrawingDocument,
    rows: &BTreeMap<String, Vec<(String, String)>>,
    logo: impl Fn(&str, &str) -> Option<&'a Outline>,
) -> BTreeMap<String, TitleLayout> {
    let empty = Outline::default();
    let mut out = BTreeMap::new();
    for sheet in &document.sheets {
        let Some(format) = document.formats.iter().find(|f| f.name == sheet.format) else {
            continue;
        };
        let size = match &format.size {
            SheetSize::Iso { name } | SheetSize::Ansi { name } => sheet_size(name),
            SheetSize::Custom { .. } => sheet_size(""),
        };
        let texts = rows.get(&sheet.name);
        let named = format.title_fields(&super::drawing_title(document.path.as_str()));
        let fields: Vec<(&str, TitleValue<'_>)> = named
            .iter()
            .enumerate()
            .map(|(i, field)| {
                let value = match &field.source {
                    FieldSource::Image { file } => TitleValue::Logo(logo(&field.label, file).unwrap_or(&empty)),
                    _ => TitleValue::Text(texts.and_then(|t| t.get(i)).map_or("", |r| r.1.as_str())),
                };
                (field.label.as_str(), value)
            })
            .collect();
        out.insert(sheet.name.clone(), layout(size, &fields));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_stack_from_the_frame_corner_and_a_logo_fits_its_cell() {
        let logo = Outline { size: [100.0, 20.0], loops: vec![vec![[0.0, 0.0], [100.0, 0.0], [100.0, 20.0], [0.0, 20.0]]] };
        let a3 = [420.0, 297.0];
        let block = layout(a3, &[("Title", TitleValue::Text("Bracket")), ("Company", TitleValue::Logo(&logo))]);
        assert_eq!(block.cells.len(), 2);
        // The first field sits on top, the last on the frame's bottom edge.
        assert_eq!(block.cells[0].at, [230.0, 20.0]);
        assert_eq!(block.cells[1].at, [230.0, 10.0]);
        assert_eq!(block.cells[0].value.as_ref().unwrap().text, "Bracket");
        let cell = &block.cells[1];
        assert!(cell.value.is_none());
        let points: Vec<[f64; 2]> = cell.logo.iter().flatten().copied().collect();
        for p in &points {
            assert!(p[0] >= cell.at[0] + LOGO_INSET - 1e-9 && p[0] <= cell.at[0] + cell.size[0] - LOGO_INSET + 1e-9, "{p:?}");
            assert!(p[1] >= cell.at[1] + LOGO_INSET - 1e-9 && p[1] < cell.label.at[1], "{p:?}");
        }
        assert_eq!(block.projection_cell, Some(([204.0, 10.0], [26.0, 20.0])));
        // Five fields split into two columns of three.
        let five: Vec<(&str, TitleValue<'_>)> = (0..5).map(|_| ("F", TitleValue::Text("v"))).collect();
        let two = layout(a3, &five);
        assert_eq!(two.cells[3].at, [320.0, 30.0]);
        assert_eq!(two.cells[3].size, [90.0, 10.0]);
    }
}
