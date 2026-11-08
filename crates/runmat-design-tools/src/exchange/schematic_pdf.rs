//! `design.export` of a circuit as a PDF set: every schematic page in one multi-page PDF, each
//! page at its paper size with its frame and title block (title, sheet, path, page `n of m`,
//! size, and the variant when one is chosen).
//!
//! Each page draws as the canvas lays it out: symbol bodies and art, pin lines with their names
//! and numbers (small parts hide them), designators and values, sheet symbols with their
//! entries, wires, junction dots, net labels and power ports, no-connect crosses, notes, and
//! blankets as dashed boxes with their captions.
//! Text sets in the PDF writer's Courier, its baseline start shifted for centred and
//! right-aligned text by the font's 0.6-of-height character width.

use crate::formats::sheet::{Shape, Sheet};
use runmat_circuit::schematic::{NetLabel, SchematicSheet, FRAME_MM, TITLE_ROW_MM, TITLE_W_MM};

/// Courier's advance per character, as a fraction of the text height.
const ADVANCE: f64 = 0.6;

fn text(sheet: &mut Sheet, at: [f64; 2], height: f64, anchor: &str, words: &str) {
    if words.is_empty() {
        return;
    }
    let w = words.chars().count() as f64 * ADVANCE * height;
    let x = match anchor {
        "middle" => at[0] - w / 2.0,
        "end" => at[0] - w,
        _ => at[0],
    };
    sheet.add("TEXT", Shape::Text { at: [x, at[1]], height, rotation: 0.0, text: words.to_owned() });
}

fn rect(sheet: &mut Sheet, layer: &str, x0: f64, y0: f64, x1: f64, y1: f64) {
    sheet.add(layer, Shape::Polyline { points: vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]], closed: true });
}

/// The frame inside the paper edge and the title block in its bottom-right corner.
fn frame(sheet: &mut Sheet, page: &SchematicSheet) {
    let (w, h) = (page.width_mm, page.height_mm);
    rect(sheet, "OUTLINE", FRAME_MM, FRAME_MM, w - FRAME_MM, h - FRAME_MM);
    let rows = page.title_block.len().max(1) as f64;
    let (x0, x1) = (w - FRAME_MM - TITLE_W_MM, w - FRAME_MM);
    let top = FRAME_MM + rows * TITLE_ROW_MM;
    rect(sheet, "OUTLINE", x0, FRAME_MM, x1, top);
    for (k, (field, value)) in page.title_block.iter().enumerate() {
        let y1 = top - k as f64 * TITLE_ROW_MM;
        let y0 = y1 - TITLE_ROW_MM;
        if k > 0 {
            sheet.line("OUTLINE", [x0, y1], [x1, y1]);
        }
        text(sheet, [x0 + 1.5, y1 - 1.6], 1.2, "start", field);
        text(sheet, [x0 + 16.0, y0 + 1.3], 2.5, "start", value);
    }
}

/// A net label or power port: the name beside its stub point, a bar for a power port.
fn label(sheet: &mut Sheet, l: &NetLabel) {
    let [x0, y0, x1, y1] = l.extent();
    let at = [l.x_mm, l.y_mm];
    if l.kind == "power" {
        let bar = if l.side == "down" { -2.54 } else { 2.54 };
        sheet.line("LABEL", at, [at[0], at[1] + bar]);
        sheet.line("LABEL", [at[0] - 1.27, at[1] + bar], [at[0] + 1.27, at[1] + bar]);
        let ty = if bar > 0.0 { at[1] + bar + 0.8 } else { at[1] + bar - 2.0 };
        text(sheet, [at[0], ty], 1.27, "middle", &l.text);
        return;
    }
    // A global or hierarchical label sits in an outline; a local one is its text on the stub.
    if l.kind != "local" {
        rect(sheet, "LABEL", x0, y0, x1, y1);
    }
    let h = 1.27;
    let ty = (y0 + y1) / 2.0 - h * 0.35;
    match l.side.as_str() {
        "left" => text(sheet, [x1 - 0.5, ty], h, "end", &l.text),
        "right" => text(sheet, [x0 + 0.5, ty], h, "start", &l.text),
        _ => text(sheet, [(x0 + x1) / 2.0, ty], h, "middle", &l.text),
    }
}

/// One schematic page as a drawing sheet.
pub fn page_sheet(page: &SchematicSheet) -> Sheet {
    let mut sheet = Sheet { paper: [page.width_mm, page.height_mm], ..Default::default() };
    frame(&mut sheet, page);
    for s in &page.symbols {
        for line in s.body.iter().chain(&s.strokes) {
            if line.len() > 1 {
                sheet.add("SYMBOL", Shape::Polyline { points: line.clone(), closed: false });
            }
        }
        let name = match &s.unit {
            Some(u) => format!("{}{u}", s.designator),
            None => s.designator.clone(),
        };
        text(&mut sheet, s.designator_at, s.designator_text_mm, &s.text_anchor, &name);
        if let Some(v) = &s.value {
            text(&mut sheet, s.value_at, s.value_text_mm, &s.text_anchor, v);
        }
        for p in s.pins.iter().filter(|p| !p.hidden) {
            text(&mut sheet, p.name_at, s.pin_text_mm, &p.name_anchor, &p.name);
            if p.number_text != p.name {
                text(&mut sheet, p.number_at, s.pin_text_mm, "middle", &p.number_text);
            }
        }
    }
    for s in &page.sheet_symbols {
        rect(&mut sheet, "SYMBOL", s.x_mm, s.y_mm, s.x_mm + s.w_mm, s.y_mm + s.h_mm);
        text(&mut sheet, [s.x_mm, s.y_mm + s.h_mm + 0.8], 1.6, "start", &s.path);
        text(&mut sheet, [s.x_mm, s.y_mm - 2.0], 1.3, "start", &s.sheet);
        for p in &s.pins {
            let left = p.side == "left";
            let d = if left { 1.0 } else { -1.0 };
            sheet.add("SYMBOL", Shape::Polyline { points: vec![[p.x_mm, p.y_mm + 0.6], [p.x_mm + d * 1.0, p.y_mm + 0.6], [p.x_mm + d * 1.6, p.y_mm], [p.x_mm + d * 1.0, p.y_mm - 0.6], [p.x_mm, p.y_mm - 0.6]], closed: true });
            text(&mut sheet, [p.x_mm + d * 2.2, p.y_mm - 0.38], 1.27, if left { "start" } else { "end" }, &p.name);
        }
    }
    for w in &page.wires {
        sheet.add("WIRE", Shape::Polyline { points: w.points.clone(), closed: false });
    }
    for j in &page.junctions {
        sheet.add("WIRE", Shape::Hatch { loops: vec![(0..12).map(|k| { let a = k as f64 * std::f64::consts::TAU / 12.0; [j[0] + 0.5 * a.cos(), j[1] + 0.5 * a.sin()] }).collect()], pattern: "SOLID".into(), angle: 0.0, scale: 1.0 });
    }
    for n in &page.no_connects {
        sheet.line("WIRE", [n[0] - 0.8, n[1] - 0.8], [n[0] + 0.8, n[1] + 0.8]);
        sheet.line("WIRE", [n[0] - 0.8, n[1] + 0.8], [n[0] + 0.8, n[1] - 0.8]);
    }
    for l in &page.labels {
        label(&mut sheet, l);
    }
    for t in &page.texts {
        text(&mut sheet, [t.x_mm, t.y_mm], t.size_mm, "start", &t.text);
    }
    // Blankets: a dashed box round the nets each covers, captioned above its top-left corner.
    if !page.blankets.is_empty() {
        sheet.linetypes = crate::formats::sheet::standard_linetypes();
        sheet.layers.push(crate::formats::sheet::Layer { name: "BLANKET".into(), color: 5, linetype: "DASHED".into() });
    }
    for b in &page.blankets {
        let [x0, y0, x1, y1] = b.rect;
        rect(&mut sheet, "BLANKET", x0, y0, x1, y1);
        text(&mut sheet, b.caption_at, b.caption_mm, "start", &b.caption);
    }
    sheet
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centred_text_starts_half_its_width_left() {
        let mut sheet = Sheet::default();
        text(&mut sheet, [10.0, 5.0], 2.0, "middle", "ABCD");
        let Shape::Text { at, .. } = &sheet.entities[0].shape else { panic!() };
        assert!((at[0] - (10.0 - 2.4)).abs() < 1e-9);
    }

    #[test]
    fn a_blanket_draws_dashed_with_its_caption() {
        let page = SchematicSheet {
            width_mm: 297.0,
            height_mm: 210.0,
            blankets: vec![runmat_circuit::schematic::SheetBlanket {
                name: "rails".into(),
                caption: "blanket rails · class power".into(),
                caption_at: [20.0, 60.6],
                caption_mm: 1.27,
                rect: [20.0, 30.0, 80.0, 60.0],
                nets: vec!["GND".into()],
            }],
            ..Default::default()
        };
        let sheet = page_sheet(&page);
        let layer = sheet.layers.iter().find(|l| l.name == "BLANKET").expect("a blanket layer");
        assert_eq!(layer.linetype, "DASHED");
        assert!(sheet.linetypes.iter().any(|t| t.name == "DASHED"));
        assert!(sheet.entities.iter().any(|e| e.layer == "BLANKET" && matches!(&e.shape, Shape::Polyline { points, closed: true } if points[0] == [20.0, 30.0] && points[2] == [80.0, 60.0])));
        assert!(sheet.entities.iter().any(|e| matches!(&e.shape, Shape::Text { text, at, .. } if text == "blanket rails · class power" && at[0] == 20.0)));
    }
}
