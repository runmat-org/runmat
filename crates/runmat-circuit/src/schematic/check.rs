//! The schematic's layout check: what a reader can't read on the drawn pages.
//!
//! | Code | Meaning |
//! |---|---|
//! | `off_sheet` | a symbol, note, label, sheet symbol or wire reaches past the frame line or over the title block |
//! | `text_overlap` | two texts cover each other: designators, values, notes, net labels, sheet names, blanket captions |
//!
//! Both are warnings: the netlist is right, the drawing is not. They name the items involved
//! and where on the page (mm, from the lower left).

use super::{
    designator_text, mm, mm_to_nm, text_box, title_height_mm, SchematicIndex, SchematicSheet,
    FRAME_MM, TITLE_W_MM,
};

/// One layout finding on one page.
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutFinding {
    /// `off_sheet` or `text_overlap`.
    pub code: &'static str,
    pub message: String,
    pub items: Vec<String>,
    /// The page (sheet instance path) and where on it, mm.
    pub sheet: String,
    pub at_mm: [f64; 2],
}

type Box4 = [f64; 4];

/// A place rounded to 0.01 mm.
fn r2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn text_mm(text: &str, size: f64, at: [f64; 2], anchor: &str) -> Box4 {
    let n = mm_to_nm;
    let b = text_box(text, n(size), n(at[0]), n(at[1]), anchor);
    [mm(b[0]), mm(b[1]), mm(b[2]), mm(b[3])]
}

fn span(points: &[[f64; 2]]) -> Box4 {
    points
        .iter()
        .fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |b, p| {
            [
                b[0].min(p[0]),
                b[1].min(p[1]),
                b[2].max(p[0]),
                b[3].max(p[1]),
            ]
        })
}

/// Text extents are upper bounds: two texts must overlap by more than this (mm) to count.
const SLACK: f64 = 0.15;

fn overlap(a: &Box4, b: &Box4) -> bool {
    a[0] + SLACK < b[2] && b[0] + SLACK < a[2] && a[1] + SLACK < b[3] && b[1] + SLACK < a[3]
}

/// The layout findings of every page.
pub fn findings(index: &SchematicIndex) -> Vec<LayoutFinding> {
    index.sheets.iter().flat_map(page_findings).collect()
}

/// The layout findings of one page.
pub fn page_findings(page: &SchematicSheet) -> Vec<LayoutFinding> {
    let mut out = Vec::new();
    let sheet = if page.page > 1 {
        format!("{} (page {})", page.path, page.page)
    } else {
        page.path.clone()
    };
    // Everything drawn, by item: (name, box, is a text).
    let mut drawn: Vec<(String, Box4, bool)> = Vec::new();
    for s in &page.symbols {
        let name = designator_text(s);
        for line in &s.body {
            drawn.push((name.clone(), span(line), false));
        }
        drawn.push((
            name.clone(),
            text_mm(&name, s.designator_text_mm, s.designator_at, &s.text_anchor),
            true,
        ));
        if let Some(v) = &s.value {
            drawn.push((
                name.clone(),
                text_mm(v, s.value_text_mm, s.value_at, &s.text_anchor),
                true,
            ));
        }
    }
    for t in &page.texts {
        drawn.push((
            format!("note {}", t.name),
            text_mm(&t.text, t.size_mm, [t.x_mm, t.y_mm], "start"),
            true,
        ));
    }
    for l in &page.labels {
        drawn.push((format!("label {}", l.text), l.extent(), l.kind != "power"));
    }
    for s in &page.sheet_symbols {
        let name = format!("sheet {}", s.path);
        drawn.push((
            name.clone(),
            [s.x_mm, s.y_mm, s.x_mm + s.w_mm, s.y_mm + s.h_mm],
            false,
        ));
        drawn.push((
            name.clone(),
            text_mm(&s.path, 1.6, [s.x_mm, s.y_mm + s.h_mm + 0.8], "start"),
            true,
        ));
        drawn.push((
            name,
            text_mm(&s.sheet, 1.3, [s.x_mm, s.y_mm - 2.0], "start"),
            true,
        ));
    }
    for w in &page.wires {
        drawn.push((format!("wire {}", w.net), span(&w.points), false));
    }
    for b in &page.blankets {
        let name = format!("blanket {}", b.name);
        drawn.push((name.clone(), b.rect, false));
        drawn.push((
            name,
            text_mm(&b.caption, b.caption_mm, b.caption_at, "start"),
            true,
        ));
    }
    // Off the sheet: past the frame line, or over the title block.
    let (w, h) = (page.width_mm, page.height_mm);
    let inside = [FRAME_MM, FRAME_MM, w - FRAME_MM, h - FRAME_MM];
    let rows = page.title_block.len().max(1);
    let title = [
        w - FRAME_MM - TITLE_W_MM,
        FRAME_MM,
        w - FRAME_MM,
        FRAME_MM + title_height_mm(rows),
    ];
    let mut reported: Vec<&str> = Vec::new();
    for (name, b, _) in &drawn {
        if reported.contains(&name.as_str()) {
            continue;
        }
        let past = b[0] < inside[0] - 1e-6
            || b[1] < inside[1] - 1e-6
            || b[2] > inside[2] + 1e-6
            || b[3] > inside[3] + 1e-6;
        // A wire's box spans its bends, so only its run past the frame counts.
        let over_title = !name.starts_with("wire ") && overlap(b, &title);
        if past || over_title {
            reported.push(name);
            out.push(LayoutFinding {
                code: "off_sheet",
                message: format!(
                    "{name} is drawn {} on the {} sheet; move it inside the frame",
                    if past {
                        "past the frame line"
                    } else {
                        "over the title block"
                    },
                    page.frame
                ),
                items: vec![name.clone()],
                sheet: sheet.clone(),
                at_mm: [r2((b[0] + b[2]) / 2.0), r2((b[1] + b[3]) / 2.0)],
            });
        }
    }
    // Texts over each other (each pair once, different items only).
    let texts: Vec<&(String, Box4, bool)> = drawn.iter().filter(|d| d.2).collect();
    for (i, a) in texts.iter().enumerate() {
        for b in &texts[i + 1..] {
            if a.0 != b.0 && overlap(&a.1, &b.1) {
                out.push(LayoutFinding {
                    code: "text_overlap",
                    message: format!("the texts of {} and {} overlap; move one of them", a.0, b.0),
                    items: vec![a.0.clone(), b.0.clone()],
                    sheet: sheet.clone(),
                    at_mm: [
                        r2((a.1[0].max(b.1[0]) + a.1[2].min(b.1[2])) / 2.0),
                        r2((a.1[1].max(b.1[1]) + a.1[3].min(b.1[3])) / 2.0),
                    ],
                });
            }
        }
    }
    out
}
