//! Sheets and the drafting standard: `sheet.add` adds a sheet in a size (its format's title
//! block copied from the drawing's first format), `drawing.standard` sets the standard every
//! format follows (ISO 128/129-1/1101 or ASME Y14.3/Y14.5).

use super::*;
use runmat_cad::drawing::standard::DrawingStandard;
use runmat_cad::drawing::{Projection, Sheet, SheetFormat, SheetSize};

fn size(word: &str) -> Result<SheetSize, EditFailure> {
    let w = word.trim().to_lowercase();
    Ok(match w.as_str() {
        "a0" | "a1" | "a2" | "a3" | "a4" => SheetSize::Iso { name: w.to_uppercase() },
        "a" | "b" | "c" | "d" | "e" => SheetSize::Ansi { name: w.to_uppercase() },
        other => return Err(model(format!("there is no `{other}` sheet size"))),
    })
}

pub fn edit(path: &str, text_: &str, files: &Files, command: &str, values: &Values) -> Option<Result<(SourceChange, String), EditFailure>> {
    if !["sheet.add", "drawing.standard"].contains(&command) {
        return None;
    }
    Some((|| {
        let mut d = Drawing::open(path, text_, files)?;
        let summary = match command {
            "sheet.add" => {
                let size = size(text(values, "size", "a3"))?;
                let label = match &size {
                    SheetSize::Iso { name } | SheetSize::Ansi { name } => name.clone(),
                    SheetSize::Custom { .. } => unreachable!(),
                };
                let format = match d.document.formats.iter().find(|f| f.size == size) {
                    Some(f) => f.name.clone(),
                    None => {
                        let name = label.to_lowercase();
                        let name = if d.document.formats.iter().any(|f| f.name == name) { format!("{name}_sheet") } else { name };
                        let first = d.document.formats.first();
                        let format = SheetFormat {
                            name: name.clone(),
                            size,
                            title_block: first.map(|f| f.title_block.clone()).unwrap_or_default(),
                            standard: first.map(|f| f.standard).unwrap_or_default(),
                        };
                        d.document.formats.push(format);
                        name
                    }
                };
                let wanted = text(values, "name", "").trim().to_owned();
                let taken = |n: &str| d.document.sheets.iter().any(|s| s.name == n);
                let name = if !wanted.is_empty() && !taken(&wanted) {
                    wanted
                } else if !wanted.is_empty() {
                    return Err(model(format!("the drawing has a sheet `{wanted}` already")));
                } else {
                    (1..).map(|n| n.to_string()).find(|n| !taken(n)).expect("a free name")
                };
                let projection = d.document.sheets.first().map(|s| s.projection).unwrap_or(Projection::ThirdAngle);
                d.document.sheets.push(Sheet { name: name.clone(), format, scale: (1, 1), projection });
                format!("Added sheet {name} ({label})")
            }
            "drawing.standard" => {
                let standard = match text(values, "standard", "iso") {
                    "iso" => DrawingStandard::Iso,
                    "asme" | "ansi" => DrawingStandard::Asme,
                    other => return Err(model(format!("there is no `{other}` drafting standard"))),
                };
                if d.document.formats.is_empty() {
                    return Err(model("add a sheet first"));
                }
                for f in &mut d.document.formats {
                    f.standard = standard;
                }
                format!("Set the drafting standard to {}", if standard == DrawingStandard::Iso { "ISO" } else { "ASME" })
            }
            _ => unreachable!(),
        };
        Ok((d.change()?, summary))
    })())
}
