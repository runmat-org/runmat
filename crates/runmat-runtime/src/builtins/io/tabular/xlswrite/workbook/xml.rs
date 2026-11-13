//! Cooperative XML projection for the selected workbook owner.

use super::*;

pub(super) async fn build_sheet_xml_cooperative(
    table: &CellTable,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
) -> BuiltinResult<String> {
    let mut xml = String::from(writecell::SHEET_XML_HEADER);
    let mut cells = 0;
    let mut rows = 0;
    for row in 0..table.rows {
        let excel_row = row + 1;
        xml.push_str(&format!(r#"    <row r="{excel_row}">"#));
        xml.push('\n');
        for col in 0..table.cols {
            let cell = table.get(row, col);
            if *cell != CellValue::Empty {
                let reference = writecell::cell_reference(row, col);
                match cell {
                    CellValue::Text(text) => {
                        writecell::append_sheet_text_start(&mut xml, &reference);
                        let mut bytes = 0;
                        for character in text.chars() {
                            writecell::append_escaped_xml_text_character(&mut xml, character);
                            bytes += character.len_utf8();
                            if bytes >= 4096 {
                                checkpoint_xml(cooperation, selected, &xml, bytes).await?;
                                bytes = 0;
                            }
                        }
                        if bytes > 0 {
                            checkpoint_xml(cooperation, selected, &xml, bytes).await?;
                        }
                        writecell::append_sheet_text_end(&mut xml);
                    }
                    other => writecell::append_sheet_cell_xml(&mut xml, &reference, other),
                }
            }
            cells += 1;
            if cells == 1024 {
                checkpoint_xml(cooperation, selected, &xml, cells).await?;
                cells = 0;
            }
        }
        xml.push_str("    </row>\n");
        rows += 1;
        if rows == 1024 {
            checkpoint_xml(cooperation, selected, &xml, rows).await?;
            rows = 0;
        }
    }
    xml.push_str(writecell::SHEET_XML_FOOTER);
    checkpoint_xml(cooperation, selected, &xml, cells + rows).await?;
    Ok(xml)
}

async fn checkpoint_xml(
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    xml: &str,
    work: usize,
) -> BuiltinResult<()> {
    cooperation.completed(work).await?;
    selected.check().map_err(|source| {
        xlswrite_error_with_source(
            &XLSWRITE_ERROR_IO,
            "xlswrite: captured filesystem authority was revoked",
            source,
        )
    })?;
    if xml.len() > MAX_XLSWRITE_OUTPUT_BYTES {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_IO,
            "xlswrite: worksheet XML exceeds maximum supported size",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "xml/tests.rs"]
mod tests;
