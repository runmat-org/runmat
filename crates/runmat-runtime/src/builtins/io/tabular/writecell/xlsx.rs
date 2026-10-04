use super::*;

pub(in crate::builtins::io::tabular) fn build_xlsx_workbook(
    table: &CellTable,
    sheet_name: &str,
    start: RangeStart,
) -> BuiltinResult<Vec<u8>> {
    let cursor = Cursor::new(Vec::new());
    let mut zip = zip::ZipWriter::new(cursor);
    write_xlsx_part(
        &mut zip,
        "[Content_Types].xml",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"#,
    )?;
    write_xlsx_part(
        &mut zip,
        "_rels/.rels",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#,
    )?;
    write_xlsx_part(
        &mut zip,
        "xl/workbook.xml",
        &format!(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="{}" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"#,
            xml_attr_escape(sheet_name)
        ),
    )?;
    write_xlsx_part(
        &mut zip,
        "xl/_rels/workbook.xml.rels",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"#,
    )?;
    write_xlsx_part(
        &mut zip,
        "xl/styles.xml",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellXfs>
</styleSheet>"#,
    )?;
    write_xlsx_part(
        &mut zip,
        "xl/worksheets/sheet1.xml",
        &build_sheet_xml(table, start),
    )?;
    let cursor = zip.finish().map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!("writecell: failed to finish spreadsheet package ({err})"),
            err,
        )
    })?;
    Ok(cursor.into_inner())
}

pub(in crate::builtins::io::tabular) fn write_xlsx_part(
    zip: &mut zip::ZipWriter<Cursor<Vec<u8>>>,
    name: &str,
    contents: &str,
) -> BuiltinResult<()> {
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    zip.start_file(name, options).map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!("writecell: failed to start spreadsheet part {name} ({err})"),
            err,
        )
    })?;
    zip.write_all(contents.as_bytes()).map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!("writecell: failed to write spreadsheet part {name} ({err})"),
            err,
        )
    })?;
    Ok(())
}

pub(in crate::builtins::io::tabular) fn build_sheet_xml(
    table: &CellTable,
    start: RangeStart,
) -> String {
    let mut xml = String::from(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
"#,
    );
    for row in 0..table.rows {
        let excel_row = start.row + row + 1;
        xml.push_str(&format!(r#"    <row r="{excel_row}">"#));
        xml.push('\n');
        for col in 0..table.cols {
            let cell = table.get(row, col);
            if *cell == CellValue::Empty {
                continue;
            }
            let reference = cell_reference(start.row + row, start.col + col);
            match cell {
                CellValue::Empty => {}
                CellValue::Number(value) => {
                    xml.push_str(&format!(
                        "      <c r=\"{reference}\"><v>{}</v></c>\n",
                        format_numeric(*value)
                    ));
                }
                CellValue::Integer(value) => {
                    xml.push_str(&format!(
                        "      <c r=\"{reference}\"><v>{}</v></c>\n",
                        value.decimal_string()
                    ));
                }
                CellValue::Boolean(value) => {
                    xml.push_str(&format!(
                        "      <c r=\"{reference}\" t=\"b\"><v>{}</v></c>\n",
                        if *value { 1 } else { 0 }
                    ));
                }
                CellValue::Text(text) => {
                    xml.push_str(&format!(
                        "      <c r=\"{reference}\" t=\"inlineStr\"><is><t>{}</t></is></c>\n",
                        xml_text_escape(text)
                    ));
                }
            }
        }
        xml.push_str("    </row>\n");
    }
    xml.push_str("  </sheetData>\n</worksheet>");
    xml
}
