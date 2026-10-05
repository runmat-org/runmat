use super::*;

#[path = "workbook/xml.rs"]
mod xml;
use xml::build_sheet_xml_cooperative;
#[path = "workbook/validation.rs"]
mod validation;
use validation::{ensure_runmat_xlswrite_workbook, preflight_existing_workbook};

#[derive(Debug, Clone)]
pub(super) struct XlsWorkbook {
    sheets: Vec<XlsSheet>,
}

#[derive(Debug, Clone)]
struct XlsSheet {
    name: String,
    table: XlsTable,
}

impl XlsWorkbook {
    fn new() -> Self {
        Self { sheets: Vec::new() }
    }

    pub(super) async fn overlay(
        &mut self,
        request: &XlsWriteRequest,
        table: XlsTable,
        cooperation: &mut TabularCooperation<'_>,
        selected: &TabularFsPath,
    ) -> BuiltinResult<()> {
        validate_write_bounds(&table, request.range)?;
        let sheet_index = self.resolve_sheet_index(&request.sheet);
        // A suspended or failed overlay drops its private candidate. The retained
        // workbook still owns a valid empty sheet, never stale geometry with no cells.
        let previous = std::mem::replace(&mut self.sheets[sheet_index].table, XlsTable::empty());
        self.sheets[sheet_index].table = previous
            .overlay(request.range, table, cooperation, selected)
            .await?;
        Ok(())
    }

    fn resolve_sheet_index(&mut self, selector: &SheetSelector) -> usize {
        match selector {
            SheetSelector::Default => {
                if self.sheets.is_empty() {
                    self.sheets.push(XlsSheet::empty("Sheet1"));
                }
                0
            }
            SheetSelector::Name(name) => {
                if let Some(index) = self
                    .sheets
                    .iter()
                    .position(|sheet| sheet.name.eq_ignore_ascii_case(name))
                {
                    index
                } else {
                    self.sheets.push(XlsSheet::empty(name));
                    self.sheets.len() - 1
                }
            }
            SheetSelector::Index(index) => {
                while self.sheets.len() <= *index {
                    let next = self.sheets.len() + 1;
                    let name = self.unique_sheet_name(&format!("Sheet{next}"));
                    self.sheets.push(XlsSheet::empty(&name));
                }
                *index
            }
        }
    }

    fn unique_sheet_name(&self, base: &str) -> String {
        if !self
            .sheets
            .iter()
            .any(|sheet| sheet.name.eq_ignore_ascii_case(base))
        {
            return base.to_string();
        }
        for suffix in 1..=MAX_XLSWRITE_SHEETS {
            let candidate = make_suffixed_sheet_name(base, suffix);
            if !self
                .sheets
                .iter()
                .any(|sheet| sheet.name.eq_ignore_ascii_case(&candidate))
            {
                return candidate;
            }
        }
        "Sheet".to_string()
    }

    pub(super) async fn into_xlsx_bytes(
        self,
        cooperation: &mut TabularCooperation<'_>,
        selected: &TabularFsPath,
    ) -> BuiltinResult<Vec<u8>> {
        let sheets = if self.sheets.is_empty() {
            vec![XlsSheet::empty("Sheet1")]
        } else {
            self.sheets
        };
        let cursor = Cursor::new(Vec::new());
        let mut zip = zip::ZipWriter::new(cursor);
        writecell::write_xlsx_part(
            &mut zip,
            "[Content_Types].xml",
            &content_types_xml(sheets.len()),
        )
        .map_err(map_writecell_error)?;
        writecell::write_xlsx_part(&mut zip, "_rels/.rels", ROOT_RELS_XML)
            .map_err(map_writecell_error)?;
        writecell::write_xlsx_part(&mut zip, "xl/workbook.xml", &workbook_xml(&sheets))
            .map_err(map_writecell_error)?;
        writecell::write_xlsx_part(
            &mut zip,
            "xl/_rels/workbook.xml.rels",
            &workbook_relationships_xml(sheets.len()),
        )
        .map_err(map_writecell_error)?;
        writecell::write_xlsx_part(&mut zip, "xl/styles.xml", STYLES_XML)
            .map_err(map_writecell_error)?;
        writecell::write_xlsx_part(&mut zip, XLSWRITE_MARKER_PART, &marker_app_xml())
            .map_err(map_writecell_error)?;

        for (index, sheet) in sheets.into_iter().enumerate() {
            let cell_table = sheet.table.into_cell_table()?;
            let xml = build_sheet_xml_cooperative(&cell_table, cooperation, selected).await?;
            if xml.len() > MAX_XLSWRITE_OUTPUT_BYTES {
                return Err(xlswrite_error_with(
                    &XLSWRITE_ERROR_IO,
                    "xlswrite: worksheet XML exceeds maximum supported size",
                ));
            }
            writecell::write_xlsx_part(
                &mut zip,
                &format!("xl/worksheets/sheet{}.xml", index + 1),
                &xml,
            )
            .map_err(map_writecell_error)?;
            cooperation.completed(1).await?;
            selected.check().map_err(|source| {
                xlswrite_error_with_source(
                    &XLSWRITE_ERROR_IO,
                    "xlswrite: captured filesystem authority was revoked",
                    source,
                )
            })?;
        }

        let cursor = zip.finish().map_err(|err| {
            xlswrite_error_with_source(
                &XLSWRITE_ERROR_IO,
                format!("xlswrite: failed to finish spreadsheet package ({err})"),
                err,
            )
        })?;
        let bytes = cursor.into_inner();
        if bytes.len() > MAX_XLSWRITE_OUTPUT_BYTES {
            return Err(xlswrite_error_with(
                &XLSWRITE_ERROR_IO,
                format!(
                    "xlswrite: workbook exceeds maximum supported output size of {MAX_XLSWRITE_OUTPUT_BYTES} bytes"
                ),
            ));
        }
        Ok(bytes)
    }
}

impl XlsSheet {
    fn empty(name: &str) -> Self {
        Self {
            name: sanitize_sheet_name(name),
            table: XlsTable::empty(),
        }
    }
}

const ROOT_RELS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"#;

const STYLES_XML: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellXfs>
</styleSheet>"#;

fn content_types_xml(sheet_count: usize) -> String {
    let mut xml = String::from(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
"#,
    );
    for index in 1..=sheet_count {
        xml.push_str(&format!(
            r#"  <Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
"#
        ));
    }
    xml.push_str(
        r#"  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"#,
    );
    xml
}

fn marker_app_xml() -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>{}</Application>
</Properties>"#,
        XLSWRITE_MARKER_TEXT
    )
}

fn workbook_xml(sheets: &[XlsSheet]) -> String {
    let mut xml = String::from(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
"#,
    );
    for (index, sheet) in sheets.iter().enumerate() {
        let sheet_id = index + 1;
        xml.push_str(&format!(
            r#"    <sheet name="{}" sheetId="{sheet_id}" r:id="rId{sheet_id}"/>
"#,
            writecell::xml_attr_escape(&sheet.name)
        ));
    }
    xml.push_str("  </sheets>\n</workbook>");
    xml
}

fn make_suffixed_sheet_name(base: &str, suffix: usize) -> String {
    let suffix = format!("_{suffix}");
    let max_base_chars = 31usize.saturating_sub(suffix.chars().count());
    let mut name: String = base.chars().take(max_base_chars).collect();
    name.push_str(&suffix);
    name
}

fn workbook_relationships_xml(sheet_count: usize) -> String {
    let mut xml = String::from(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
"#,
    );
    for index in 1..=sheet_count {
        xml.push_str(&format!(
            r#"  <Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>
"#
        ));
    }
    xml.push_str(&format!(
        r#"  <Relationship Id="rId{}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"#,
        sheet_count + 1
    ));
    xml
}

pub(super) async fn load_workbook_or_new(
    selected: &TabularFsPath,
    cooperation: &mut TabularCooperation<'_>,
) -> BuiltinResult<XlsWorkbook> {
    match read_existing_workbook_bytes(selected, cooperation).await? {
        Some(bytes) => workbook_from_bytes(selected, bytes, cooperation).await,
        None => Ok(XlsWorkbook::new()),
    }
}

async fn read_existing_workbook_bytes(
    selected: &TabularFsPath,
    cooperation: &mut TabularCooperation<'_>,
) -> BuiltinResult<Option<Vec<u8>>> {
    let path = selected.path();
    let file = match selected.open_read().await {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(xlswrite_error_with_source(
                &XLSWRITE_ERROR_IO,
                format!(
                    "xlswrite: unable to open existing '{}': {err}",
                    path.display()
                ),
                err,
            ))
        }
    };
    checkpoint_existing(cooperation, selected, 0).await?;
    let mut limited = file.take(MAX_XLSWRITE_WORKBOOK_BYTES + 1);
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        selected.check().map_err(|source| {
            xlswrite_error_with_source(
                &XLSWRITE_ERROR_IO,
                "xlswrite: captured filesystem authority was revoked",
                source,
            )
        })?;
        let read = limited.read(&mut chunk).map_err(|err| {
            xlswrite_error_with_source(
                &XLSWRITE_ERROR_IO,
                format!(
                    "xlswrite: unable to read existing '{}': {err}",
                    path.display()
                ),
                err,
            )
        })?;
        if read == 0 {
            break;
        }
        bytes.try_reserve(read).map_err(|source| {
            xlswrite_error_with_source(
                &XLSWRITE_ERROR_IO,
                "xlswrite: existing workbook read allocation failed",
                source,
            )
        })?;
        bytes.extend_from_slice(&chunk[..read]);
        checkpoint_existing(cooperation, selected, read).await?;
    }
    selected.check().map_err(|source| {
        xlswrite_error_with_source(
            &XLSWRITE_ERROR_IO,
            "xlswrite: captured filesystem authority was revoked",
            source,
        )
    })?;
    if bytes.len() as u64 > MAX_XLSWRITE_WORKBOOK_BYTES {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_IO,
            format!(
                "xlswrite: existing workbook exceeds maximum supported size of {MAX_XLSWRITE_WORKBOOK_BYTES} bytes"
            ),
        ));
    }
    Ok(Some(bytes))
}

async fn workbook_from_bytes(
    selected: &TabularFsPath,
    bytes: Vec<u8>,
    cooperation: &mut TabularCooperation<'_>,
) -> BuiltinResult<XlsWorkbook> {
    let path = selected.path();
    preflight_existing_workbook(&bytes, cooperation, selected).await?;
    ensure_runmat_xlswrite_workbook(path, &bytes, cooperation, selected).await?;
    checkpoint_existing(cooperation, selected, 0).await?;
    let mut workbook = open_workbook_auto_from_rs(Cursor::new(bytes)).map_err(|err| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_IO,
            format!(
                "xlswrite: unable to open existing spreadsheet '{}': {err}",
                path.display()
            ),
        )
    })?;
    checkpoint_existing(cooperation, selected, 0).await?;
    let names = workbook.sheet_names().to_vec();
    let mut sheets = Vec::with_capacity(names.len());
    for name in names {
        let name = validate_sheet_name(&name)?;
        let range = workbook.worksheet_range(&name).map_err(|err| {
            xlswrite_error_with(
                &XLSWRITE_ERROR_IO,
                format!("xlswrite: unable to read existing sheet '{name}': {err:?}"),
            )
        })?;
        checkpoint_existing(cooperation, selected, 0).await?;
        sheets.push(XlsSheet {
            name,
            table: XlsTable::from_calamine_range_cooperative(&range, cooperation, selected).await?,
        });
    }
    Ok(XlsWorkbook { sheets })
}

async fn checkpoint_existing(
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    work: usize,
) -> BuiltinResult<()> {
    cooperation.completed(work).await?;
    selected.check().map_err(|source| {
        xlswrite_error_with_source(
            &XLSWRITE_ERROR_IO,
            "xlswrite: captured filesystem authority was revoked",
            source,
        )
    })
}

#[cfg(test)]
#[path = "workbook/tests.rs"]
mod tests;
