//! BOM tables: the rendering data of a drawing's bill of materials.
//!
//! A [`BomTableSpec`] says what the table shows: its columns, its structure (parts only, top
//! level, or indented), its sort keys, an optional grouping column, which item kinds it leaves
//! out, and how items are numbered. [`render_bom_table`] turns the model's configuration-aware
//! BOM into rows of cell text, group headers and item numbers, ready for a renderer; balloons
//! read the same numbers, so a balloon and its table row always agree.
//!
//! Item numbers are kept across edits: `numbering.keep` pins an item key to its number (the
//! drawing writes it back when the table is first placed), so adding a part does not renumber
//! every balloon. New items take the next free numbers in display order. Indented tables
//! number hierarchically (`2.1`, `2.1.3`) by position.

use crate::assembly::bom::{Bom, BomItem, BomKind, IndentedRow};
use crate::diagnostics::{Cause, DiagCode, DiagnosticReport, Severity, Site, Subject};
use crate::ids::PropertyName;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "column")]
pub enum BomColumn {
    Item,
    PartNumber,
    Quantity,
    /// The item's `description` property.
    Description,
    Material,
    Configuration,
    Revision,
    /// The defining file (and part output).
    Definition,
    Kind,
    /// The indent level (indented tables).
    Level,
    /// Any item property.
    Property {
        name: PropertyName,
    },
}

impl BomColumn {
    fn key(&self) -> String {
        match self {
            BomColumn::Property { name } => format!("property.{}", name),
            other => serde_json::to_value(other).expect("column serialises")["column"]
                .as_str()
                .unwrap_or("column")
                .to_owned(),
        }
    }

    fn default_title(&self) -> String {
        match self {
            BomColumn::Item => "ITEM".into(),
            BomColumn::PartNumber => "PART NUMBER".into(),
            BomColumn::Quantity => "QTY".into(),
            BomColumn::Description => "DESCRIPTION".into(),
            BomColumn::Material => "MATERIAL".into(),
            BomColumn::Configuration => "CONFIGURATION".into(),
            BomColumn::Revision => "REV".into(),
            BomColumn::Definition => "FILE".into(),
            BomColumn::Kind => "TYPE".into(),
            BomColumn::Level => "LEVEL".into(),
            BomColumn::Property { name } => name.as_str().to_uppercase(),
        }
    }

    fn numeric(&self) -> bool {
        matches!(
            self,
            BomColumn::Item | BomColumn::Quantity | BomColumn::Level
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BomColumnSpec {
    pub column: BomColumn,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl BomColumnSpec {
    pub fn new(column: BomColumn) -> Self {
        Self {
            column,
            title: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BomStructure {
    /// Every part (leaf) summed over the tree.
    #[default]
    PartsOnly,
    /// The top assembly's direct children (sub-assemblies as one line each).
    TopLevel,
    /// Every level, sub-assembly lines followed by their contents.
    Indented,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BomSort {
    pub column: BomColumn,
    #[serde(default)]
    pub descending: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BomNumbering {
    pub start: u32,
    /// Item key ([`bom_item_key`]) → number, kept across edits.
    #[serde(default)]
    pub keep: BTreeMap<String, u32>,
}

impl Default for BomNumbering {
    fn default() -> Self {
        Self {
            start: 1,
            keep: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BomTableSpec {
    #[serde(default = "default_columns")]
    pub columns: Vec<BomColumnSpec>,
    #[serde(default)]
    pub structure: BomStructure,
    /// Sort keys, most significant first (default: item numbering order of the BOM).
    #[serde(default)]
    pub sort: Vec<BomSort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_by: Option<BomColumn>,
    #[serde(default)]
    pub exclude_kinds: Vec<BomKind>,
    #[serde(default)]
    pub numbering: BomNumbering,
}

fn default_columns() -> Vec<BomColumnSpec> {
    [
        BomColumn::Item,
        BomColumn::PartNumber,
        BomColumn::Description,
        BomColumn::Material,
        BomColumn::Quantity,
    ]
    .into_iter()
    .map(BomColumnSpec::new)
    .collect()
}

impl Default for BomTableSpec {
    fn default() -> Self {
        Self {
            columns: default_columns(),
            structure: BomStructure::default(),
            sort: Vec::new(),
            group_by: None,
            exclude_kinds: Vec::new(),
            numbering: BomNumbering::default(),
        }
    }
}

/// Item properties a table shows (description, finish, vendor …), from the definitions.
pub trait BomProperties {
    fn property(&self, item: &BomItem, name: &PropertyName) -> Option<String>;
}

/// A material as a parts list names it: `material.aluminum_6061` → `Aluminum 6061` (the
/// library scope and the identifier's underscores dropped). Materials known by a trade or
/// standard designation print as written in industry: `fr4` → `FR-4`, `abs` → `ABS`.
pub fn material_name(id: &str) -> String {
    identifier_words(id.strip_prefix("material.").unwrap_or(id), true)
}

/// An identifier as parts-list words: underscores become spaces, and designations and
/// acronyms print as written in industry (`esc_heatsink` → `ESC heatsink`, `fr4` → `FR-4`).
/// `capital` capitalizes the first word, for text that starts a cell; without it the words
/// stay lower case (`flight_controller` → `flight controller`), for text after a comma.
pub fn identifier_words(id: &str, capital: bool) -> String {
    let words: Vec<String> = id
        .split('_')
        .filter(|w| !w.is_empty())
        .enumerate()
        .map(|(k, word)| match DESIGNATIONS.iter().find(|(key, _)| key.eq_ignore_ascii_case(word)) {
            Some((_, shown)) => (*shown).to_owned(),
            None if k == 0 && capital => {
                let mut chars = word.chars();
                chars
                    .next()
                    .map(|first| first.to_uppercase().chain(chars).collect())
                    .unwrap_or_default()
            }
            None => word.to_owned(),
        })
        .collect();
    words.join(" ")
}

/// Words a parts list prints as their designation: materials known by a trade or standard
/// name, and the electronics and system acronyms part names use.
const DESIGNATIONS: &[(&str, &str)] = &[
    ("fr4", "FR-4"),
    ("abs", "ABS"),
    ("pla", "PLA"),
    ("petg", "PETG"),
    ("pc", "PC"),
    ("pom", "POM"),
    ("ptfe", "PTFE"),
    ("peek", "PEEK"),
    ("pvc", "PVC"),
    ("hdpe", "HDPE"),
    ("cfrp", "CFRP"),
    ("gfrp", "GFRP"),
    ("lipo", "LiPo"),
    ("esc", "ESC"),
    ("gps", "GPS"),
    ("gnss", "GNSS"),
    ("imu", "IMU"),
    ("mcu", "MCU"),
    ("cpu", "CPU"),
    ("gpu", "GPU"),
    ("pcb", "PCB"),
    ("pcba", "PCBA"),
    ("led", "LED"),
    ("lcd", "LCD"),
    ("usb", "USB"),
    ("ldo", "LDO"),
    ("fpv", "FPV"),
    ("vtx", "VTX"),
    ("rf", "RF"),
    ("adc", "ADC"),
    ("dac", "DAC"),
    ("bms", "BMS"),
    ("emi", "EMI"),
    ("ic", "IC"),
    ("daq", "DAQ"),
];

/// No properties: property columns are empty (and reported).
pub struct NoBomProperties;

impl BomProperties for NoBomProperties {
    fn property(&self, _item: &BomItem, _name: &PropertyName) -> Option<String> {
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Align {
    Left,
    Right,
    Center,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColumnHeader {
    pub key: String,
    pub title: String,
    pub align: Align,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BomTableRow {
    /// The item number as shown (`3`, or `2.1` in an indented table).
    pub number: String,
    pub level: u32,
    pub cells: Vec<String>,
    pub item: BomItem,
    pub quantity: u64,
    /// Occurrence paths of the line (for balloons and cross-highlighting).
    #[serde(default)]
    pub paths: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "line")]
pub enum BomTableLine {
    Group { title: String, rows: u32 },
    Row(Box<BomTableRow>),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BomTableData {
    pub table: String,
    pub columns: Vec<ColumnHeader>,
    pub lines: Vec<BomTableLine>,
    /// Item key → number, for balloons and for writing `numbering.keep` back.
    pub numbers: BTreeMap<String, String>,
    pub total_quantity: u64,
    pub diagnostics: DiagnosticReport,
}

impl BomTableData {
    pub fn rows(&self) -> impl Iterator<Item = &BomTableRow> {
        self.lines.iter().filter_map(|l| match l {
            BomTableLine::Row(r) => Some(&**r),
            _ => None,
        })
    }

    /// The number a balloon on this occurrence shows.
    pub fn number_of_path(&self, path: &str) -> Option<&str> {
        self.rows()
            .find(|r| r.paths.iter().any(|p| p == path))
            .map(|r| r.number.as_str())
    }
}

/// The stable key of a BOM item: everything that makes it its own line (kind, part number,
/// definition, configuration, material and revision).
pub fn bom_item_key(item: &BomItem) -> String {
    format!(
        "{}|{}|{}|{}|{}|{}",
        kind_text(item.kind),
        item.part_number.as_deref().unwrap_or(""),
        item.definition,
        item.configuration,
        item.material.as_ref().map(|m| m.0.as_str()).unwrap_or(""),
        item.revision.as_ref().map(|r| r.0.as_str()).unwrap_or("")
    )
}

/// What a table renders from: the model's flat BOM, or its indented BOM.
pub enum BomSource<'a> {
    Flat(&'a Bom),
    Indented(&'a [IndentedRow]),
}

struct Line {
    level: u32,
    item: BomItem,
    quantity: u64,
    paths: Vec<String>,
    /// Indented tables: the parent line's index.
    parent: Option<usize>,
}

fn kind_text(k: BomKind) -> &'static str {
    match k {
        BomKind::Part => "part",
        BomKind::Assembly => "assembly",
        BomKind::Circuit => "circuit",
        BomKind::Fastener => "fastener",
    }
}

fn site(table: &str, param: &str) -> Site {
    Site::with_param(
        Subject::Value {
            name: format!("table.{}", table),
        },
        param,
    )
}

/// Render a BOM table.
pub fn render_bom_table(
    table: &str,
    spec: &BomTableSpec,
    source: BomSource<'_>,
    props: &dyn BomProperties,
) -> BomTableData {
    let mut out = BomTableData {
        table: table.to_owned(),
        ..Default::default()
    };
    let excluded = |k: BomKind| spec.exclude_kinds.contains(&k);
    // Lines in source order.
    let mut lines: Vec<Line> = Vec::new();
    match (&source, spec.structure) {
        (BomSource::Flat(bom), BomStructure::PartsOnly) => {
            for r in &bom.rows {
                if r.item.kind == BomKind::Assembly || excluded(r.item.kind) {
                    continue;
                }
                lines.push(Line {
                    level: 1,
                    item: r.item.clone(),
                    quantity: r.quantity,
                    paths: r.paths.clone(),
                    parent: None,
                });
            }
        }
        (BomSource::Indented(_), BomStructure::PartsOnly) => {
            out.diagnostics.push(
                DiagCode::DrawingReference,
                Severity::Error,
                site(table, "structure"),
                format!(
                    "table `{}` lists parts only, which needs the model's flat BOM (indented quantities are per parent)",
                    table
                ),
                Cause::Root,
                Vec::new(),
            );
            return out;
        }
        (BomSource::Indented(rows), structure) => {
            // The latest line per (level, parent path) so children find their parent line.
            let mut last_at: BTreeMap<u32, usize> = BTreeMap::new();
            let mut skip_below: Option<u32> = None;
            for r in rows.iter() {
                if let Some(l) = skip_below {
                    if r.level > l {
                        continue;
                    }
                    skip_below = None;
                }
                if structure == BomStructure::TopLevel && r.level > 1 {
                    continue;
                }
                if excluded(r.item.kind) {
                    skip_below = Some(r.level);
                    continue;
                }
                let parent = if r.level > 1 && structure == BomStructure::Indented {
                    last_at.get(&(r.level - 1)).copied()
                } else {
                    None
                };
                lines.push(Line {
                    level: if structure == BomStructure::Indented {
                        r.level
                    } else {
                        1
                    },
                    item: r.item.clone(),
                    quantity: r.quantity,
                    paths: Vec::new(),
                    parent,
                });
                last_at.insert(r.level, lines.len() - 1);
                last_at.retain(|l, _| *l <= r.level);
            }
        }
        (BomSource::Flat(_), s) => {
            out.diagnostics.push(
                DiagCode::DrawingReference,
                Severity::Error,
                site(table, "structure"),
                format!(
                    "table `{}` is {:?}, which needs the model's indented BOM; the model gives only a flat one",
                    table, s
                ),
                Cause::Root,
                Vec::new(),
            );
            return out;
        }
    }

    // Cell text per line and column.
    let mut missing: BTreeMap<String, u32> = BTreeMap::new();
    let mut cell = |l: &Line, c: &BomColumn| -> String {
        let prop = |name: &str, missing: &mut BTreeMap<String, u32>| {
            props
                .property(&l.item, &PropertyName::new(name))
                .unwrap_or_else(|| {
                    *missing.entry(name.to_owned()).or_insert(0) += 1;
                    String::new()
                })
        };
        match c {
            BomColumn::Item => String::new(),
            BomColumn::PartNumber => l.item.part_number.clone().unwrap_or_default(),
            BomColumn::Quantity => l.quantity.to_string(),
            BomColumn::Description => prop("description", &mut missing),
            BomColumn::Material => l
                .item
                .material
                .as_ref()
                .map(|m| material_name(&m.0))
                .unwrap_or_default(),
            BomColumn::Configuration => l.item.configuration.clone(),
            BomColumn::Revision => l
                .item
                .revision
                .as_ref()
                .map(|r| r.0.clone())
                .unwrap_or_default(),
            BomColumn::Definition => l.item.definition.to_string(),
            BomColumn::Kind => kind_text(l.item.kind).into(),
            BomColumn::Level => l.level.to_string(),
            BomColumn::Property { name } => prop(name.as_str(), &mut missing),
        }
    };
    let mut cols: Vec<BomColumn> = spec.columns.iter().map(|c| c.column.clone()).collect();
    for s in &spec.sort {
        if !cols.contains(&s.column) {
            cols.push(s.column.clone());
        }
    }
    if let Some(g) = &spec.group_by {
        if !cols.contains(g) {
            cols.push(g.clone());
        }
    }
    let text: Vec<BTreeMap<BomColumn, String>> = lines
        .iter()
        .map(|l| cols.iter().map(|c| (c.clone(), cell(l, c))).collect())
        .collect();
    for (name, n) in &missing {
        out.diagnostics.push(
            DiagCode::DrawingReference,
            Severity::Warning,
            site(table, &format!("property.{}", name)),
            format!(
                "table `{}` shows `{}`, which {} item(s) do not have",
                table, name, n
            ),
            Cause::Root,
            Vec::new(),
        );
    }

    // Order: indented tables keep the tree order (siblings sorted); flat ones sort fully.
    let cmp_lines = |a: usize, b: usize| -> std::cmp::Ordering {
        for s in &spec.sort {
            let (x, y) = (&text[a][&s.column], &text[b][&s.column]);
            let o = if s.column.numeric() {
                let p = |v: &str| v.parse::<f64>().unwrap_or(0.0);
                p(x).partial_cmp(&p(y)).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                x.cmp(y)
            };
            let o = if s.descending { o.reverse() } else { o };
            if o != std::cmp::Ordering::Equal {
                return o;
            }
        }
        a.cmp(&b)
    };
    let order: Vec<usize> = if spec.structure == BomStructure::Indented {
        fn visit(
            parent: Option<usize>,
            lines: &[Line],
            cmp: &dyn Fn(usize, usize) -> std::cmp::Ordering,
            out: &mut Vec<usize>,
        ) {
            let mut kids: Vec<usize> = (0..lines.len())
                .filter(|i| lines[*i].parent == parent)
                .collect();
            kids.sort_by(|a, b| cmp(*a, *b));
            for k in kids {
                out.push(k);
                visit(Some(k), lines, cmp, out);
            }
        }
        let mut o = Vec::new();
        visit(None, &lines, &cmp_lines, &mut o);
        o
    } else {
        let mut o: Vec<usize> = (0..lines.len()).collect();
        o.sort_by(|a, b| cmp_lines(*a, *b));
        o
    };
    // Grouping (flat tables): groups in key order, rows in sort order within each.
    let groups: Vec<(Option<String>, Vec<usize>)> = match (&spec.group_by, spec.structure) {
        (Some(g), s) if s != BomStructure::Indented => {
            let mut by: BTreeMap<String, Vec<usize>> = BTreeMap::new();
            for &i in &order {
                by.entry(text[i][g].clone()).or_default().push(i);
            }
            by.into_iter().map(|(k, v)| (Some(k), v)).collect()
        }
        _ => vec![(None, order.clone())],
    };

    // Numbers: kept ones first, then the next free numbers in display order.
    let mut numbers: Vec<String> = vec![String::new(); lines.len()];
    if spec.structure == BomStructure::Indented {
        let mut counters: BTreeMap<Option<usize>, u32> = BTreeMap::new();
        for &i in &order {
            let n = counters
                .entry(lines[i].parent)
                .or_insert(spec.numbering.start - 1);
            *n += 1;
            numbers[i] = match lines[i].parent {
                Some(p) => format!("{}.{}", numbers[p], n),
                None => n.to_string(),
            };
        }
    } else {
        let mut used: BTreeSet<u32> = BTreeSet::new();
        let mut kept: BTreeMap<u32, String> = BTreeMap::new();
        for (i, l) in lines.iter().enumerate() {
            let key = bom_item_key(&l.item);
            if let Some(&n) = spec.numbering.keep.get(&key) {
                if let Some(other) = kept.insert(n, key.clone()) {
                    out.diagnostics.push(
                        DiagCode::DrawingInvalid,
                        Severity::Error,
                        site(table, "numbering.keep"),
                        format!(
                            "table `{}` keeps number {} for both `{}` and `{}`",
                            table, n, other, key
                        ),
                        Cause::Root,
                        Vec::new(),
                    );
                    continue;
                }
                used.insert(n);
                numbers[i] = n.to_string();
            }
        }
        let mut next = spec.numbering.start;
        for (_, rows) in &groups {
            for &i in rows {
                if !numbers[i].is_empty() {
                    continue;
                }
                while used.contains(&next) {
                    next += 1;
                }
                used.insert(next);
                numbers[i] = next.to_string();
            }
        }
    }

    out.columns = spec
        .columns
        .iter()
        .map(|c| ColumnHeader {
            key: c.column.key(),
            title: c.title.clone().unwrap_or_else(|| c.column.default_title()),
            align: if c.column.numeric() {
                Align::Right
            } else {
                Align::Left
            },
        })
        .collect();
    for (title, rows) in groups {
        if let Some(t) = title {
            out.lines.push(BomTableLine::Group {
                title: t,
                rows: rows.len() as u32,
            });
        }
        for i in rows {
            let l = &lines[i];
            let cells = spec
                .columns
                .iter()
                .map(|c| match c.column {
                    BomColumn::Item => numbers[i].clone(),
                    _ => text[i][&c.column].clone(),
                })
                .collect();
            out.total_quantity += l.quantity;
            out.numbers
                .entry(bom_item_key(&l.item))
                .or_insert_with(|| numbers[i].clone());
            out.lines.push(BomTableLine::Row(Box::new(BomTableRow {
                number: numbers[i].clone(),
                level: l.level,
                cells,
                item: l.item.clone(),
                quantity: l.quantity,
                paths: l.paths.clone(),
            })));
        }
    }
    out
}

#[cfg(test)]
mod material_name_tests {
    use super::{identifier_words, material_name};

    #[test]
    fn identifiers_print_acronyms_upper_case_and_underscores_as_spaces() {
        assert_eq!(identifier_words("esc_heatsink", true), "ESC heatsink");
        assert_eq!(identifier_words("esc", false), "ESC");
        assert_eq!(identifier_words("flight_controller", false), "flight controller");
        assert_eq!(identifier_words("gps_mount", true), "GPS mount");
        assert_eq!(identifier_words("body_top", true), "Body top");
        assert_eq!(material_name("lipo_pack"), "LiPo pack");
    }

    #[test]
    fn materials_print_by_name_and_designation() {
        assert_eq!(material_name("material.aluminum_6061"), "Aluminum 6061");
        assert_eq!(material_name("material.fr4"), "FR-4");
        assert_eq!(material_name("steel_1018"), "Steel 1018");
        assert_eq!(material_name("abs"), "ABS");
        assert_eq!(material_name("glass_filled_peek"), "Glass filled PEEK");
    }
}
