//! The `.drawing` model: sheets, formats and title blocks; views defined against the model by
//! references; associative dimensions and annotations; tables; and diagnostics when anything a
//! drawing points at no longer resolves.
//!
//! Geometry comes through two seams, so this module stays kernel-free:
//! - [`HiddenLineRemover`]: a view's projected edges with their entity names (OCCT `HLRBRep`
//!   at K1), cached per view by the model's chained key and the view definition;
//! - [`ModelReferences`]: resolve a [`Reference`] against the model a view shows (a part's
//!   regeneration, an occurrence's part inside a resolved assembly).
//!
//! - [`bom_table`]: BOM-table rendering data (columns, sort, groups, kept item numbers);
//! - [`flat_pattern`]: sheet metal flat patterns as 2D sheets, bend tables and DXF;
//! - [`hole_table`]: hole-table rows through the [`hole_table::HoleGeometry`] seam;
//! - [`merge`]: three-way drawing merges by stable id, with reference re-resolution;
//! - [`revision_table`]: the revision table's rows, which the title block's revision reads.
//!
//! Annotations carry stable ids, so drawings merge like any other design file. A dimension
//! names model entities (never projected 2D positions), so it follows the model; when its
//! references break it is reported with the view and annotation, and kept (dangling) rather
//! than dropped.

pub mod bom_table;
pub mod cut_list_table;
pub mod flat_pattern;
pub mod hole_table;
pub mod index;
pub mod logo;
pub mod merge;
pub mod revision_table;
pub mod standard;
pub mod title_layout;

use crate::assembly::bom::{Bom, BomItem, BomKind, IndentedRow};
use crate::assembly::model::RelPath;
use crate::assembly::options::AxisValue;
use crate::diagnostics::{Cause, DiagCode, DiagnosticReport, Severity, Site, Subject};
use crate::ids::{ConfigurationName, FilePath, PropertyName};
use crate::model::VersionHeader;
use crate::refs::name::EntityName;
use crate::refs::reference::Reference;
use crate::regen::kernel::KernelError;
use crate::value::ValueExpr;
use bom_table::{
    bom_item_key, render_bom_table, BomProperties, BomSource, BomStructure, BomTableData,
    BomTableSpec,
};
use hole_table::{hole_table, HoleGeometry, HoleTableData, HoleTableSpec};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "size")]
pub enum SheetSize {
    Iso { name: String },
    Ansi { name: String },
    Custom { width: ValueExpr, height: ValueExpr },
}

/// Where a title-block field takes its text.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "source")]
pub enum FieldSource {
    Literal {
        text: String,
    },
    /// A property of the drawn model (part number, material, mass, drawn by …).
    Property {
        name: PropertyName,
    },
    /// The revision letter from the release record.
    Revision,
    SheetNumber,
    SheetCount,
    Scale,
    /// A logo: the outline of an SVG file, by path from the drawing's folder
    /// (`{ source = image, file = "logo.svg" }`). The sheet and every export draw it in the
    /// field's cell ([`title_layout`]).
    Image {
        file: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TitleField {
    pub label: String,
    pub source: FieldSource,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SheetFormat {
    pub name: String,
    pub size: SheetSize,
    pub title_block: Vec<TitleField>,
    /// The drafting standard its sheets follow (`standard = asme`; ISO when absent): line
    /// weights, figure heights, dimension style and the frames GD&T may show
    /// ([`standard::DrawingStandard::style`]).
    #[serde(default, skip_serializing_if = "standard::DrawingStandard::is_iso")]
    pub standard: standard::DrawingStandard,
}

impl SheetFormat {
    /// The title block's fields: the format's own, else the standard set every sheet carries
    /// (ISO 7200's identification fields): title (the drawing's file name), part number,
    /// material, scale, sheet and sheet count, revision and the drafting standard.
    pub fn title_fields(&self, title: &str) -> Vec<TitleField> {
        if !self.title_block.is_empty() {
            return self.title_block.clone();
        }
        let field = |label: &str, source: FieldSource| TitleField {
            label: label.into(),
            source,
        };
        let property = |name: &str| FieldSource::Property {
            name: PropertyName::new(name),
        };
        vec![
            field(
                "TITLE",
                FieldSource::Literal {
                    text: title.to_uppercase(),
                },
            ),
            field("PART NUMBER", property("part_number")),
            field("MATERIAL", property("material")),
            field(
                "STANDARD",
                FieldSource::Literal {
                    text: self.standard.style().name.into(),
                },
            ),
            field("SCALE", FieldSource::Scale),
            field("SHEET", FieldSource::SheetNumber),
            field("OF", FieldSource::SheetCount),
            field("REV", FieldSource::Revision),
        ]
    }
}

/// A drawing's title as its title block states it: its file name without the folder or `.drawing`.
pub fn drawing_title(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.strip_suffix(".drawing")
        .unwrap_or(name)
        .replace(['_', '-'], " ")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Projection {
    FirstAngle,
    ThirdAngle,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sheet {
    pub name: String,
    pub format: String,
    /// `numerator : denominator`.
    pub scale: (u32, u32),
    pub projection: Projection,
}

/// What a view shows: a part (configuration) or an assembly (configuration and variant).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelRef {
    pub file: FilePath,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configuration: Option<ConfigurationName>,
    #[serde(default)]
    pub variant: BTreeMap<String, AxisValue>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Orientation {
    Front,
    Back,
    Top,
    Bottom,
    Left,
    Right,
    Isometric,
    Dimetric,
    Trimetric,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectDirection {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "view")]
pub enum ViewKind {
    /// A model view in a named orientation (or a frame published by the model).
    Standard {
        orientation: Orientation,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        frame: Option<Reference>,
    },
    Projected {
        from: String,
        direction: ProjectDirection,
    },
    /// A cut by a plane through the model (a face or datum plane reference) seen from `from`.
    Section {
        from: String,
        plane: Reference,
        label: String,
        #[serde(default)]
        aligned: bool,
    },
    /// An enlarged circle of `from` around a model point.
    Detail {
        from: String,
        center: Reference,
        radius: ValueExpr,
        label: String,
    },
    /// Perpendicular to a model edge seen in `from`.
    Auxiliary {
        from: String,
        edge: Reference,
        label: String,
    },
    /// Material removed to a depth inside a closed boundary on `from`.
    BrokenOut {
        from: String,
        boundary: Vec<[f64; 2]>,
        depth: ValueExpr,
    },
    /// Long parts with pieces removed between pairs of model points.
    Break {
        from: String,
        gaps: Vec<(Reference, Reference)>,
    },
    Crop {
        from: String,
        boundary: Vec<[f64; 2]>,
    },
    /// `from` overlaid with the model in another configuration (mechanism travel).
    AlternatePosition {
        from: String,
        configuration: ConfigurationName,
    },
    /// The flat-pattern sheet of a sheet metal part: the developed outline of the flat
    /// pattern (or unfold) feature `feature`, its bend lines with notes ("B1 UP 90° R3 BA
    /// 6.095") and, with `bend_table`, the bend table beside it
    /// ([`flat_pattern::flat_pattern_sheet`]).
    FlatPattern {
        feature: crate::ids::FeatureId,
        #[serde(default = "yes", skip_serializing_if = "is_yes")]
        bend_notes: bool,
        #[serde(default = "yes", skip_serializing_if = "is_yes")]
        bend_table: bool,
    },
}

fn yes() -> bool {
    true
}

fn is_yes(b: &bool) -> bool {
    *b
}

impl ViewKind {
    pub fn parent(&self) -> Option<&str> {
        match self {
            ViewKind::Standard { .. } | ViewKind::FlatPattern { .. } => None,
            ViewKind::Projected { from, .. }
            | ViewKind::Section { from, .. }
            | ViewKind::Detail { from, .. }
            | ViewKind::Auxiliary { from, .. }
            | ViewKind::BrokenOut { from, .. }
            | ViewKind::Break { from, .. }
            | ViewKind::Crop { from, .. }
            | ViewKind::AlternatePosition { from, .. } => Some(from),
        }
    }

    pub fn references(&self) -> Vec<(&'static str, &Reference)> {
        match self {
            ViewKind::Standard { frame: Some(f), .. } => vec![("frame", f)],
            ViewKind::Section { plane, .. } => vec![("plane", plane)],
            ViewKind::Detail { center, .. } => vec![("center", center)],
            ViewKind::Auxiliary { edge, .. } => vec![("edge", edge)],
            ViewKind::Break { gaps, .. } => gaps
                .iter()
                .flat_map(|(a, b)| [("gap", a), ("gap", b)])
                .collect(),
            _ => Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Display {
    VisibleLines,
    HiddenLines,
    Shaded,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct View {
    pub name: String,
    pub sheet: String,
    pub model: ModelRef,
    pub kind: ViewKind,
    /// Sheet coordinates of the view's origin.
    pub at: [f64; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<(u32, u32)>,
    pub display: Display,
    /// Occurrences left out (fasteners in an assembly view).
    #[serde(default)]
    pub exclude: Vec<RelPath>,
    /// Show the assembly in this named exploded state (its occurrence transforms and explode
    /// lines), for assembly instructions and service views.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exploded: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DimensionStyle {
    Linear,
    Aligned,
    Angular,
    Radius,
    Diameter,
    /// From a zero edge: `references = [zero, feature]`.
    Ordinate,
    /// From the group's base.
    Baseline,
    /// End to end.
    Chain,
}

impl DimensionStyle {
    /// How many references the style takes: `(min, max)`.
    pub fn arity(self) -> (usize, usize) {
        match self {
            DimensionStyle::Radius | DimensionStyle::Diameter => (1, 1),
            DimensionStyle::Linear | DimensionStyle::Aligned => (1, 2),
            DimensionStyle::Angular => (2, 2),
            // The zero, then the feature.
            DimensionStyle::Ordinate => (2, 2),
            DimensionStyle::Baseline | DimensionStyle::Chain => (2, usize::MAX),
        }
    }
}

/// Where a linear dimension meets a cylinder (ASME Y14.5 and ISO 129-1 measure an overall
/// length over two bosses to their far sides): `center` the axis, `min` the side facing the
/// other reference, `max` the side away from it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArcCondition {
    #[default]
    Center,
    Min,
    Max,
}

impl ArcCondition {
    pub fn is_center(&self) -> bool {
        *self == ArcCondition::Center
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "annotation")]
pub enum AnnotationKind {
    Dimension {
        style: DimensionStyle,
        references: Vec<Reference>,
        /// A size tolerance from the part's PMI by name, inherited.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pmi: Option<String>,
        /// The ordinate origin or baseline base for group styles.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        group: Option<String>,
        /// Reference (driven) dimension, shown in parentheses.
        #[serde(default)]
        reference: bool,
        /// Where a linear dimension meets a cylindrical reference: its axis (the default), or
        /// the near or far side of its outline, as an overall length over two bosses measures.
        #[serde(default, skip_serializing_if = "ArcCondition::is_center")]
        condition: ArcCondition,
    },
    HoleCallout {
        hole: Reference,
    },
    /// A GD&T frame for a tolerance in the part's PMI.
    Gdt {
        tolerance: String,
    },
    DatumSymbol {
        label: String,
    },
    DatumTarget {
        label: String,
    },
    SurfaceFinish {
        on: Reference,
    },
    Weld {
        on: Reference,
        symbol: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        size: Option<ValueExpr>,
        #[serde(default)]
        all_around: bool,
        #[serde(default)]
        field: bool,
    },
    /// Text with `{property}` fields.
    Note {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        leader: Option<Reference>,
    },
    /// A BOM balloon for an occurrence, numbered by the named BOM table.
    Balloon {
        occurrence: RelPath,
        table: String,
    },
    CenterMark {
        on: Reference,
    },
    Centerline {
        between: Vec<Reference>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Annotation {
    /// Stable across edits and merges.
    pub id: String,
    pub view: String,
    pub kind: AnnotationKind,
    #[serde(default)]
    pub at: [f64; 2],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "table")]
pub enum TableKind {
    Bom {
        view: String,
        #[serde(default)]
        include_fasteners: bool,
        #[serde(default)]
        spec: BomTableSpec,
    },
    Hole {
        view: String,
        origin: Reference,
        #[serde(default)]
        spec: HoleTableSpec,
    },
    Revision {
        #[serde(default)]
        rows: Vec<revision_table::RevisionRow>,
    },
    /// The drawing's general notes, numbered in order and laid out above the title block
    /// (ISO 7200): `table notes = notes(rows = ["Dimensions in millimeters.", …], sheet = "1")`.
    Notes {
        #[serde(default)]
        rows: Vec<String>,
    },
    Bend {
        view: String,
    },
    Weld {
        view: String,
    },
    CutList {
        view: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DrawingTable {
    pub name: String,
    pub sheet: String,
    pub kind: TableKind,
    pub at: [f64; 2],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DrawingDocument {
    pub path: FilePath,
    #[serde(default)]
    pub header: VersionHeader,
    pub formats: Vec<SheetFormat>,
    pub sheets: Vec<Sheet>,
    pub views: Vec<View>,
    pub annotations: Vec<Annotation>,
    pub tables: Vec<DrawingTable>,
}

impl DrawingDocument {
    pub fn view(&self, name: &str) -> Option<&View> {
        self.views.iter().find(|v| v.name == name)
    }
}

// ------------------------------------------------------------------------------ seams

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectedEdge {
    /// The model entity (an edge's two face names, or a silhouette's face).
    pub entity: Vec<EntityName>,
    pub visible: bool,
    pub polyline: Vec<[f64; 2]>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ViewGeometry {
    pub edges: Vec<ProjectedEdge>,
    /// The cache key the geometry was made under.
    pub key: String,
}

pub trait HiddenLineRemover {
    fn project(&self, view: &View, model_key: &str) -> Result<ViewGeometry, KernelError>;
}

/// Resolves references against the model a view shows, and reads what the drawing needs.
pub trait ModelReferences {
    /// The entities a reference names in the model, or why it does not resolve.
    fn resolve(
        &self,
        model: &ModelRef,
        reference: &Reference,
    ) -> Result<Vec<Vec<EntityName>>, String>;
    /// A property of the drawn model, as text.
    fn property(&self, model: &ModelRef, name: &PropertyName) -> Option<String>;
    /// Whether an occurrence exists in the model (assembly views).
    fn occurrence(&self, model: &ModelRef, path: &RelPath) -> bool;
    /// PMI names the model declares: (tolerances, sizes and locations, datums).
    fn pmi(&self, model: &ModelRef) -> (BTreeSet<String>, BTreeSet<String>, BTreeSet<String>);
    /// The geometric characteristic of the model's PMI tolerance `name`, where the model
    /// knows it (for the drawing standard's check of the frames a sheet may show).
    fn tolerance_characteristic(
        &self,
        _model: &ModelRef,
        _name: &str,
    ) -> Option<crate::pmi::Characteristic> {
        None
    }
    /// The configuration-aware BOM of the model.
    fn bom(&self, model: &ModelRef) -> Option<Bom>;
    /// The exploded views the model declares (an assembly's; a part has none).
    fn exploded_views(&self, _model: &ModelRef) -> BTreeSet<String> {
        BTreeSet::new()
    }
    /// The developed layout a flat pattern or unfold feature of the model made (a part's;
    /// `None` when the feature made none).
    fn flat_layout(
        &self,
        _model: &ModelRef,
        _feature: &crate::ids::FeatureId,
    ) -> Option<crate::model::sheet_metal::FlatLayout> {
        None
    }
    /// The cut lists of the model's structural-member features, each with its feature id (a
    /// weldment part's; `None` when the model has none).
    fn cut_lists(
        &self,
        _model: &ModelRef,
    ) -> Option<Vec<(String, Vec<crate::model::weldment::CutListRow>)>> {
        None
    }
    /// The indented BOM of the model, for indented BOM tables.
    fn indented_bom(&self, _model: &ModelRef) -> Option<Vec<IndentedRow>> {
        None
    }
    /// A property of one BOM item's definition (description, finish, vendor …).
    fn item_property(
        &self,
        _model: &ModelRef,
        _item: &BomItem,
        _name: &PropertyName,
    ) -> Option<String> {
        None
    }
}

struct ItemProps<'a> {
    model: &'a dyn ModelReferences,
    at: &'a ModelRef,
}

impl BomProperties for ItemProps<'_> {
    fn property(&self, item: &BomItem, name: &PropertyName) -> Option<String> {
        self.model.item_property(self.at, item, name)
    }
}

// ------------------------------------------------------------------------------ checking

/// BOM item numbers by item order, for a table with no numbering of its own. Drawing tables
/// number through [`bom_table::render_bom_table`], which balloons follow.
pub fn item_numbers(bom: &Bom) -> BTreeMap<BomItem, u32> {
    bom.rows
        .iter()
        .enumerate()
        .map(|(i, r)| (r.item.clone(), i as u32 + 1))
        .collect()
}

#[derive(Clone, Debug, Default)]
pub struct DrawingReport {
    pub diagnostics: DiagnosticReport,
    /// Annotation id → resolved entities (dangling annotations are absent).
    pub resolved: BTreeMap<String, Vec<Vec<EntityName>>>,
    /// Title-block text per sheet.
    pub title_blocks: BTreeMap<String, Vec<(String, String)>>,
    /// Balloon id → item number as shown (`3`, or `2.1` from an indented table).
    pub balloons: BTreeMap<String, String>,
    /// Rendering data per BOM table.
    pub bom_tables: BTreeMap<String, BomTableData>,
    /// Rows per hole table (when hole geometry is available).
    pub hole_tables: BTreeMap<String, HoleTableData>,
    /// Rows per weldment cut-list table.
    pub cut_list_tables: BTreeMap<String, cut_list_table::CutListTableData>,
}

fn dsite(subject: &str, param: &str) -> Site {
    Site::with_param(
        Subject::Value {
            name: subject.into(),
        },
        param,
    )
}

pub fn check_drawing(doc: &DrawingDocument, model: &dyn ModelReferences) -> DrawingReport {
    check_drawing_with(doc, model, None)
}

/// [`check_drawing`], also generating hole-table rows through the geometry seam.
pub fn check_drawing_with(
    doc: &DrawingDocument,
    model: &dyn ModelReferences,
    holes: Option<&dyn HoleGeometry>,
) -> DrawingReport {
    let mut rep = DrawingReport::default();
    let err = |rep: &mut DrawingReport, code: DiagCode, s: Site, m: String| {
        rep.diagnostics
            .push(code, Severity::Error, s, m, Cause::Root, Vec::new());
    };
    let formats: BTreeSet<&str> = doc.formats.iter().map(|f| f.name.as_str()).collect();
    let sheets: BTreeSet<&str> = doc.sheets.iter().map(|s| s.name.as_str()).collect();
    for s in &doc.sheets {
        if !formats.contains(s.format.as_str()) {
            err(
                &mut rep,
                DiagCode::DrawingInvalid,
                dsite(&format!("sheet.{}", s.name), "format"),
                format!("sheet `{}` uses an unknown format `{}`", s.name, s.format),
            );
        }
        if s.scale.0 == 0 || s.scale.1 == 0 {
            err(
                &mut rep,
                DiagCode::DrawingInvalid,
                dsite(&format!("sheet.{}", s.name), "scale"),
                format!("sheet `{}` has a zero scale", s.name),
            );
        }
    }
    // Views: parents exist, no cycles, references resolve.
    let names: BTreeSet<&str> = doc.views.iter().map(|v| v.name.as_str()).collect();
    for v in &doc.views {
        let s = |p: &str| dsite(&format!("view.{}", v.name), p);
        if !sheets.contains(v.sheet.as_str()) {
            err(
                &mut rep,
                DiagCode::DrawingView,
                s("sheet"),
                format!("view `{}` is on an unknown sheet `{}`", v.name, v.sheet),
            );
        }
        let mut seen = BTreeSet::new();
        let mut cur = v.kind.parent();
        while let Some(p) = cur {
            if !names.contains(p) {
                err(
                    &mut rep,
                    DiagCode::DrawingView,
                    s("from"),
                    format!("view `{}` derives from an unknown view `{}`", v.name, p),
                );
                break;
            }
            if !seen.insert(p) || p == v.name {
                err(
                    &mut rep,
                    DiagCode::DrawingView,
                    s("from"),
                    format!("view `{}` derives from itself", v.name),
                );
                break;
            }
            cur = doc.view(p).and_then(|x| x.kind.parent());
        }
        if let Some(parent) = v.kind.parent().and_then(|p| doc.view(p)) {
            if parent.model.file != v.model.file
                && !matches!(v.kind, ViewKind::AlternatePosition { .. })
            {
                err(
                    &mut rep,
                    DiagCode::DrawingView,
                    s("model"),
                    format!(
                        "view `{}` shows `{}` but derives from a view of `{}`",
                        v.name, v.model.file, parent.model.file
                    ),
                );
            }
        }
        for (param, r) in v.kind.references() {
            if let Err(m) = model.resolve(&v.model, r) {
                err(
                    &mut rep,
                    DiagCode::DrawingReference,
                    s(param),
                    format!("view `{}` {}: `{}` {}", v.name, param, r, m),
                );
            }
        }
        if let ViewKind::FlatPattern { feature, .. } = &v.kind {
            if model.flat_layout(&v.model, feature).is_none() {
                err(
                    &mut rep,
                    DiagCode::DrawingReference,
                    s("feature"),
                    format!(
                        "view `{}` shows the flat pattern of `{}`, which made none in `{}`",
                        v.name, feature, v.model.file
                    ),
                );
            }
        }
        if let Some(e) = &v.exploded {
            if !model.exploded_views(&v.model).contains(e) {
                err(
                    &mut rep,
                    DiagCode::DrawingReference,
                    s("exploded"),
                    format!(
                        "view `{}` shows exploded view `{}`, which `{}` does not declare",
                        v.name, e, v.model.file
                    ),
                );
            }
        }
        for x in &v.exclude {
            if !model.occurrence(&v.model, x) {
                rep.diagnostics.push(
                    DiagCode::DrawingReference,
                    Severity::Warning,
                    s("exclude"),
                    format!(
                        "view `{}` excludes `{}`, which is not in the model",
                        v.name, x
                    ),
                    Cause::Root,
                    Vec::new(),
                );
            }
        }
    }
    // Tables.
    let mut boms: BTreeMap<&str, BTreeMap<String, String>> = BTreeMap::new();
    for t in &doc.tables {
        let s = |p: &str| dsite(&format!("table.{}", t.name), p);
        if !sheets.contains(t.sheet.as_str()) {
            err(
                &mut rep,
                DiagCode::DrawingInvalid,
                s("sheet"),
                format!("table `{}` is on an unknown sheet", t.name),
            );
        }
        let view_of = match &t.kind {
            TableKind::Bom { view, .. }
            | TableKind::Hole { view, .. }
            | TableKind::Bend { view }
            | TableKind::Weld { view }
            | TableKind::CutList { view } => Some(view),
            TableKind::Revision { .. } | TableKind::Notes { .. } => None,
        };
        let Some(vn) = view_of else { continue };
        let Some(v) = doc.view(vn) else {
            err(
                &mut rep,
                DiagCode::DrawingView,
                s("view"),
                format!("table `{}` reads an unknown view `{}`", t.name, vn),
            );
            continue;
        };
        match &t.kind {
            TableKind::Bom {
                include_fasteners,
                spec,
                ..
            } => {
                let mut spec = spec.clone();
                if !include_fasteners && !spec.exclude_kinds.contains(&BomKind::Fastener) {
                    spec.exclude_kinds.push(BomKind::Fastener);
                }
                let props = ItemProps {
                    model,
                    at: &v.model,
                };
                let data = if spec.structure == BomStructure::PartsOnly {
                    model
                        .bom(&v.model)
                        .map(|b| render_bom_table(&t.name, &spec, BomSource::Flat(&b), &props))
                } else {
                    model.indented_bom(&v.model).map(|rows| {
                        render_bom_table(&t.name, &spec, BomSource::Indented(&rows), &props)
                    })
                };
                match data {
                    Some(d) => {
                        rep.diagnostics.append(&d.diagnostics);
                        boms.insert(t.name.as_str(), d.numbers.clone());
                        rep.bom_tables.insert(t.name.clone(), d);
                    }
                    None => err(
                        &mut rep,
                        DiagCode::DrawingReference,
                        s("view"),
                        format!(
                            "table `{}`: `{}` has no {}BOM",
                            t.name,
                            v.model.file,
                            if spec.structure == BomStructure::PartsOnly {
                                ""
                            } else {
                                "indented "
                            }
                        ),
                    ),
                }
            }
            TableKind::Hole { origin, spec, .. } => {
                if let Err(m) = model.resolve(&v.model, origin) {
                    err(
                        &mut rep,
                        DiagCode::DrawingReference,
                        s("origin"),
                        format!("hole table `{}` origin `{}` {}", t.name, origin, m),
                    );
                } else if let Some(g) = holes {
                    let data = hole_table(&t.name, spec, v, origin, g);
                    rep.diagnostics.append(&data.diagnostics);
                    rep.hole_tables.insert(t.name.clone(), data);
                }
            }
            TableKind::CutList { .. } => {
                match model.cut_lists(&v.model).filter(|l| !l.is_empty()) {
                    Some(lists) => {
                        rep.cut_list_tables.insert(
                            t.name.clone(),
                            cut_list_table::cut_list_table(&t.name, &lists),
                        );
                    }
                    None => err(
                        &mut rep,
                        DiagCode::DrawingReference,
                        s("view"),
                        format!(
                            "table `{}`: `{}` has no structural members to list",
                            t.name, v.model.file
                        ),
                    ),
                }
            }
            _ => {}
        }
    }
    // Annotations.
    let mut ids = BTreeSet::new();
    for a in &doc.annotations {
        let s = |p: &str| dsite(&format!("annotation.{}", a.id), p);
        if !ids.insert(a.id.as_str()) {
            err(
                &mut rep,
                DiagCode::DrawingInvalid,
                s("id"),
                format!("annotation id `{}` appears twice", a.id),
            );
        }
        let Some(v) = doc.view(&a.view) else {
            err(
                &mut rep,
                DiagCode::DrawingView,
                s("view"),
                format!("annotation `{}` is on an unknown view `{}`", a.id, a.view),
            );
            continue;
        };
        let (tols, sizes, datums) = model.pmi(&v.model);
        let mut refs: Vec<(&str, &Reference)> = Vec::new();
        match &a.kind {
            AnnotationKind::Dimension {
                style,
                references,
                pmi,
                condition,
                ..
            } => {
                if !condition.is_center()
                    && !matches!(style, DimensionStyle::Linear | DimensionStyle::Aligned)
                {
                    err(
                        &mut rep,
                        DiagCode::DrawingInvalid,
                        s("condition"),
                        format!(
                            "dimension `{}`: an arc condition applies to linear and aligned dimensions, not {:?}",
                            a.id, style
                        ),
                    );
                }
                let (lo, hi) = style.arity();
                if references.len() < lo || references.len() > hi {
                    err(
                        &mut rep,
                        DiagCode::DrawingInvalid,
                        s("references"),
                        format!(
                            "a {:?} dimension takes {}..{} references, `{}` has {}",
                            style,
                            lo,
                            if hi == usize::MAX {
                                "n".into()
                            } else {
                                hi.to_string()
                            },
                            a.id,
                            references.len()
                        ),
                    );
                }
                if let Some(p) = pmi {
                    if !sizes.contains(p) {
                        err(&mut rep, DiagCode::DrawingReference, s("pmi"), format!("dimension `{}` inherits tolerance `{}`, a size or location the model does not declare", a.id, p));
                    }
                }
                refs.extend(references.iter().map(|r| ("references", r)));
            }
            AnnotationKind::HoleCallout { hole } => refs.push(("hole", hole)),
            AnnotationKind::Gdt { tolerance } => {
                // The frame's sheet follows its format's standard.
                let standard = doc
                    .sheets
                    .iter()
                    .find(|sheet| sheet.name == v.sheet)
                    .and_then(|sheet| doc.formats.iter().find(|f| f.name == sheet.format))
                    .map(|f| f.standard)
                    .unwrap_or_default();
                if let Some(refusal) = model
                    .tolerance_characteristic(&v.model, tolerance)
                    .and_then(|c| standard.refusal(c))
                {
                    err(
                        &mut rep,
                        DiagCode::DrawingInvalid,
                        s("tolerance"),
                        format!("GD&T frame `{}` shows `{tolerance}`: {refusal}", a.id),
                    );
                }
                if !tols.contains(tolerance) {
                    err(
                        &mut rep,
                        DiagCode::DrawingReference,
                        s("tolerance"),
                        format!(
                            "GD&T frame `{}` shows `{}`, which the model's PMI does not declare",
                            a.id, tolerance
                        ),
                    );
                }
            }
            AnnotationKind::DatumSymbol { label } | AnnotationKind::DatumTarget { label } => {
                if !datums.contains(label) {
                    err(
                        &mut rep,
                        DiagCode::DrawingReference,
                        s("label"),
                        format!("datum `{}` is not declared in the model's PMI", label),
                    );
                }
            }
            AnnotationKind::SurfaceFinish { on }
            | AnnotationKind::Weld { on, .. }
            | AnnotationKind::CenterMark { on } => refs.push(("on", on)),
            AnnotationKind::Note { text, leader } => {
                if let Some(l) = leader {
                    refs.push(("leader", l));
                }
                let mut rest = text.as_str();
                while let Some(i) = rest.find('{') {
                    let Some(j) = rest[i..].find('}') else { break };
                    let field = &rest[i + 1..i + j];
                    if note_field(model, &v.model, &doc.tables, field).is_none() {
                        rep.diagnostics.push(
                            DiagCode::DrawingReference,
                            Severity::Warning,
                            s("text"),
                            format!(
                                "note `{}` shows `{{{}}}`, which the model does not have",
                                a.id, field
                            ),
                            Cause::Root,
                            Vec::new(),
                        );
                    }
                    rest = &rest[i + j + 1..];
                }
            }
            AnnotationKind::Balloon { occurrence, table } => {
                if !model.occurrence(&v.model, occurrence) {
                    err(
                        &mut rep,
                        DiagCode::DrawingReference,
                        s("occurrence"),
                        format!(
                            "balloon `{}` points at `{}`, which is not in the model",
                            a.id, occurrence
                        ),
                    );
                } else if let Some(numbers) = boms.get(table.as_str()) {
                    let bom = model.bom(&v.model).unwrap_or_default();
                    let wanted = occurrence.to_string();
                    match bom
                        .rows
                        .iter()
                        .find(|r| r.paths.contains(&wanted))
                        .and_then(|r| numbers.get(&bom_item_key(&r.item)))
                    {
                        Some(n) => {
                            rep.balloons.insert(a.id.clone(), n.clone());
                        }
                        None => err(
                            &mut rep,
                            DiagCode::DrawingReference,
                            s("occurrence"),
                            format!("balloon `{}`: `{}` has no BOM line", a.id, occurrence),
                        ),
                    }
                } else {
                    err(
                        &mut rep,
                        DiagCode::DrawingReference,
                        s("table"),
                        format!(
                            "balloon `{}` numbers by an unknown BOM table `{}`",
                            a.id, table
                        ),
                    );
                }
            }
            AnnotationKind::Centerline { between } => {
                refs.extend(between.iter().map(|r| ("between", r)))
            }
        }
        let mut all = Vec::new();
        let mut dangling = false;
        for (param, r) in refs {
            match model.resolve(&v.model, r) {
                Ok(e) => all.extend(e),
                Err(m) => {
                    dangling = true;
                    err(
                        &mut rep,
                        DiagCode::DrawingReference,
                        s(param),
                        format!("annotation `{}` on view `{}`: `{}` {}", a.id, a.view, r, m),
                    );
                }
            }
        }
        if !dangling {
            rep.resolved.insert(a.id.clone(), all);
        }
    }
    // Title blocks.
    let count = doc.sheets.len();
    for (i, sh) in doc.sheets.iter().enumerate() {
        let Some(f) = doc.formats.iter().find(|f| f.name == sh.format) else {
            continue;
        };
        let model_of = doc
            .views
            .iter()
            .find(|v| v.sheet == sh.name)
            .map(|v| &v.model);
        let mut rows = Vec::new();
        // The standard fields read what the model has and leave the rest blank.
        let standard_fields = f.title_block.is_empty();
        for field in &f.title_fields(&drawing_title(doc.path.as_str())) {
            let text = match &field.source {
                FieldSource::Literal { text } => text.clone(),
                FieldSource::SheetNumber => (i + 1).to_string(),
                FieldSource::SheetCount => count.to_string(),
                FieldSource::Scale => format!("{}:{}", sh.scale.0, sh.scale.1),
                FieldSource::Image { .. } => String::new(),
                FieldSource::Revision => model_of
                    .and_then(|m| model.property(m, &PropertyName::new("revision")))
                    .or_else(|| revision_table::latest(&doc.tables).map(str::to_owned))
                    .unwrap_or_else(|| "—".into()),
                FieldSource::Property { name } => {
                    match model_of.and_then(|m| model.property(m, name)) {
                        Some(t) => t,
                        None if standard_fields => "—".into(),
                        None => {
                            rep.diagnostics.push(DiagCode::DrawingReference, Severity::Warning, dsite(&format!("sheet.{}", sh.name), &field.label), format!("title block `{}` reads `{}`, which the drawn model does not have", field.label, name), Cause::Root, Vec::new());
                            "—".into()
                        }
                    }
                }
            };
            rows.push((field.label.clone(), text));
        }
        rep.title_blocks.insert(sh.name.clone(), rows);
    }
    rep
}

/// A note's `{field}`: the drawn model's property, and for `{revision}` with none the drawing's
/// latest revision (as the title block's revision field reads it).
pub(crate) fn note_field(
    model: &dyn ModelReferences,
    at: &ModelRef,
    tables: &[DrawingTable],
    field: &str,
) -> Option<String> {
    model.property(at, &PropertyName::new(field)).or_else(|| {
        (field == "revision")
            .then(|| revision_table::latest(tables).map(str::to_owned))
            .flatten()
    })
}
