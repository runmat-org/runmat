//! A weldment's cut list as a drawing table: one row per group of identical members, with its
//! item number, quantity, profile, material, cut length and end angles, written in the drawing
//! as `table cuts = cut_list(view = front, sheet = "1", at = [10, 200])`. The rows come from
//! the drawn part's structural-member features, in the configuration the view shows.

use crate::drawing::bom_table::{material_name, Align, ColumnHeader};
use crate::model::weldment::CutListRow;
use serde::{Deserialize, Serialize};

/// The columns, left to right: key, title, and whether the cells right-align (numbers do).
pub const COLUMNS: [(&str, &str, bool); 7] = [
    ("item", "Item", true),
    ("quantity", "Qty", true),
    ("description", "Description", false),
    ("material", "Material", false),
    ("length", "Length", true),
    ("angle_1", "Angle 1", true),
    ("angle_2", "Angle 2", true),
];

fn columns() -> Vec<ColumnHeader> {
    COLUMNS
        .iter()
        .map(|(key, title, right)| ColumnHeader {
            key: (*key).into(),
            title: (*title).into(),
            align: if *right { Align::Right } else { Align::Left },
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CutListTableRow {
    /// The cells as the sheet shows them, one per column.
    pub cells: Vec<String>,
    /// The structural-member feature the row's members come from (`feature:<id>`).
    pub feature: String,
}

/// A cut-list table as the sheet and the exports draw it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CutListTableData {
    pub table: String,
    /// The header, in the BOM table's column form, so the sheet draws both alike.
    pub columns: Vec<ColumnHeader>,
    pub rows: Vec<CutListTableRow>,
    /// Members over every row.
    pub total_quantity: u32,
    /// Stock to order: every member's cut length added up (m).
    pub total_length: f64,
}

fn mm(m: f64) -> String {
    format!("{:.1}", m * 1000.0)
}

fn deg(rad: f64) -> String {
    format!("{:.1}°", rad.to_degrees())
}

/// The table `name` over the drawn part's cut lists, each with the feature that makes it.
/// Items keep their numbers within a feature and number on across features.
pub fn cut_list_table(name: &str, lists: &[(String, Vec<CutListRow>)]) -> CutListTableData {
    let mut rows = Vec::new();
    let mut total_quantity = 0;
    let mut total_length = 0.0;
    let mut item = 0;
    for (feature, list) in lists {
        for r in list {
            item += 1;
            total_quantity += r.quantity;
            total_length += r.length * f64::from(r.quantity);
            rows.push(CutListTableRow {
                cells: vec![
                    item.to_string(),
                    r.quantity.to_string(),
                    r.description.clone(),
                    r.material.as_deref().map(material_name).unwrap_or_else(|| "—".into()),
                    mm(r.length),
                    deg(r.start_angle),
                    deg(r.end_angle),
                ],
                feature: format!("feature:{feature}"),
            });
        }
    }
    CutListTableData {
        table: name.to_owned(),
        columns: columns(),
        rows,
        total_quantity,
        total_length,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(item: u32, quantity: u32, length: f64, angle: f64) -> CutListRow {
        CutListRow {
            item,
            quantity,
            description: "Tube 20 × 20 × 2".into(),
            material: Some("material.steel".into()),
            length,
            start_angle: angle,
            end_angle: angle,
            members: Vec::new(),
            mass_per_member: None,
        }
    }

    #[test]
    fn rows_number_on_across_features_and_total_the_stock() {
        let lists = vec![
            ("frame".to_owned(), vec![row(1, 2, 0.22, std::f64::consts::FRAC_PI_4), row(2, 2, 0.12, std::f64::consts::FRAC_PI_4)]),
            ("brace".to_owned(), vec![row(1, 1, 0.1, 0.0)]),
        ];
        let t = cut_list_table("cuts", &lists);
        assert_eq!(t.columns.iter().map(|c| c.title.as_str()).collect::<Vec<_>>(), ["Item", "Qty", "Description", "Material", "Length", "Angle 1", "Angle 2"]);
        assert_eq!(t.rows[0].cells, ["1", "2", "Tube 20 × 20 × 2", "Steel", "220.0", "45.0°", "45.0°"]);
        assert_eq!(t.rows[2].cells[0], "3");
        assert_eq!(t.rows[2].feature, "feature:brace");
        assert_eq!(t.total_quantity, 5);
        assert!((t.total_length - 0.78).abs() < 1e-12);
    }
}
