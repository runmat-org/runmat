//! A weldment drawing's cut-list table, from the drawn part's structural members.

use super::compile_json;
use crate::runtime::Files;
use serde_json::Value;

fn drawing(table: &str) -> String {
    format!("runmat 0.7\nstd 1\n\ndrawing frame {{\n    format a3 = {{ size = {{ size = iso, name = A3 }}, title_block = [] }}\n    sheet \"1\" = {{ format = a3, scale = [1, 1], projection = third_angle }}\n    view front = standard(orientation = top, sheet = \"1\", model = {{ file = \"frame.part\" }}, at = [150, 150], display = visible_lines)\n{table}}}\n")
}

#[test]
fn a_cut_list_table_lists_the_drawn_weldments_members() {
    let files: Files = [("/p/frame.part".to_owned(), crate::analyze::tests::frame())].into();
    let text = drawing("    table cuts = cut_list(view = front, sheet = \"1\", at = [20, 120])\n");
    let out: Value = serde_json::from_str(&compile_json("/p/frame.drawing", &text, &files)).unwrap();
    let table = &out["projection"]["data"]["cut_list_tables"]["cuts"];
    assert_eq!(table["rows"].as_array().map(Vec::len), Some(2), "{out}");
    assert_eq!(table["columns"][4]["title"], "Length");
    assert_eq!(table["columns"][4]["align"], "right");
    // The long sides: 200 mm between centre lines plus half the 20 mm tube at each mitre.
    assert_eq!(table["rows"][0]["cells"][4], "220.0", "{table}");
    assert_eq!(table["rows"][0]["cells"][5], "45.0°");
    assert_eq!(table["rows"][0]["feature"], "feature:members");
    assert_eq!(table["total_quantity"], 4);
    assert_eq!(out["diagnostics"].as_array().map(Vec::len), Some(0), "{out}");
}

#[test]
fn a_cut_list_on_a_part_without_members_says_so() {
    let bracket = include_str!("../../../tests/corpus/documents/bracket.part");
    let files: Files = [("/p/frame.part".to_owned(), bracket.to_owned())].into();
    let text = drawing("    table cuts = cut_list(view = front, sheet = \"1\", at = [0, 0])\n");
    let out: Value = serde_json::from_str(&compile_json("/p/frame.drawing", &text, &files)).unwrap();
    let messages: Vec<&str> = out["projection"]["data"]["diagnostics"].as_array().into_iter().flatten().filter_map(|d| d["message"].as_str()).collect();
    assert!(messages.iter().any(|m| m.contains("has no structural members")), "{out}");
}
