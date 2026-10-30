use super::*;

fn output_cell(value: &Value) -> &CellArray {
    let Value::Cell(cell) = value else {
        panic!("expected cell array output");
    };
    cell
}

fn output_value(value: &Value, col: usize) -> Value {
    output_cell(value).get(0, col).expect("output cell")
}

fn numeric_column(value: &Value, col: usize) -> Vec<f64> {
    let Value::Tensor(tensor) = output_value(value, col) else {
        panic!("expected tensor");
    };
    tensor.materialize_f64()
}

fn text_column(value: &Value, col: usize) -> Vec<String> {
    let Value::Cell(cell) = output_value(value, col) else {
        panic!("expected text cell column");
    };
    let mut out = Vec::new();
    for row in 0..cell.rows {
        let Value::String(runmat_value::StringElement::Text(text)) =
            cell.get(row, 0).expect("text cell")
        else {
            panic!("expected string");
        };
        out.push(text.to_string());
    }
    out
}

#[path = "tests/admission.rs"]
mod admission;
#[path = "tests/file.rs"]
mod file;
#[path = "tests/parsing.rs"]
mod parsing;
