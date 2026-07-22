#[path = "support/mod.rs"]
mod test_helpers;

use futures::executor::block_on;
use runmat_builtins::{NumericDType, Tensor, Value};
use test_helpers::execute_source;

fn unique_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "runmat_vm_matfile_{name}_{}.mat",
        std::process::id()
    ))
}

fn tensor(data: &[f64], shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor {
        data: data.to_vec(),
        integer_data: None,
        shape: shape.clone(),
        rows: *shape.first().unwrap_or(&1),
        cols: *shape.get(1).unwrap_or(&data.len()),
        dtype: NumericDType::F64,
    })
}

fn write_sample(path: &std::path::Path) {
    let bytes = block_on(
        runmat_runtime::builtins::io::mat::save::encode_workspace_to_mat_bytes(&[(
            "A".to_string(),
            tensor(&[1.0, 2.0, 3.0], vec![1, 3]),
        )]),
    )
    .expect("encode MAT");
    std::fs::write(path, bytes).expect("write MAT");
}

#[test]
fn matfile_dot_reads_properties_and_writes_whole_variables() {
    let path = unique_path("dot_round_trip");
    write_sample(&path);
    let source_path = path.to_string_lossy().replace('\'', "''");
    let input = format!(
        "\
        m = matfile('{source_path}'); \
        A = m.A; \
        props = m.Properties; \
        if A(2) ~= 2; error('matfile read mismatch'); end; \
        if ~strcmp(props.Source, '{source_path}'); error('matfile source mismatch'); end; \
        if props.Writable; error('matfile should default to read-only'); end; \
        mw = matfile('{source_path}', 'Writable', true); \
        mw.B = 42; \
        m2 = matfile('{source_path}'); \
        out = m2.B;"
    );
    let vars = execute_source(&input).expect("execute matfile source");
    assert!(vars.iter().any(|value| value == &Value::Num(42.0)));
    let _ = std::fs::remove_file(path);
}
