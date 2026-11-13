use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn writecell_honours_delimiter_quote_strings_and_append() {
    let _lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("txt");
    let filename = path.to_string_lossy().into_owned();
    let first = cell(vec![Value::from("a,b"), Value::Num(2.0)], 1, 2);
    let second = cell(vec![Value::from("tail"), Value::Num(3.0)], 1, 2);

    block_on(writecell_builtin(
        first,
        vec![
            Value::from(filename.clone()),
            Value::from("Delimiter"),
            Value::from("|"),
            Value::from("QuoteStrings"),
            Value::Bool(false),
        ],
    ))
    .expect("initial write");
    block_on(writecell_builtin(
        second,
        vec![
            Value::from(filename.clone()),
            Value::from("Delimiter"),
            Value::from("|"),
            Value::from("QuoteStrings"),
            Value::Bool(false),
            Value::from("WriteMode"),
            Value::from("append"),
        ],
    ))
    .expect("append write");

    let contents = fs::read_to_string(&path).expect("read contents");
    assert_eq!(contents, "a,b|2\ntail|3\n");
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn writecell_append_inserts_missing_row_boundary() {
    let _lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("txt");
    fs::write(&path, "existing").expect("seed");
    let filename = path.to_string_lossy().into_owned();
    let values = cell(vec![Value::from("tail"), Value::Num(3.0)], 1, 2);

    block_on(writecell_builtin(
        values,
        vec![
            Value::from(filename),
            Value::from("Delimiter"),
            Value::from("|"),
            Value::from("QuoteStrings"),
            Value::Bool(false),
            Value::from("WriteMode"),
            Value::from("append"),
        ],
    ))
    .expect("append write");

    let contents = fs::read_to_string(&path).expect("read contents");
    assert_eq!(contents, "existing\ntail|3\n");
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn writecell_append_quotes_across_bounded_chunks_on_selected_provider() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/tabular",
    ));
    provider
        .write_project_path("/tabular/long.csv", b"head")
        .expect("fixture");
    let _guard = runmat_filesystem::replace_provider(provider.clone());
    runmat_filesystem::set_current_dir("/tabular").expect("cwd");
    let text = format!("{}\"{}", "a".repeat(4095), "b".repeat(4097));
    block_on(writecell_builtin(
        cell(vec![Value::from(text.clone())], 1, 1),
        vec![
            Value::from("long.csv"),
            Value::from("WriteMode"),
            Value::from("append"),
        ],
    ))
    .expect("append");
    let actual = provider
        .read_project_path("/tabular/long.csv")
        .expect("written bytes");
    let expected = format!("head\n\"{}\"\n", text.replace('"', "\"\""));
    assert_eq!(actual, expected.as_bytes());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn writecell_concurrent_appends_share_one_boundary_insertion() {
    let _lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("txt");
    fs::write(&path, "existing").expect("seed");
    let filename = path.to_string_lossy().into_owned();
    let writers = 8usize;
    let barrier = Arc::new(Barrier::new(writers));
    let mut handles = Vec::new();
    for idx in 0..writers {
        let barrier = Arc::clone(&barrier);
        let filename = filename.clone();
        handles.push(thread::spawn(move || {
            barrier.wait();
            let values = cell(
                vec![Value::from(format!("row{idx}")), Value::Num(idx as f64)],
                1,
                2,
            );
            block_on(writecell_builtin(
                values,
                vec![
                    Value::from(filename),
                    Value::from("Delimiter"),
                    Value::from("|"),
                    Value::from("QuoteStrings"),
                    Value::Bool(false),
                    Value::from("WriteMode"),
                    Value::from("append"),
                ],
            ))
            .expect("append write");
        }));
    }
    for handle in handles {
        handle.join().expect("writer thread");
    }

    let contents = fs::read_to_string(&path).expect("read contents");
    let lines = contents.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), writers + 1);
    assert_eq!(lines[0], "existing");
    assert!(lines.iter().all(|line| !line.is_empty()));
    for idx in 0..writers {
        let expected = format!("row{idx}|{idx}");
        assert!(lines.iter().any(|line| *line == expected));
    }
    let _ = fs::remove_file(path);
}
