#[path = "support/mod.rs"]
mod test_helpers;

use runmat_value::{IntegerStorage, Value};
use test_helpers::execute_source;

fn source_path(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}

struct TestDir(std::path::PathBuf);

impl TestDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "runmat_vm_data_integer_lifecycle_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create test directory");
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn compiled_data_lifecycle_preserves_wide_uint64_storage() {
    let dir = TestDir::new();
    let source = source_path(&dir.0.join("source.data"));
    let copied = source_path(&dir.0.join("copy.data"));
    let exported = source_path(&dir.0.join("export.data"));
    let imported = source_path(&dir.0.join("import.data"));
    let moved = source_path(&dir.0.join("moved.data"));
    let script = format!(
        "meta = struct(\"dtype\", \"uint64\", \"shape\", uint64([1 2]), \"chunk\", uint8([1 2])); \
         arrays = struct(\"samples\", meta); \
         schema = struct(\"arrays\", arrays); \
         ds = feval(\"data.create\", \"{source}\", schema); \
         arr = feval(\"Dataset.array\", ds, \"samples\"); \
         feval(\"DataArray.write\", arr, uint64([9223372036854775808 18446744073709551615])); \
         feval(\"data.copy\", \"{source}\", \"{copied}\"); \
         feval(\"data.export\", \"{copied}\", \"data\", \"{exported}\"); \
         feval(\"data.import\", \"{imported}\", \"data\", \"{exported}\"); \
         feval(\"data.move\", \"{imported}\", \"{moved}\"); \
         if ~feval(\"data.exists\", \"{moved}\"); error('moved dataset missing'); end; \
         info = feval(\"data.inspect\", \"{moved}\"); \
         if info.arrayCount ~= 1; error('wrong array count'); end; \
         moved_ds = feval(\"data.open\", \"{moved}\"); \
         moved_arr = feval(\"Dataset.array\", moved_ds, \"samples\"); \
         out = feval(\"DataArray.read\", moved_arr); \
         feval(\"data.delete\", \"{moved}\"); \
         if feval(\"data.exists\", \"{moved}\"); error('dataset delete failed'); end;"
    );

    let vars = execute_source(&script).expect("execute compiled data lifecycle");
    let expected = IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]);
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.integer_storage() == Some(&expected))
    }));
}
