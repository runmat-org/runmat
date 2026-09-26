#[path = "support/mod.rs"]
mod test_helpers;

#[test]
fn compiled_filesystem_calls_preserve_outputs_shapes_and_effects() {
    runmat_gc::gc_test_context(|| {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().to_string_lossy().replace('\'', "''");
        std::fs::write(temporary.path().join("input.m"), b"value = 1;\n").unwrap();
        let source = format!(
            r#"
root = '{root}';
folder = fullfile(root, 'archive');
mkdir(folder);
assert(isfolder(folder));
[status, message, identifier] = mkdir(folder);
assert(status && ~isempty(message) && ~isempty(identifier));
source = fullfile(root, 'input.m');
target = fullfile(folder, 'renamed.m');
movefile(source, target);
assert(~isfile(source) && isfile(target));
paths = {{target, source; folder, target}};
assert(isequal(isfile(paths), [true, false; false, true]));
assert(isequal(isfolder(paths), [false, false; true, false]));
listing = ls(folder);
assert(ischar(listing) && ~isempty(listing));
inventory = what(folder);
assert(isequal(inventory.m, {{'renamed.m'}}));
assert(isequal(size(inventory.mat), [0 1]));
assert(ischar(matlabroot()));
rmdir(folder, 's');
assert(~isfolder(folder));
failed = false;
try
    movefile(source, target);
catch err
    failed = true;
end
assert(failed);
"#
        );
        test_helpers::execute_source(&source).expect("compiled filesystem contracts");
        assert!(!temporary.path().join("archive").exists());
    });
}
