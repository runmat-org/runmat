#[path = "support/mod.rs"]
mod test_helpers;

#[test]
fn compiled_environment_and_path_calls_preserve_shapes_and_effects() {
    runmat_gc::gc_test_context(|| {
        test_helpers::execute_source(
            r#"
names = {'RUNMAT_VM_PATH_ENV_0', 'RUNMAT_VM_PATH_ENV_2'; 'RUNMAT_VM_PATH_ENV_1', 'RUNMAT_VM_PATH_ENV_3'};
values = ["zero", "two"; "one", "three"];
setenv(names, values);
found = getenv(names);
present = isenv(names);
unsetenv(names);
absent = isenv(names);
assert(isequal(found, {'zero', 'two'; 'one', 'three'}));
assert(isequal(present, true(2, 2)));
assert(isequal(absent, false(2, 2)));
paths = fullfile({'a', 'c'; 'b', 'd'}, ["zero", "two"; "one", "three"]);
assert(strcmp(paths(1, 1), fullfile('a', 'zero')));
assert(strcmp(paths(2, 1), fullfile('b', 'one')));
assert(strcmp(paths(1, 2), fullfile('c', 'two')));
assert(strcmp(paths(2, 2), fullfile('d', 'three')));
separator = pathsep();
assert(ischar(separator));
assert(numel(separator) == 1);
folder = tempdir();
first = tempname(folder);
second = tempname(folder);
assert(ischar(first));
assert(startsWith(first, folder));
assert(~strcmp(first, second));
assert(exist(first, 'file') == 0);
assert(exist(first, 'dir') == 0);
"#,
        )
        .expect("compiled environment/path contracts");
    });
}
