//! Exercise the published runtime library, not a second unit-test copy of it.
use runmat_builtins::*;

#[test]
fn migrated_filesystem_and_host_calls_have_one_canonical_binding() {
    let entries = [
        &ADDPATH_CATALOG_ENTRY,
        &COPYFILE_CATALOG_ENTRY,
        &DIR_CATALOG_ENTRY,
        &EXIST_CATALOG_ENTRY,
        &FILEPARTS_CATALOG_ENTRY,
        &FILESEP_CATALOG_ENTRY,
        &FULLFILE_CATALOG_ENTRY,
        &GENPATH_CATALOG_ENTRY,
        &GETENV_CATALOG_ENTRY,
        &ISENV_CATALOG_ENTRY,
        &ISFILE_CATALOG_ENTRY,
        &ISFOLDER_CATALOG_ENTRY,
        &LS_CATALOG_ENTRY,
        &MATLABROOT_CATALOG_ENTRY,
        &MKDIR_CATALOG_ENTRY,
        &MOVEFILE_CATALOG_ENTRY,
        &OPEN_CATALOG_ENTRY,
        &OPENTOLINE_CATALOG_ENTRY,
        &PATHSEP_CATALOG_ENTRY,
        &PCODE_CATALOG_ENTRY,
        &RMDIR_CATALOG_ENTRY,
        &RMPATH_CATALOG_ENTRY,
        &RUN_CATALOG_ENTRY,
        &SAVEPATH_CATALOG_ENTRY,
        &SETENV_CATALOG_ENTRY,
        &TEMPDIR_CATALOG_ENTRY,
        &TEMPNAME_CATALOG_ENTRY,
        &UIGETDIR_CATALOG_ENTRY,
        &UIGETFILE_CATALOG_ENTRY,
        &UIPUTFILE_CATALOG_ENTRY,
        &UNSETENV_CATALOG_ENTRY,
        &WHAT_CATALOG_ENTRY,
    ];
    for entry in entries {
        let name = entry.identity.name;
        let bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
        assert_eq!(bindings.len(), entry.bindings.len(), "{name}");
        for required in entry.bindings {
            assert!(
                bindings
                    .iter()
                    .any(|binding| binding.identity == entry.binding_identity(required)),
                "missing canonical binding for {name}"
            );
        }
        assert!(
            builtin_function_by_name(name).is_none(),
            "{name} still publishes a legacy registration"
        );
    }
}

#[test]
fn filesep_canonical_dispatch_returns_the_platform_separator() {
    let result =
        futures::executor::block_on(runmat_runtime::call_builtin_async("filesep", &[])).unwrap();
    assert_eq!(
        result,
        runmat_value::Value::CharArray(runmat_value::CharArray::new_row(
            std::path::MAIN_SEPARATOR_STR,
        ))
    );
}
