use std::convert::TryFrom;

use runmat_value::{CellArray, CharArray, StringArray, Value};
use tempfile::tempdir;

use super::super::super::path_mutation::test_support::{canonical, PathGuard};

#[test]
fn prepends_returns_previous_path_and_deduplicates() {
    let _guard = PathGuard::new();
    let first = tempdir().expect("first");
    let second = tempdir().expect("second");
    let first = canonical(first.path());
    let second = canonical(second.path());
    crate::builtins::common::path_state::set_path_string(&format!(
        "{first}{}{second}",
        crate::builtins::common::path_state::PATH_LIST_SEPARATOR
    ))
    .expect("test search path update");

    let returned = super::run(vec![Value::String(second.clone().into())]).expect("addpath");
    assert_eq!(
        String::try_from(&returned).expect("text"),
        format!(
            "{first}{}{second}",
            crate::builtins::common::path_state::PATH_LIST_SEPARATOR
        )
    );
    assert_eq!(
        crate::builtins::common::path_state::current_path_segments(),
        vec![second, first]
    );
}

#[test]
fn appends_text_containers_and_character_rows() {
    let _guard = PathGuard::new();
    let first = tempdir().expect("first");
    let second = tempdir().expect("second");
    let first_text = first.path().to_string_lossy().into_owned();
    let second_text = second.path().to_string_lossy().into_owned();
    let strings = StringArray::new(vec![first_text], vec![1, 1]).expect("strings");
    let cells = CellArray::new(
        vec![Value::CharArray(CharArray::new_row(&second_text))],
        1,
        1,
    )
    .expect("cell");
    crate::builtins::common::path_state::set_path_string("").expect("test search path update");
    super::run(vec![
        Value::StringArray(strings),
        Value::Cell(cells),
        Value::String("-end".into()),
    ])
    .expect("addpath");
    assert_eq!(
        crate::builtins::common::path_state::current_path_segments(),
        vec![canonical(first.path()), canonical(second.path())]
    );
}

#[test]
fn asymmetric_cell_paths_follow_column_major_order() {
    let _guard = PathGuard::new();
    let directories = (0..4).map(|_| tempdir().unwrap()).collect::<Vec<_>>();
    let cells = CellArray::new(
        directories
            .iter()
            .map(|dir| Value::String(dir.path().to_string_lossy().into_owned().into()))
            .collect(),
        2,
        2,
    )
    .unwrap();
    crate::builtins::common::path_state::set_path_string("").expect("test search path update");
    super::run(vec![Value::Cell(cells), Value::from("-end")]).unwrap();
    assert_eq!(
        crate::builtins::common::path_state::current_path_segments(),
        [0, 2, 1, 3].map(|index| canonical(directories[index].path()))
    );
}

#[test]
fn trailing_singleton_character_folder_is_admitted() {
    let _guard = PathGuard::new();
    let directory = tempdir().unwrap();
    let characters = directory
        .path()
        .to_string_lossy()
        .chars()
        .collect::<Vec<_>>();
    let shape = vec![
        1,
        directory.path().to_string_lossy().encode_utf16().count(),
        1,
    ];
    let value = CharArray::from_unicode_scalars_with_shape(characters, shape).unwrap();
    crate::builtins::common::path_state::set_path_string("").expect("test search path update");
    super::run(vec![Value::CharArray(value)]).unwrap();
    assert_eq!(
        crate::builtins::common::path_state::current_path_segments(),
        vec![canonical(directory.path())]
    );
}

#[test]
fn moving_the_user_entry_keeps_its_origin_for_later_setting_changes() {
    use crate::builtins::common::path_state::{PathOrigin, ResolvedUserPath, SearchPath};
    use std::{rc::Rc, sync::Arc};

    let _guard = PathGuard::new();
    let directory = tempdir().expect("user directory");
    let user = canonical(directory.path());
    let search = Arc::new(SearchPath::new(String::new()));
    search.install_captured_factory(
        ResolvedUserPath {
            presentation: user.clone(),
            search_segment: Some(user.clone()),
        },
        1,
        "environment",
        None,
    );
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_search_path(search.clone());
    let _active = crate::context::RuntimeContextGuard::enter(runtime);
    super::run(vec![
        Value::String(user.clone().into()),
        Value::String("-end".into()),
    ])
    .expect("move user entry");
    assert_eq!(
        search.snapshot().last().expect("moved entry").origin,
        PathOrigin::User
    );
    search.replace_user_if_newer(2, Some("new-user".into()));
    assert!(search.snapshot().iter().all(|entry| entry.path != user));
}
