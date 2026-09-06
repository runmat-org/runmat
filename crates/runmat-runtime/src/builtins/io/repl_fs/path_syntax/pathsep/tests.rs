use futures::executor::block_on;
use runmat_value::Value;
#[test]
fn returns_one_platform_character_and_rejects_inputs() {
    let result = block_on(super::pathsep_builtin(Vec::new())).unwrap();
    assert_eq!(
        String::try_from(&result).unwrap(),
        crate::builtins::common::path_state::PATH_LIST_SEPARATOR.to_string()
    );
    assert!(block_on(super::pathsep_builtin(vec![Value::Num(1.0)])).is_err());
}
