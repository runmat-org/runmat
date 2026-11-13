use futures::executor::block_on;
use runmat_value::Value;
#[test]
fn returns_one_platform_character_and_rejects_inputs() {
    let result = block_on(super::filesep_builtin(
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    assert_eq!(
        String::try_from(&result).unwrap(),
        std::path::MAIN_SEPARATOR.to_string()
    );
    assert!(block_on(super::filesep_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)])
    ))
    .is_err());
}
