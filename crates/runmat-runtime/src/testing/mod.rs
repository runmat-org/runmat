mod class_registry;
mod constraint_equality;
mod context;
mod objects;
mod services;

pub use class_registry::ensure_testing_classes;
pub(crate) use constraint_equality::constraint_values_equal;
pub(crate) use context::ContextState;
pub use context::{
    active_test_context, ensure_runtime_teardown_available, ensure_runtime_teardown_available_in,
    install_test_context, install_test_context_in, record_runtime_teardown, record_test_command,
    ActiveTestContext, RuntimeTeardownInvocation, TestContextGuard, TestContextHandle,
};
pub use objects::{
    diagnostic_object, fixture_object, object_array_or_scalar, object_array_or_scalar_in_context,
    object_from_fields_in_context, plugin_object, replace_object_properties_in_context,
    test_case_object, test_case_object_in_context, test_result_object,
    test_result_object_in_context, test_suite_object, test_suite_object_in_context,
    TEST_CASE_CLASS, TEST_RESULT_CLASS, TEST_SUITE_CLASS,
};
pub(crate) use services::run_test_suite_in_context;
pub use services::{
    discover_tests, discover_tests_in_context, install_test_services, run_tests,
    run_tests_in_context, RunSuiteFuture, RuntimeTestServices, TestServicesGuard,
};
