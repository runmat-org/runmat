use std::rc::Rc;

use crate::context::RuntimeContext;
use crate::execution::RuntimeExecutionService;

use super::policy::WarningAction;
use super::{with_policy, WarningMode};

#[test]
fn runtime_contexts_own_independent_warning_policies() {
    let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));

    {
        let _scope = first.enter();
        with_policy(|policy| policy.set_identifier_mode("RunMat:test:first", WarningMode::Off));
    }
    {
        let _scope = second.enter();
        assert_eq!(
            with_policy(|policy| policy.lookup_mode("RunMat:test:first")),
            WarningMode::On
        );
        with_policy(|policy| policy.set_global_mode(WarningMode::Error));
    }
    {
        let _scope = first.enter();
        assert_eq!(
            with_policy(|policy| policy.lookup_mode("RunMat:test:first")),
            WarningMode::Off
        );
        assert_eq!(
            with_policy(|policy| policy.lookup_mode("RunMat:test:other")),
            WarningMode::On
        );
    }
}

#[test]
fn once_mode_records_triggering_per_identifier() {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let _scope = runtime.enter();
    with_policy(|policy| policy.set_global_mode(WarningMode::Once));

    assert_eq!(
        with_policy(|policy| policy.action_for("RunMat:test:once")),
        WarningAction::Display
    );
    assert_eq!(
        with_policy(|policy| policy.action_for("RunMat:test:once")),
        WarningAction::Suppress
    );
    assert_eq!(
        with_policy(|policy| policy.action_for("RunMat:test:different")),
        WarningAction::Display
    );
}
