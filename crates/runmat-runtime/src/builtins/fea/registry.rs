use crate::builtins::fea::contracts::identities::{
    FEA_BOUNDARY_CONDITION_CLASS, FEA_COMPARE_CLASS, FEA_DOMAIN_CLASS, FEA_FIELD_CLASS,
    FEA_INTERFACE_CLASS, FEA_LOAD_CASE_CLASS, FEA_MATERIAL_ASSIGNMENT_CLASS, FEA_MATERIAL_CLASS,
    FEA_MODEL_CLASS, FEA_PLAN_CLASS, FEA_RESULTS_CLASS, FEA_RUN_OPTIONS_CLASS,
    FEA_RUN_RESULT_CLASS, FEA_STEP_CLASS, FEA_STUDY_CLASS, FEA_SWEEP_CLASS, FEA_TRENDS_CLASS,
    FEA_VALIDATION_CLASS, FIELD_NAME, PLAN_NAME, PLOT_NAME, RESULTS_NAME, RUN_NAME, VALIDATE_NAME,
};
use runmat_types::MemberAccess;
use std::collections::HashMap;
use std::sync::OnceLock;

pub(in crate::builtins::fea) fn ensure_fea_classes_registered() {
    static REGISTER: OnceLock<()> = OnceLock::new();
    REGISTER.get_or_init(|| {
        let workflow_methods = workflow_methods();
        for class_name in [FEA_STUDY_CLASS, FEA_SWEEP_CLASS] {
            crate::class_registry::register_class(crate::class_registry::RuntimeClass {
                name: class_name.into(),
                parent: None,
                properties: HashMap::new(),
                methods: workflow_methods.clone(),
            });
        }
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: FEA_RUN_RESULT_CLASS.into(),
            parent: None,
            properties: HashMap::new(),
            methods: run_result_methods(),
        });
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: FEA_RESULTS_CLASS.into(),
            parent: None,
            properties: HashMap::new(),
            methods: results_methods(),
        });
        for class_name in [FEA_VALIDATION_CLASS, FEA_PLAN_CLASS, FEA_RUN_RESULT_CLASS] {
            if class_name == FEA_RUN_RESULT_CLASS {
                continue;
            }
            crate::class_registry::register_class(crate::class_registry::RuntimeClass {
                name: class_name.into(),
                parent: None,
                properties: HashMap::new(),
                methods: HashMap::new(),
            });
        }
        for class_name in [
            FEA_MODEL_CLASS,
            FEA_MATERIAL_CLASS,
            FEA_MATERIAL_ASSIGNMENT_CLASS,
            FEA_BOUNDARY_CONDITION_CLASS,
            FEA_LOAD_CASE_CLASS,
            FEA_STEP_CLASS,
            FEA_DOMAIN_CLASS,
            FEA_INTERFACE_CLASS,
            FEA_RUN_OPTIONS_CLASS,
            FEA_FIELD_CLASS,
            FEA_COMPARE_CLASS,
            FEA_TRENDS_CLASS,
        ] {
            crate::class_registry::register_class(crate::class_registry::RuntimeClass {
                name: class_name.into(),
                parent: None,
                properties: HashMap::new(),
                methods: if class_name == FEA_FIELD_CLASS {
                    field_methods()
                } else {
                    HashMap::new()
                },
            });
        }
    });
}

pub(in crate::builtins::fea) fn workflow_methods(
) -> HashMap<runmat_types::MethodName, crate::class_registry::RuntimeMethod> {
    [
        (
            runmat_types::StaticMethodName::new("validate"),
            VALIDATE_NAME,
        ),
        (runmat_types::StaticMethodName::new("plan"), PLAN_NAME),
        (runmat_types::StaticMethodName::new("run"), RUN_NAME),
    ]
    .into_iter()
    .map(|(name, function_name)| {
        (
            name.into(),
            crate::class_registry::RuntimeMethod {
                name: name.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: function_name.into(),
                implicit_class_argument: None,
            },
        )
    })
    .collect()
}

pub(in crate::builtins::fea) fn run_result_methods(
) -> HashMap<runmat_types::MethodName, crate::class_registry::RuntimeMethod> {
    [
        (runmat_types::StaticMethodName::new("results"), RESULTS_NAME),
        (runmat_types::StaticMethodName::new("field"), FIELD_NAME),
        (runmat_types::StaticMethodName::new("plot"), PLOT_NAME),
    ]
    .into_iter()
    .map(|(name, function_name)| {
        (
            name.into(),
            crate::class_registry::RuntimeMethod {
                name: name.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: function_name.into(),
                implicit_class_argument: None,
            },
        )
    })
    .collect()
}

pub(in crate::builtins::fea) fn results_methods(
) -> HashMap<runmat_types::MethodName, crate::class_registry::RuntimeMethod> {
    [
        (runmat_types::StaticMethodName::new("field"), FIELD_NAME),
        (runmat_types::StaticMethodName::new("plot"), PLOT_NAME),
    ]
    .into_iter()
    .map(|(name, function_name)| {
        (
            name.into(),
            crate::class_registry::RuntimeMethod {
                name: name.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: function_name.into(),
                implicit_class_argument: None,
            },
        )
    })
    .collect()
}

pub(in crate::builtins::fea) fn field_methods(
) -> HashMap<runmat_types::MethodName, crate::class_registry::RuntimeMethod> {
    [(runmat_types::StaticMethodName::new("plot"), PLOT_NAME)]
        .into_iter()
        .map(|(name, function_name)| {
            (
                name.into(),
                crate::class_registry::RuntimeMethod {
                    name: name.into(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: function_name.into(),
                    implicit_class_argument: None,
                },
            )
        })
        .collect()
}
