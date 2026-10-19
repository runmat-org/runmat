use std::sync::Once;

use runmat_analysis_core::{
    AccuracyCriterion, AccuracyPlan, AccuracyPlanId, AccuracyResourceCeiling, AccuracyTarget,
    AccuracyTolerance, NumericalEquivalencePolicy, ObservationId,
};

use super::*;
use crate::builtins::fea::study_policies::{accuracy as policy, observation};
use crate::builtins::fea::study_values::{
    from_typed_value, nonempty_id, object_or_cell, register_immutable_class, to_typed_value,
    VALUE_FIELD,
};

pub(super) const TARGET_CLASS: &str = "fea.AccuracyTarget";
pub(super) const TOLERANCE_CLASS: &str = "fea.AccuracyTolerance";
pub(super) const EQUIVALENCE_CLASS: &str = "fea.NumericalEquivalencePolicy";
pub(super) const RESOURCES_CLASS: &str = "fea.AccuracyResourceCeiling";
pub(super) const CRITERION_CLASS: &str = "fea.AccuracyCriterion";
pub(super) const PLAN_CLASS: &str = "fea.AccuracyPlan";

const RESOURCES_FIELDS: &[&str] = &[
    "maximum_refinement_cycles",
    "maximum_solver_tightenings",
    "maximum_samples",
    "maximum_wall_time_milliseconds",
    "maximum_resident_bytes",
];
const CRITERION_FIELDS: &[&str] = &[
    "criterion_id",
    "target",
    "contribution",
    "tolerance",
    "evidence",
    "mandatory",
];
const PLAN_FIELDS: &[&str] = &[
    "accuracy_plan_id",
    "criteria",
    "unavailable_evidence",
    "exceeded_limit",
    "equivalence",
    "resources",
];

const fn required(
    name: &'static str,
    ty: BuiltinParamType,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}
const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const ID_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Stable identifier.",
)];
const VALUE_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "value",
    BuiltinParamType::NumericScalar,
    "Finite positive tolerance.",
)];
const ABSOLUTE_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "quantity",
        BuiltinParamType::Any,
        "Typed observation quantity.",
    ),
    required(
        "canonicalValue",
        BuiltinParamType::NumericScalar,
        "Positive tolerance in canonical SI units.",
    ),
];
const RESOURCES_INPUTS: [BuiltinParamDescriptor; 5] = [
    required(
        "maximumRefinementCycles",
        BuiltinParamType::NumericScalar,
        "Maximum automatic spatial refinements.",
    ),
    required(
        "maximumSolverTightenings",
        BuiltinParamType::NumericScalar,
        "Maximum solver-tolerance tightenings.",
    ),
    required(
        "maximumSamples",
        BuiltinParamType::NumericScalar,
        "Maximum observation or study samples.",
    ),
    required(
        "maximumWallTimeMilliseconds",
        BuiltinParamType::NumericScalar,
        "Maximum wall time in milliseconds.",
    ),
    required(
        "maximumResidentBytes",
        BuiltinParamType::NumericScalar,
        "Maximum resident bytes.",
    ),
];
const CRITERION_INPUTS: [BuiltinParamDescriptor; 6] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Criterion identifier.",
    ),
    required("target", BuiltinParamType::Any, "Typed accuracy target."),
    required(
        "contribution",
        BuiltinParamType::Any,
        "Typed error contribution.",
    ),
    required(
        "tolerance",
        BuiltinParamType::Any,
        "Typed accuracy tolerance.",
    ),
    required(
        "evidence",
        BuiltinParamType::Any,
        "Typed required evidence.",
    ),
    required(
        "mandatory",
        BuiltinParamType::LogicalArray,
        "Whether failure blocks acceptance.",
    ),
];
const PLAN_INPUTS: [BuiltinParamDescriptor; 6] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Accuracy-plan identifier.",
    ),
    required(
        "criteria",
        BuiltinParamType::Any,
        "Typed criterion or cell array.",
    ),
    required(
        "unavailableEvidence",
        BuiltinParamType::Any,
        "Typed unavailable-evidence policy.",
    ),
    required(
        "exceededLimit",
        BuiltinParamType::Any,
        "Typed limit action.",
    ),
    required(
        "equivalence",
        BuiltinParamType::Any,
        "Typed numerical-equivalence policy.",
    ),
    required(
        "resources",
        BuiltinParamType::Any,
        "Typed resource ceiling.",
    ),
];

const NUMERIC_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "tolerance or resource ceiling",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Resource ceilings remain exact; physical tolerances cross once into canonical binary64 storage.",
}];
const NUMERIC_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "typed accuracy parameter",
        inputs: &NUMERIC_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Accuracy authoring constructs immutable host values and does not execute a solver.",
    }];
const STRUCTURAL_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "This constructor accepts identifiers and typed FEA policy objects only.",
};

macro_rules! descriptor {
    ($descriptor:ident, $signatures:ident, $label:literal, $inputs:expr) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: $inputs,
            outputs: &OUT_ANY,
        }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

descriptor!(
    TARGET_OBSERVATION_DESCRIPTOR,
    TARGET_OBSERVATION_SIGNATURES,
    "target = fea.AccuracyTarget.observation(id)",
    &ID_INPUTS
);
descriptor!(
    TARGET_QUANTITY_DESCRIPTOR,
    TARGET_QUANTITY_SIGNATURES,
    "target = fea.AccuracyTarget.quantity(quantity)",
    &[required(
        "quantity",
        BuiltinParamType::Any,
        "Typed observation quantity."
    )]
);
descriptor!(
    TARGET_NUMERICAL_RESULT_DESCRIPTOR,
    TARGET_NUMERICAL_RESULT_SIGNATURES,
    "target = fea.AccuracyTarget.numericalResult()",
    &NO_INPUTS
);
descriptor!(
    TARGET_GLOBAL_DESCRIPTOR,
    TARGET_GLOBAL_SIGNATURES,
    "target = fea.AccuracyTarget.globalEquilibrium()",
    &NO_INPUTS
);
descriptor!(
    TARGET_INTERFACE_DESCRIPTOR,
    TARGET_INTERFACE_SIGNATURES,
    "target = fea.AccuracyTarget.interfaceClosure()",
    &NO_INPUTS
);
descriptor!(
    TARGET_EQUIVALENCE_DESCRIPTOR,
    TARGET_EQUIVALENCE_SIGNATURES,
    "target = fea.AccuracyTarget.executionEquivalence()",
    &NO_INPUTS
);

#[runtime_builtin(
    name = "fea.AccuracyTarget.observation",
    category = "fea",
    summary = "Target a typed observation for accuracy acceptance.",
    keywords = "fea,accuracy,target,observation,typed",
    descriptor(TARGET_OBSERVATION_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::accuracy"
)]
pub async fn target_observation(id: String) -> BuiltinResult<Value> {
    const NAME: &str = "fea.AccuracyTarget.observation";
    to_typed_value(
        NAME,
        TARGET_CLASS,
        AccuracyTarget::Observation(ObservationId(nonempty_id(NAME, "observation id", id)?)),
    )
}
#[runtime_builtin(
    name = "fea.AccuracyTarget.quantity",
    category = "fea",
    summary = "Target a physical quantity for accuracy acceptance.",
    keywords = "fea,accuracy,target,quantity,typed",
    descriptor(TARGET_QUANTITY_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::accuracy"
)]
pub async fn target_quantity(quantity: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.AccuracyTarget.quantity";
    to_typed_value(
        NAME,
        TARGET_CLASS,
        AccuracyTarget::Quantity(observation::quantity(NAME, &quantity)?),
    )
}

macro_rules! no_arg_target {
    ($function:ident, $name:literal, $descriptor:ident, $value:expr) => {
        #[runtime_builtin(name = $name, category = "fea", summary = "Select a typed study-level accuracy target.", keywords = "fea,accuracy,target,typed", descriptor($descriptor), integer_audit(STRUCTURAL_AUDIT), builtin_path = "crate::builtins::fea::accuracy")]
        pub async fn $function() -> BuiltinResult<Value> { to_typed_value($name, TARGET_CLASS, $value) }
    };
}
no_arg_target!(
    target_numerical_result,
    "fea.AccuracyTarget.numericalResult",
    TARGET_NUMERICAL_RESULT_DESCRIPTOR,
    AccuracyTarget::NumericalResult
);
no_arg_target!(
    target_global,
    "fea.AccuracyTarget.globalEquilibrium",
    TARGET_GLOBAL_DESCRIPTOR,
    AccuracyTarget::GlobalEquilibrium
);
no_arg_target!(
    target_interface,
    "fea.AccuracyTarget.interfaceClosure",
    TARGET_INTERFACE_DESCRIPTOR,
    AccuracyTarget::InterfaceClosure
);
no_arg_target!(
    target_equivalence,
    "fea.AccuracyTarget.executionEquivalence",
    TARGET_EQUIVALENCE_DESCRIPTOR,
    AccuracyTarget::ExecutionEquivalence
);

descriptor!(
    TOLERANCE_RELATIVE_DESCRIPTOR,
    TOLERANCE_RELATIVE_SIGNATURES,
    "tolerance = fea.AccuracyTolerance.relative(value)",
    &VALUE_INPUTS
);
descriptor!(
    TOLERANCE_ABSOLUTE_DESCRIPTOR,
    TOLERANCE_ABSOLUTE_SIGNATURES,
    "tolerance = fea.AccuracyTolerance.absolute(quantity, canonicalValue)",
    &ABSOLUTE_INPUTS
);
#[runtime_builtin(
    name = "fea.AccuracyTolerance.relative",
    category = "fea",
    summary = "Create a relative accuracy tolerance.",
    keywords = "fea,accuracy,tolerance,relative,typed",
    descriptor(TOLERANCE_RELATIVE_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::accuracy"
)]
pub async fn tolerance_relative(value: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.AccuracyTolerance.relative";
    to_typed_value(
        NAME,
        TOLERANCE_CLASS,
        AccuracyTolerance::Relative(values::finite_numeric_scalar(NAME, &value, "value")?),
    )
}
#[runtime_builtin(
    name = "fea.AccuracyTolerance.absolute",
    category = "fea",
    summary = "Create a dimensioned absolute accuracy tolerance.",
    keywords = "fea,accuracy,tolerance,absolute,typed",
    descriptor(TOLERANCE_ABSOLUTE_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::accuracy"
)]
pub async fn tolerance_absolute(quantity: Value, value: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.AccuracyTolerance.absolute";
    to_typed_value(
        NAME,
        TOLERANCE_CLASS,
        AccuracyTolerance::Absolute {
            quantity: observation::quantity(NAME, &quantity)?,
            canonical_value: values::finite_numeric_scalar(NAME, &value, "canonicalValue")?,
        },
    )
}

descriptor!(
    EQUIVALENCE_DETERMINISTIC_DESCRIPTOR,
    EQUIVALENCE_DETERMINISTIC_SIGNATURES,
    "policy = fea.NumericalEquivalencePolicy.deterministic()",
    &NO_INPUTS
);
descriptor!(
    EQUIVALENCE_RELATIVE_DESCRIPTOR,
    EQUIVALENCE_RELATIVE_SIGNATURES,
    "policy = fea.NumericalEquivalencePolicy.relativeTolerance(value)",
    &VALUE_INPUTS
);
descriptor!(
    EQUIVALENCE_ABSOLUTE_DESCRIPTOR,
    EQUIVALENCE_ABSOLUTE_SIGNATURES,
    "policy = fea.NumericalEquivalencePolicy.absoluteTolerance(value)",
    &VALUE_INPUTS
);
#[runtime_builtin(
    name = "fea.NumericalEquivalencePolicy.deterministic",
    category = "fea",
    summary = "Require deterministic numerical equivalence.",
    keywords = "fea,accuracy,equivalence,deterministic,typed",
    descriptor(EQUIVALENCE_DETERMINISTIC_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::accuracy"
)]
pub async fn equivalence_deterministic() -> BuiltinResult<Value> {
    to_typed_value(
        "fea.NumericalEquivalencePolicy.deterministic",
        EQUIVALENCE_CLASS,
        NumericalEquivalencePolicy::Deterministic,
    )
}
#[runtime_builtin(
    name = "fea.NumericalEquivalencePolicy.relativeTolerance",
    category = "fea",
    summary = "Bound relative numerical-equivalence variation.",
    keywords = "fea,accuracy,equivalence,relative,typed",
    descriptor(EQUIVALENCE_RELATIVE_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::accuracy"
)]
pub async fn equivalence_relative(value: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.NumericalEquivalencePolicy.relativeTolerance";
    to_typed_value(
        NAME,
        EQUIVALENCE_CLASS,
        NumericalEquivalencePolicy::RelativeTolerance(values::finite_numeric_scalar(
            NAME, &value, "value",
        )?),
    )
}
#[runtime_builtin(
    name = "fea.NumericalEquivalencePolicy.absoluteTolerance",
    category = "fea",
    summary = "Bound absolute numerical-equivalence variation.",
    keywords = "fea,accuracy,equivalence,absolute,typed",
    descriptor(EQUIVALENCE_ABSOLUTE_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::accuracy"
)]
pub async fn equivalence_absolute(value: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.NumericalEquivalencePolicy.absoluteTolerance";
    to_typed_value(
        NAME,
        EQUIVALENCE_CLASS,
        NumericalEquivalencePolicy::AbsoluteTolerance(values::finite_numeric_scalar(
            NAME, &value, "value",
        )?),
    )
}

descriptor!(RESOURCES_DESCRIPTOR, RESOURCES_SIGNATURES, "resources = fea.AccuracyResourceCeiling.new(maximumRefinementCycles, maximumSolverTightenings, maximumSamples, maximumWallTimeMilliseconds, maximumResidentBytes)", &RESOURCES_INPUTS);
#[runtime_builtin(
    name = "fea.AccuracyResourceCeiling.new",
    category = "fea",
    summary = "Bound automatic accuracy work and resource use.",
    keywords = "fea,accuracy,resources,ceiling,typed",
    descriptor(RESOURCES_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::accuracy"
)]
pub async fn resources(args: Vec<Value>) -> BuiltinResult<Value> {
    const NAME: &str = "fea.AccuracyResourceCeiling.new";
    if args.len() != 5 {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "expected five resource ceilings",
        ));
    }
    let ceiling = AccuracyResourceCeiling {
        maximum_refinement_cycles: usize_from_value(NAME, &args[0])?,
        maximum_solver_tightenings: usize_from_value(NAME, &args[1])?,
        maximum_samples: usize_from_value(NAME, &args[2])?,
        maximum_wall_time_milliseconds: u64::try_from(usize_from_value(NAME, &args[3])?)
            .map_err(|_| builtin_error(NAME, &ERROR_INPUT, "wall-time ceiling exceeds u64"))?,
        maximum_resident_bytes: u64::try_from(usize_from_value(NAME, &args[4])?)
            .map_err(|_| builtin_error(NAME, &ERROR_INPUT, "resident-byte ceiling exceeds u64"))?,
    };
    serializable_to_object(NAME, &ERROR_INTERNAL, RESOURCES_CLASS, &ceiling)
}

descriptor!(CRITERION_DESCRIPTOR, CRITERION_SIGNATURES, "criterion = fea.AccuracyCriterion.new(id, target, contribution, tolerance, evidence, mandatory)", &CRITERION_INPUTS);
#[runtime_builtin(
    name = "fea.AccuracyCriterion.new",
    category = "fea",
    summary = "Create one typed accuracy acceptance criterion.",
    keywords = "fea,accuracy,criterion,evidence,typed",
    descriptor(CRITERION_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::accuracy"
)]
pub async fn criterion(args: Vec<Value>) -> BuiltinResult<Value> {
    const NAME: &str = "fea.AccuracyCriterion.new";
    if args.len() != 6 {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "expected id, target, contribution, tolerance, evidence, and mandatory",
        ));
    }
    serializable_to_object(
        NAME,
        &ERROR_INTERNAL,
        CRITERION_CLASS,
        &AccuracyCriterion {
            criterion_id: nonempty_id(
                NAME,
                "criterion id",
                scalar_string(&args[0], NAME, &ERROR_INPUT)?,
            )?,
            target: from_typed_value(NAME, TARGET_CLASS, &args[1])?,
            contribution: policy::contribution(NAME, &args[2])?,
            tolerance: from_typed_value(NAME, TOLERANCE_CLASS, &args[3])?,
            evidence: policy::evidence(NAME, &args[4])?,
            mandatory: logical_from_value(NAME, &args[5])?,
        },
    )
}

descriptor!(PLAN_DESCRIPTOR, PLAN_SIGNATURES, "plan = fea.AccuracyPlan.new(id, criteria, unavailableEvidence, exceededLimit, equivalence, resources)", &PLAN_INPUTS);
descriptor!(
    PLAN_EXPERIMENTAL_DESCRIPTOR,
    PLAN_EXPERIMENTAL_SIGNATURES,
    "plan = fea.AccuracyPlan.experimentalBaseline(id)",
    &ID_INPUTS
);
#[runtime_builtin(
    name = "fea.AccuracyPlan.new",
    category = "fea",
    summary = "Create a typed study-step accuracy plan.",
    keywords = "fea,accuracy,plan,acceptance,typed",
    descriptor(PLAN_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::accuracy"
)]
pub async fn plan(args: Vec<Value>) -> BuiltinResult<Value> {
    const NAME: &str = "fea.AccuracyPlan.new";
    if args.len() != 6 {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "expected id, criteria, unavailableEvidence, exceededLimit, equivalence, and resources",
        ));
    }
    serializable_to_object(
        NAME,
        &ERROR_INTERNAL,
        PLAN_CLASS,
        &AccuracyPlan {
            accuracy_plan_id: AccuracyPlanId(nonempty_id(
                NAME,
                "accuracy plan id",
                scalar_string(&args[0], NAME, &ERROR_INPUT)?,
            )?),
            criteria: object_or_cell(NAME, &args[1], CRITERION_CLASS)?,
            unavailable_evidence: policy::unavailable(NAME, &args[2])?,
            exceeded_limit: policy::limit_action(NAME, &args[3])?,
            equivalence: from_typed_value(NAME, EQUIVALENCE_CLASS, &args[4])?,
            resources: object_payload(NAME, &args[5], RESOURCES_CLASS)?,
        },
    )
}
#[runtime_builtin(
    name = "fea.AccuracyPlan.experimentalBaseline",
    category = "fea",
    summary = "Create the explicit non-production baseline accuracy plan.",
    keywords = "fea,accuracy,plan,experimental,typed",
    descriptor(PLAN_EXPERIMENTAL_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::accuracy"
)]
pub async fn experimental_plan(id: String) -> BuiltinResult<Value> {
    const NAME: &str = "fea.AccuracyPlan.experimentalBaseline";
    serializable_to_object(
        NAME,
        &ERROR_INTERNAL,
        PLAN_CLASS,
        &AccuracyPlan::experimental_baseline(nonempty_id(NAME, "accuracy plan id", id)?),
    )
}

pub(super) fn accuracy_plan(name: &'static str, value: &Value) -> BuiltinResult<AccuracyPlan> {
    object_payload(name, value, PLAN_CLASS)
}

pub(super) fn ensure_accuracy_classes_registered() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        register_immutable_class(
            TARGET_CLASS,
            "runmat.fea.accuracy-target",
            VALUE_FIELD,
            &[
                ("observation", "fea.AccuracyTarget.observation"),
                ("quantity", "fea.AccuracyTarget.quantity"),
                ("globalEquilibrium", "fea.AccuracyTarget.globalEquilibrium"),
                ("interfaceClosure", "fea.AccuracyTarget.interfaceClosure"),
                (
                    "executionEquivalence",
                    "fea.AccuracyTarget.executionEquivalence",
                ),
            ],
        );
        register_immutable_class(
            TOLERANCE_CLASS,
            "runmat.fea.accuracy-tolerance",
            VALUE_FIELD,
            &[
                ("relative", "fea.AccuracyTolerance.relative"),
                ("absolute", "fea.AccuracyTolerance.absolute"),
            ],
        );
        register_immutable_class(
            EQUIVALENCE_CLASS,
            "runmat.fea.numerical-equivalence-policy",
            VALUE_FIELD,
            &[
                (
                    "deterministic",
                    "fea.NumericalEquivalencePolicy.deterministic",
                ),
                (
                    "relativeTolerance",
                    "fea.NumericalEquivalencePolicy.relativeTolerance",
                ),
                (
                    "absoluteTolerance",
                    "fea.NumericalEquivalencePolicy.absoluteTolerance",
                ),
            ],
        );
        register_immutable_class(
            RESOURCES_CLASS,
            "runmat.fea.accuracy-resource-ceiling",
            RESOURCES_FIELDS,
            &[("new", "fea.AccuracyResourceCeiling.new")],
        );
        register_immutable_class(
            CRITERION_CLASS,
            "runmat.fea.accuracy-criterion",
            CRITERION_FIELDS,
            &[("new", "fea.AccuracyCriterion.new")],
        );
        register_immutable_class(
            PLAN_CLASS,
            "runmat.fea.accuracy-plan",
            PLAN_FIELDS,
            &[
                ("new", "fea.AccuracyPlan.new"),
                (
                    "experimentalBaseline",
                    "fea.AccuracyPlan.experimentalBaseline",
                ),
            ],
        );
    });
}
