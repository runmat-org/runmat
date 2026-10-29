use std::sync::Once;

use runmat_analysis_core::{
    AccuracyCriterion, AccuracyPlan, AccuracyPlanId, AccuracyResourceCeiling, AccuracyTarget,
    AccuracyTolerance, NumericalEquivalencePolicy, ObservationId,
};

use super::*;
use crate::builtins::fea::study_policies::{accuracy as policy, observation};
use crate::builtins::fea::study_values::{
    from_typed_value, nonempty_id, object_or_cell, register_immutable_runtime_classes,
    register_immutable_schemas, to_typed_value, ImmutableStudyClass, VALUE_FIELD,
};

use runmat_builtins::fea_contract::policy_types::classes;

pub(super) const TARGET_CLASS: &str = classes::ACCURACY_TARGET.display_name();
pub(super) const TOLERANCE_CLASS: &str = classes::ACCURACY_TOLERANCE.display_name();
pub(super) const EQUIVALENCE_CLASS: &str = classes::NUMERICAL_EQUIVALENCE_POLICY.display_name();
pub(super) const RESOURCES_CLASS: &str = classes::ACCURACY_RESOURCE_CEILING.display_name();
pub(super) const CRITERION_CLASS: &str = classes::ACCURACY_CRITERION.display_name();
pub(super) const PLAN_CLASS: &str = classes::ACCURACY_PLAN.display_name();

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

#[runtime_builtin(
    name = "fea.AccuracyTarget.observation",
    binding_variant = "default",
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
    binding_variant = "default",
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
    ($function:ident, $name:literal, $value:expr) => {
        #[runtime_builtin(name = $name, binding_variant = "default", builtin_path = "crate::builtins::fea::accuracy")]
        pub async fn $function() -> BuiltinResult<Value> { to_typed_value($name, TARGET_CLASS, $value) }
    };
}
no_arg_target!(
    target_numerical_result,
    "fea.AccuracyTarget.numericalResult",
    AccuracyTarget::NumericalResult
);
no_arg_target!(
    target_global,
    "fea.AccuracyTarget.globalEquilibrium",
    AccuracyTarget::GlobalEquilibrium
);
no_arg_target!(
    target_interface,
    "fea.AccuracyTarget.interfaceClosure",
    AccuracyTarget::InterfaceClosure
);
no_arg_target!(
    target_equivalence,
    "fea.AccuracyTarget.executionEquivalence",
    AccuracyTarget::ExecutionEquivalence
);

#[runtime_builtin(
    name = "fea.AccuracyTolerance.relative",
    binding_variant = "default",
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
    binding_variant = "default",
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

#[runtime_builtin(
    name = "fea.NumericalEquivalencePolicy.deterministic",
    binding_variant = "default",
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
    binding_variant = "default",
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
    binding_variant = "default",
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

#[runtime_builtin(
    name = "fea.AccuracyResourceCeiling.new",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::accuracy"
)]
pub async fn resources(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
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
        // `[]` authors "no resident-byte limit" (placement still applies machine capacity).
        maximum_resident_bytes: if matches!(&args[4], Value::Tensor(tensor) if tensor.is_empty()) {
            None
        } else {
            Some(
                u64::try_from(usize_from_value(NAME, &args[4])?).map_err(|_| {
                    builtin_error(NAME, &ERROR_INPUT, "resident-byte ceiling exceeds u64")
                })?,
            )
        },
    };
    serializable_to_object(NAME, &ERROR_INTERNAL, RESOURCES_CLASS, &ceiling)
}

#[runtime_builtin(
    name = "fea.AccuracyCriterion.new",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::accuracy"
)]
pub async fn criterion(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
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

#[runtime_builtin(
    name = "fea.AccuracyPlan.new",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::accuracy"
)]
pub async fn plan(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
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
    name = "fea.AccuracyPlan.baseline",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::accuracy"
)]
pub async fn baseline_plan(id: String) -> BuiltinResult<Value> {
    const NAME: &str = "fea.AccuracyPlan.baseline";
    serializable_to_object(
        NAME,
        &ERROR_INTERNAL,
        PLAN_CLASS,
        &AccuracyPlan::baseline(nonempty_id(NAME, "accuracy plan id", id)?),
    )
}

pub(super) fn accuracy_plan(name: &'static str, value: &Value) -> BuiltinResult<AccuracyPlan> {
    object_payload(name, value, PLAN_CLASS)
}

const ACCURACY_STUDY_CLASSES: [ImmutableStudyClass; 6] = [
    ImmutableStudyClass {
        class_name: TARGET_CLASS,
        type_identity: "runmat.fea.accuracy-target",
        fields: VALUE_FIELD,
        methods: &[
            ("observation", "fea.AccuracyTarget.observation"),
            ("quantity", "fea.AccuracyTarget.quantity"),
            ("numericalResult", "fea.AccuracyTarget.numericalResult"),
            ("globalEquilibrium", "fea.AccuracyTarget.globalEquilibrium"),
            ("interfaceClosure", "fea.AccuracyTarget.interfaceClosure"),
            (
                "executionEquivalence",
                "fea.AccuracyTarget.executionEquivalence",
            ),
        ],
    },
    ImmutableStudyClass {
        class_name: TOLERANCE_CLASS,
        type_identity: "runmat.fea.accuracy-tolerance",
        fields: VALUE_FIELD,
        methods: &[
            ("relative", "fea.AccuracyTolerance.relative"),
            ("absolute", "fea.AccuracyTolerance.absolute"),
        ],
    },
    ImmutableStudyClass {
        class_name: EQUIVALENCE_CLASS,
        type_identity: "runmat.fea.numerical-equivalence-policy",
        fields: VALUE_FIELD,
        methods: &[
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
    },
    ImmutableStudyClass {
        class_name: RESOURCES_CLASS,
        type_identity: "runmat.fea.accuracy-resource-ceiling",
        fields: RESOURCES_FIELDS,
        methods: &[("new", "fea.AccuracyResourceCeiling.new")],
    },
    ImmutableStudyClass {
        class_name: CRITERION_CLASS,
        type_identity: "runmat.fea.accuracy-criterion",
        fields: CRITERION_FIELDS,
        methods: &[("new", "fea.AccuracyCriterion.new")],
    },
    ImmutableStudyClass {
        class_name: PLAN_CLASS,
        type_identity: "runmat.fea.accuracy-plan",
        fields: PLAN_FIELDS,
        methods: &[
            ("new", "fea.AccuracyPlan.new"),
            ("baseline", "fea.AccuracyPlan.baseline"),
        ],
    },
];

pub(super) fn ensure_accuracy_classes_registered() {
    static SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(classes::ACCURACY_TARGET);
    SCHEMAS.call_once(|| {
        static FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        register_immutable_schemas(&FIELDS, &ACCURACY_STUDY_CLASSES);
    });
    RUNTIME_CLASSES.ensure(|| register_immutable_runtime_classes(&ACCURACY_STUDY_CLASSES));
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_analysis_core::{
        AccuracyLimitAction, ErrorContributionKind, ObservationQuantity, RequiredAccuracyEvidence,
        UnavailableEvidencePolicy,
    };
    use runmat_value::CellArray;

    fn registered() {
        super::super::ensure_value_classes_registered();
    }

    fn tip_criterion() -> Value {
        block_on(criterion(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("tip_mesh".into()),
                block_on(target_observation("tip_displacement".into())).unwrap(),
                block_on(policy::contribution_spatial()).unwrap(),
                block_on(tolerance_relative(Value::Num(0.01))).unwrap(),
                block_on(policy::evidence_convergence()).unwrap(),
                Value::Bool(true),
            ]),
        ))
        .unwrap()
    }

    fn ceiling(bytes: Value) -> BuiltinResult<Value> {
        block_on(resources(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(3.0),
                Value::Int(IntValue::U8(2)),
                Value::Num(100.0),
                Value::Num(600_000.0),
                bytes,
            ]),
        ))
    }

    #[test]
    fn targets_preserve_typed_values_and_reject_empty_or_untyped_inputs() {
        registered();
        let observation = block_on(target_observation("tip_displacement".into())).unwrap();
        assert_eq!(
            from_typed_value::<AccuracyTarget>("test", TARGET_CLASS, &observation).unwrap(),
            AccuracyTarget::Observation(ObservationId("tip_displacement".into()))
        );
        assert!(block_on(target_observation("  ".into())).is_err());

        let stress = block_on(observation::quantity_von_mises_stress()).unwrap();
        let quantity = block_on(target_quantity(stress)).unwrap();
        assert_eq!(
            from_typed_value::<AccuracyTarget>("test", TARGET_CLASS, &quantity).unwrap(),
            AccuracyTarget::Quantity(ObservationQuantity::VonMisesStress)
        );
        assert!(block_on(target_quantity(Value::String("stress".into()))).is_err());

        for (value, expected) in [
            (
                block_on(target_numerical_result()),
                AccuracyTarget::NumericalResult,
            ),
            (block_on(target_global()), AccuracyTarget::GlobalEquilibrium),
            (
                block_on(target_interface()),
                AccuracyTarget::InterfaceClosure,
            ),
            (
                block_on(target_equivalence()),
                AccuracyTarget::ExecutionEquivalence,
            ),
        ] {
            let value = value.unwrap();
            assert_eq!(
                from_typed_value::<AccuracyTarget>("test", TARGET_CLASS, &value).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn tolerances_and_equivalence_store_finite_scalars_and_reject_non_numeric_inputs() {
        registered();
        let relative = block_on(tolerance_relative(Value::Int(IntValue::I8(1)))).unwrap();
        assert_eq!(
            from_typed_value::<AccuracyTolerance>("test", TOLERANCE_CLASS, &relative).unwrap(),
            AccuracyTolerance::Relative(1.0)
        );
        let displacement = block_on(observation::quantity_displacement()).unwrap();
        let absolute = block_on(tolerance_absolute(displacement, Value::Num(1e-4))).unwrap();
        assert_eq!(
            from_typed_value::<AccuracyTolerance>("test", TOLERANCE_CLASS, &absolute).unwrap(),
            AccuracyTolerance::Absolute {
                quantity: ObservationQuantity::Displacement,
                canonical_value: 1e-4,
            }
        );
        assert!(block_on(tolerance_relative(Value::Bool(true))).is_err());
        assert!(block_on(tolerance_relative(Value::Num(f64::NAN))).is_err());
        assert!(block_on(tolerance_absolute(
            Value::String("displacement".into()),
            Value::Num(1.0)
        ))
        .is_err());

        let deterministic = block_on(equivalence_deterministic()).unwrap();
        let relative = block_on(equivalence_relative(Value::Num(1e-10))).unwrap();
        let absolute = block_on(equivalence_absolute(Value::Num(1e-12))).unwrap();
        for (value, expected) in [
            (deterministic, NumericalEquivalencePolicy::Deterministic),
            (
                relative,
                NumericalEquivalencePolicy::RelativeTolerance(1e-10),
            ),
            (
                absolute,
                NumericalEquivalencePolicy::AbsoluteTolerance(1e-12),
            ),
        ] {
            assert_eq!(
                from_typed_value::<NumericalEquivalencePolicy>("test", EQUIVALENCE_CLASS, &value)
                    .unwrap(),
                expected
            );
        }
        assert!(block_on(equivalence_relative(Value::String("1e-10".into()))).is_err());
    }

    #[test]
    fn object_layer_restores_absent_nullable_fields_and_wire_decode_stays_strict() {
        registered();
        for resident in [Some(4096), None] {
            let mut plan = AccuracyPlan::baseline("bracket.accuracy");
            plan.resources.maximum_resident_bytes = resident;
            let value = serializable_to_object("test", &ERROR_INTERNAL, PLAN_CLASS, &plan).unwrap();
            assert_eq!(
                object_payload::<AccuracyPlan>("test", &value, PLAN_CLASS).unwrap(),
                plan
            );
            let value =
                serializable_to_object("test", &ERROR_INTERNAL, RESOURCES_CLASS, &plan.resources)
                    .unwrap();
            assert_eq!(
                object_payload::<AccuracyResourceCeiling>("test", &value, RESOURCES_CLASS).unwrap(),
                plan.resources
            );
        }
        // Wire decode keeps rejecting an omitted resident-byte ceiling.
        let mut wire =
            serde_json::to_value(AccuracyPlan::baseline("bracket.accuracy").resources).unwrap();
        wire.as_object_mut()
            .unwrap()
            .remove("maximum_resident_bytes");
        let err = serde_json::from_value::<AccuracyResourceCeiling>(wire.clone()).unwrap_err();
        assert!(err
            .to_string()
            .contains("missing field `maximum_resident_bytes`"));
        // A required non-nullable property absent from an object is still rejected as missing.
        wire.as_object_mut().unwrap().remove("maximum_samples");
        let err = serde_json::from_value::<AccuracyResourceCeiling>(wire).unwrap_err();
        assert!(
            err.to_string().contains("missing field `maximum_samples`"),
            "{err}"
        );
    }

    #[test]
    fn resource_ceilings_stay_exact_within_the_wire_bound_and_reject_invalid_counts() {
        registered();
        let max = runmat_types::wire_integer::MAX_WIRE_INTEGER;
        let value = ceiling(Value::Int(IntValue::U64(max))).unwrap();
        let ceiling_value: AccuracyResourceCeiling =
            object_payload("test", &value, RESOURCES_CLASS).unwrap();
        assert_eq!(ceiling_value.maximum_resident_bytes, Some(max));
        assert_eq!(ceiling_value.maximum_solver_tightenings, 2);
        assert_eq!(ceiling_value.maximum_wall_time_milliseconds, 600_000);
        assert_eq!(
            ceiling(Value::Num(1024.0))
                .and_then(|value| object_payload::<AccuracyResourceCeiling>(
                    "test",
                    &value,
                    RESOURCES_CLASS
                ))
                .unwrap()
                .maximum_resident_bytes,
            Some(1024)
        );
        let unbounded = ceiling(empty_double_value()).unwrap();
        let unbounded: AccuracyResourceCeiling =
            object_payload("test", &unbounded, RESOURCES_CLASS).unwrap();
        assert_eq!(unbounded.maximum_resident_bytes, None);
        assert!(ceiling(Value::Int(IntValue::U64(max + 1))).is_err());
        assert!(ceiling(Value::Int(IntValue::U64(u64::MAX))).is_err());
        assert!(ceiling(Value::Num((max + 1) as f64)).is_err());
        assert!(ceiling(Value::Num(-1.0)).is_err());
        assert!(ceiling(Value::Num(2.5)).is_err());
        assert!(block_on(resources(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num((max + 1) as f64),
                Value::Num(0.0),
                Value::Num(0.0),
                Value::Num(0.0),
                Value::Num(1.0),
            ])
        ))
        .is_err());
        assert!(block_on(resources(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)])
        ))
        .is_err());
    }

    #[test]
    fn criterion_requires_typed_policies_and_a_logical_mandatory_flag() {
        registered();
        let value = tip_criterion();
        let criterion_value: AccuracyCriterion =
            object_payload("test", &value, CRITERION_CLASS).unwrap();
        assert_eq!(criterion_value.criterion_id, "tip_mesh");
        assert_eq!(
            criterion_value.contribution,
            ErrorContributionKind::SpatialDiscretization
        );
        assert_eq!(
            criterion_value.evidence,
            RequiredAccuracyEvidence::ConvergenceStudy
        );
        assert!(criterion_value.mandatory);

        let numeric_flag = block_on(criterion(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("tip_mesh".into()),
                block_on(target_global()).unwrap(),
                block_on(policy::contribution_algebraic()).unwrap(),
                block_on(tolerance_relative(Value::Num(1e-8))).unwrap(),
                block_on(policy::evidence_residual()).unwrap(),
                Value::Num(1.0),
            ]),
        ));
        assert!(numeric_flag.unwrap_err().to_string().contains("logical"));
        assert!(block_on(criterion(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "tip_mesh".into()
            )])
        ))
        .is_err());
    }

    #[test]
    fn plans_accept_criteria_cells_and_round_trip_portably() {
        registered();
        let criteria = Value::Cell(CellArray::new(vec![tip_criterion()], 1, 1).unwrap());
        let value = block_on(plan(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("static_acceptance".into()),
                criteria,
                block_on(policy::unavailable_fail()).unwrap(),
                block_on(policy::limit_refine()).unwrap(),
                block_on(equivalence_relative(Value::Num(1e-12))).unwrap(),
                ceiling(Value::Num(2e9)).unwrap(),
            ]),
        ))
        .unwrap();
        let encoded = crate::execution::value_codec::encode_inline_value(&value).unwrap();
        let decoded = crate::execution::value_codec::decode_inline_value(&encoded).unwrap();
        let plan_value = accuracy_plan("test", &decoded).unwrap();
        assert_eq!(plan_value.accuracy_plan_id.0, "static_acceptance");
        assert_eq!(plan_value.criteria.len(), 1);
        assert_eq!(
            plan_value.unavailable_evidence,
            UnavailableEvidencePolicy::Fail
        );
        assert_eq!(plan_value.exceeded_limit, AccuracyLimitAction::Refine);
        assert!(block_on(plan(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "p".into()
            )])
        ))
        .is_err());
    }

    #[test]
    fn baseline_matches_the_core_baseline() {
        registered();
        let value = block_on(baseline_plan("bracket.accuracy".into())).unwrap();
        assert_eq!(
            accuracy_plan("test", &value).unwrap(),
            AccuracyPlan::baseline("bracket.accuracy")
        );
        assert!(block_on(baseline_plan(String::new())).is_err());
    }
}
