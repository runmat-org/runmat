use super::*;
use runmat_analysis_core::CoupledPhysicsFamily;

fn step_to_object(
    builtin: &'static str,
    class_name: &'static str,
    step: ProcedureStep,
) -> BuiltinResult<Value> {
    if step.step_id.0.is_empty() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "procedure-step id must not be empty",
        ));
    }
    serializable_to_object(builtin, &ERROR_INTERNAL, class_name, &step)
}

pub(super) fn require_step_class(
    builtin: &'static str,
    class_name: &'static str,
    step: &ProcedureStep,
) -> BuiltinResult<()> {
    let expected = match (step.family, step.procedure, step.solution_space) {
        (
            PhysicsFamily::Structural,
            AnalysisProcedure::Static
            | AnalysisProcedure::ModalExtraction
            | AnalysisProcedure::HarmonicResponse
            | AnalysisProcedure::TransientResponse
            | AnalysisProcedure::Nonlinear,
            SolutionSpace::Direct,
        ) => Some(FEA_STRUCTURAL_STEP_CLASS),
        (
            PhysicsFamily::Thermal,
            AnalysisProcedure::Static | AnalysisProcedure::TransientResponse,
            SolutionSpace::Direct,
        ) => Some(FEA_THERMAL_STEP_CLASS),
        (
            PhysicsFamily::Electromagnetic,
            AnalysisProcedure::HarmonicResponse,
            SolutionSpace::Direct,
        ) => Some(FEA_ELECTROMAGNETIC_STEP_CLASS),
        (PhysicsFamily::Acoustic, AnalysisProcedure::HarmonicResponse, SolutionSpace::Direct) => {
            Some(FEA_ACOUSTIC_STEP_CLASS)
        }
        (
            PhysicsFamily::Fluid,
            AnalysisProcedure::Static | AnalysisProcedure::TransientResponse,
            SolutionSpace::Direct,
        ) => Some(FEA_FLUID_STEP_CLASS),
        (
            PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
            AnalysisProcedure::TransientResponse,
            SolutionSpace::Direct,
        ) => Some(FEA_THERMOMECHANICAL_STEP_CLASS),
        (
            PhysicsFamily::Coupled(CoupledPhysicsFamily::Electrothermal),
            AnalysisProcedure::TransientResponse | AnalysisProcedure::Nonlinear,
            SolutionSpace::Direct,
        ) => Some(FEA_ELECTROTHERMAL_STEP_CLASS),
        (
            PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
            AnalysisProcedure::Static | AnalysisProcedure::TransientResponse,
            SolutionSpace::Direct,
        ) => Some(FEA_CONJUGATE_HEAT_TRANSFER_STEP_CLASS),
        (
            PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction),
            AnalysisProcedure::TransientResponse,
            SolutionSpace::Direct,
        ) => Some(FEA_FLUID_STRUCTURE_STEP_CLASS),
        _ => None,
    };
    match expected {
        Some(expected) if expected == class_name => Ok(()),
        Some(expected) => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("procedure step belongs to `{expected}`, not `{class_name}`"),
        )),
        None => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "procedure step does not describe a supported concrete FEA step",
        )),
    }
}

macro_rules! step_builtin {
    ($function:ident, $descriptor:ident, $name:literal, $class:expr, $build:expr) => {
        #[runtime_builtin(
                                                    name = $name,
                                                    binding_variant = "default",
                                                    builtin_path = "crate::builtins::fea::steps"
                                                )]
        pub async fn $function(id: Value) -> BuiltinResult<Value> {
            crate::builtins::fea::constraints::fixed_outputs(&runmat_builtins::$descriptor, $name)?;
            let id = scalar_string(&id, $name, &ERROR_INPUT)?;
            step_to_object($name, $class, ($build)(id))
        }
    };
}

step_builtin!(
    structural_static_builtin,
    FEA_STRUCTURAL_STEP_STATIC_STEP_DESCRIPTOR,
    "fea.StructuralStep.static",
    FEA_STRUCTURAL_STEP_CLASS,
    ProcedureStep::structural_static
);
step_builtin!(
    structural_modal_builtin,
    FEA_STRUCTURAL_STEP_MODAL_DESCRIPTOR,
    "fea.StructuralStep.modal",
    FEA_STRUCTURAL_STEP_CLASS,
    ProcedureStep::structural_modal
);
step_builtin!(
    structural_harmonic_builtin,
    FEA_STRUCTURAL_STEP_HARMONIC_DESCRIPTOR,
    "fea.StructuralStep.harmonic",
    FEA_STRUCTURAL_STEP_CLASS,
    ProcedureStep::structural_harmonic
);
step_builtin!(
    structural_transient_builtin,
    FEA_STRUCTURAL_STEP_TRANSIENT_DESCRIPTOR,
    "fea.StructuralStep.transient",
    FEA_STRUCTURAL_STEP_CLASS,
    ProcedureStep::structural_transient
);
step_builtin!(
    structural_nonlinear_builtin,
    FEA_STRUCTURAL_STEP_NONLINEAR_DESCRIPTOR,
    "fea.StructuralStep.nonlinear",
    FEA_STRUCTURAL_STEP_CLASS,
    ProcedureStep::structural_nonlinear
);
step_builtin!(
    thermal_steady_builtin,
    FEA_THERMAL_STEP_STEADY_DESCRIPTOR,
    "fea.ThermalStep.steady",
    FEA_THERMAL_STEP_CLASS,
    ProcedureStep::thermal_steady
);
step_builtin!(
    thermal_transient_builtin,
    FEA_THERMAL_STEP_TRANSIENT_DESCRIPTOR,
    "fea.ThermalStep.transient",
    FEA_THERMAL_STEP_CLASS,
    ProcedureStep::thermal_transient
);
step_builtin!(
    electromagnetic_harmonic_builtin,
    FEA_ELECTROMAGNETIC_STEP_HARMONIC_DESCRIPTOR,
    "fea.ElectromagneticStep.harmonic",
    FEA_ELECTROMAGNETIC_STEP_CLASS,
    ProcedureStep::electromagnetic_harmonic
);
step_builtin!(
    acoustic_harmonic_builtin,
    FEA_ACOUSTIC_STEP_HARMONIC_DESCRIPTOR,
    "fea.AcousticStep.harmonic",
    FEA_ACOUSTIC_STEP_CLASS,
    ProcedureStep::acoustic_harmonic
);
step_builtin!(
    fluid_steady_builtin,
    FEA_FLUID_STEP_STEADY_STATE_DESCRIPTOR,
    "fea.FluidStep.steadyState",
    FEA_FLUID_STEP_CLASS,
    ProcedureStep::fluid_static
);
step_builtin!(
    fluid_transient_builtin,
    FEA_FLUID_STEP_TRANSIENT_DESCRIPTOR,
    "fea.FluidStep.transient",
    FEA_FLUID_STEP_CLASS,
    ProcedureStep::fluid_transient
);

macro_rules! coupled_step_builtin {
    ($function:ident, $descriptor:ident, $name:literal, $class:expr, $family:expr, $procedure:expr) => {
        step_builtin!($function, $descriptor, $name, $class, |id| {
            ProcedureStep::coupled(id, $family, $procedure)
        });
    };
}

coupled_step_builtin!(
    thermomechanical_transient_builtin,
    FEA_THERMO_MECHANICAL_STEP_TRANSIENT_DESCRIPTOR,
    "fea.ThermoMechanicalStep.transient",
    FEA_THERMOMECHANICAL_STEP_CLASS,
    CoupledPhysicsFamily::Thermomechanical,
    AnalysisProcedure::TransientResponse
);
coupled_step_builtin!(
    electrothermal_transient_builtin,
    FEA_ELECTRO_THERMAL_STEP_TRANSIENT_DESCRIPTOR,
    "fea.ElectroThermalStep.transient",
    FEA_ELECTROTHERMAL_STEP_CLASS,
    CoupledPhysicsFamily::Electrothermal,
    AnalysisProcedure::TransientResponse
);
coupled_step_builtin!(
    electrothermal_nonlinear_builtin,
    FEA_ELECTRO_THERMAL_STEP_NONLINEAR_DESCRIPTOR,
    "fea.ElectroThermalStep.nonlinear",
    FEA_ELECTROTHERMAL_STEP_CLASS,
    CoupledPhysicsFamily::Electrothermal,
    AnalysisProcedure::Nonlinear
);
coupled_step_builtin!(
    conjugate_heat_transfer_steady_builtin,
    FEA_CONJUGATE_HEAT_TRANSFER_STEP_STEADY_STATE_DESCRIPTOR,
    "fea.ConjugateHeatTransferStep.steadyState",
    FEA_CONJUGATE_HEAT_TRANSFER_STEP_CLASS,
    CoupledPhysicsFamily::ConjugateHeatTransfer,
    AnalysisProcedure::Static
);
coupled_step_builtin!(
    conjugate_heat_transfer_transient_builtin,
    FEA_CONJUGATE_HEAT_TRANSFER_STEP_TRANSIENT_DESCRIPTOR,
    "fea.ConjugateHeatTransferStep.transient",
    FEA_CONJUGATE_HEAT_TRANSFER_STEP_CLASS,
    CoupledPhysicsFamily::ConjugateHeatTransfer,
    AnalysisProcedure::TransientResponse
);
coupled_step_builtin!(
    fluid_structure_transient_builtin,
    FEA_FLUID_STRUCTURE_STEP_TRANSIENT_DESCRIPTOR,
    "fea.FluidStructureStep.transient",
    FEA_FLUID_STRUCTURE_STEP_CLASS,
    CoupledPhysicsFamily::FluidStructureInteraction,
    AnalysisProcedure::TransientResponse
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::fea::test_support::load_stl_fixture;
    use futures::executor::block_on;

    fn step_id(value: &str) -> Value {
        Value::String(value.into())
    }

    #[test]
    fn concrete_step_constructors_cover_every_current_family_and_procedure() {
        let steps = [
            (
                block_on(structural_static_builtin(step_id("structural_static"))).unwrap(),
                FEA_STRUCTURAL_STEP_CLASS,
                AnalysisProcedure::Static,
            ),
            (
                block_on(structural_modal_builtin(step_id("structural_modal"))).unwrap(),
                FEA_STRUCTURAL_STEP_CLASS,
                AnalysisProcedure::ModalExtraction,
            ),
            (
                block_on(structural_harmonic_builtin(step_id("structural_harmonic"))).unwrap(),
                FEA_STRUCTURAL_STEP_CLASS,
                AnalysisProcedure::HarmonicResponse,
            ),
            (
                block_on(structural_transient_builtin(step_id(
                    "structural_transient",
                )))
                .unwrap(),
                FEA_STRUCTURAL_STEP_CLASS,
                AnalysisProcedure::TransientResponse,
            ),
            (
                block_on(structural_nonlinear_builtin(step_id(
                    "structural_nonlinear",
                )))
                .unwrap(),
                FEA_STRUCTURAL_STEP_CLASS,
                AnalysisProcedure::Nonlinear,
            ),
            (
                block_on(thermal_steady_builtin(step_id("thermal_steady"))).unwrap(),
                FEA_THERMAL_STEP_CLASS,
                AnalysisProcedure::Static,
            ),
            (
                block_on(thermal_transient_builtin(step_id("thermal"))).unwrap(),
                FEA_THERMAL_STEP_CLASS,
                AnalysisProcedure::TransientResponse,
            ),
            (
                block_on(electromagnetic_harmonic_builtin(step_id("em"))).unwrap(),
                FEA_ELECTROMAGNETIC_STEP_CLASS,
                AnalysisProcedure::HarmonicResponse,
            ),
            (
                block_on(acoustic_harmonic_builtin(step_id("acoustic"))).unwrap(),
                FEA_ACOUSTIC_STEP_CLASS,
                AnalysisProcedure::HarmonicResponse,
            ),
            (
                block_on(fluid_steady_builtin(step_id("fluid_steady"))).unwrap(),
                FEA_FLUID_STEP_CLASS,
                AnalysisProcedure::Static,
            ),
            (
                block_on(fluid_transient_builtin(step_id("fluid_transient"))).unwrap(),
                FEA_FLUID_STEP_CLASS,
                AnalysisProcedure::TransientResponse,
            ),
            (
                block_on(thermomechanical_transient_builtin(step_id(
                    "thermomechanical",
                )))
                .unwrap(),
                FEA_THERMOMECHANICAL_STEP_CLASS,
                AnalysisProcedure::TransientResponse,
            ),
            (
                block_on(electrothermal_transient_builtin(step_id(
                    "electrothermal_transient",
                )))
                .unwrap(),
                FEA_ELECTROTHERMAL_STEP_CLASS,
                AnalysisProcedure::TransientResponse,
            ),
            (
                block_on(electrothermal_nonlinear_builtin(step_id(
                    "electrothermal_nonlinear",
                )))
                .unwrap(),
                FEA_ELECTROTHERMAL_STEP_CLASS,
                AnalysisProcedure::Nonlinear,
            ),
            (
                block_on(conjugate_heat_transfer_steady_builtin(step_id(
                    "cht_steady",
                )))
                .unwrap(),
                FEA_CONJUGATE_HEAT_TRANSFER_STEP_CLASS,
                AnalysisProcedure::Static,
            ),
            (
                block_on(conjugate_heat_transfer_transient_builtin(step_id(
                    "cht_transient",
                )))
                .unwrap(),
                FEA_CONJUGATE_HEAT_TRANSFER_STEP_CLASS,
                AnalysisProcedure::TransientResponse,
            ),
            (
                block_on(fluid_structure_transient_builtin(step_id("fsi"))).unwrap(),
                FEA_FLUID_STRUCTURE_STEP_CLASS,
                AnalysisProcedure::TransientResponse,
            ),
        ];

        for (value, class_name, procedure) in steps {
            let step: ProcedureStep =
                object_payload("fea.typedStep.test", &value, class_name).unwrap();
            assert_eq!(step.procedure, procedure);
            assert_eq!(step.solution_space, SolutionSpace::Direct);
            require_step_class("fea.typedStep.test", class_name, &step).unwrap();
            let encoded = crate::execution::value_codec::encode_inline_value(&value).unwrap();
            assert_eq!(
                crate::execution::value_codec::decode_inline_value(&encoded).unwrap(),
                value
            );
        }
    }

    #[test]
    fn model_composition_accepts_concrete_steps_and_rejects_mislabeled_steps() {
        let tmp = tempfile::TempDir::new().unwrap();
        let geometry = load_stl_fixture(&tmp.path().join("part.stl"), 1.0);
        let step = block_on(structural_static_builtin(step_id("static"))).unwrap();
        let model = create_model_object_from_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("model".into()),
                geometry.clone(),
                Value::String("Steps".into()),
                step.clone(),
            ]),
        )
        .expect("concrete step should compose");
        let model: AnalysisModel = object_payload(MODEL_NAME, &model, FEA_MODEL_CLASS).unwrap();
        assert_eq!(model.sequence.steps.len(), 1);
        assert_eq!(model.sequence.steps[0].procedure, AnalysisProcedure::Static);

        let Value::Object(mut mislabeled) = step else {
            unreachable!("step constructor returned an object")
        };
        mislabeled = crate::builtins::common::test_support::reclassify_object_unbounded(
            &mislabeled,
            FEA_THERMAL_STEP_CLASS,
        );
        let error = create_model_object_from_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("mislabeled".into()),
                geometry,
                Value::String("Steps".into()),
                Value::Object(mislabeled),
            ]),
        )
        .expect_err("family-mismatched class must reject");
        assert!(error.message().contains("procedure step belongs to"));
    }

    #[test]
    fn step_classes_register_only_their_supported_methods() {
        ensure_fea_classes_registered();
        for (class_name, method) in [
            (FEA_STRUCTURAL_STEP_CLASS, "static"),
            (FEA_STRUCTURAL_STEP_CLASS, "modal"),
            (FEA_STRUCTURAL_STEP_CLASS, "harmonic"),
            (FEA_THERMAL_STEP_CLASS, "transient"),
            (FEA_ELECTROMAGNETIC_STEP_CLASS, "harmonic"),
            (FEA_ACOUSTIC_STEP_CLASS, "harmonic"),
            (FEA_FLUID_STEP_CLASS, "steadyState"),
            (FEA_THERMOMECHANICAL_STEP_CLASS, "transient"),
            (FEA_ELECTROTHERMAL_STEP_CLASS, "nonlinear"),
            (FEA_CONJUGATE_HEAT_TRANSFER_STEP_CLASS, "steadyState"),
            (FEA_FLUID_STRUCTURE_STEP_CLASS, "transient"),
        ] {
            let (registered, owner) = crate::class_registry::lookup_method(
                &runmat_types::ClassIdentity::from(class_name),
                &runmat_types::MethodName::from(method),
            )
            .unwrap_or_else(|| panic!("missing {class_name}.{method}"));
            assert_eq!(owner.display_name(), class_name);
            assert!(registered.is_static);
        }
        assert!(crate::class_registry::lookup_method(
            &runmat_types::ClassIdentity::from(FEA_THERMAL_STEP_CLASS),
            &runmat_types::MethodName::from("static")
        )
        .is_none());
    }
}
