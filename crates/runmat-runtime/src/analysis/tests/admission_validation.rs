//! Pure physical domain validation precedes lease admission.
use super::*;
use crate::analysis::fea_run_admission::{
    FeaRunAdmission, FeaRunDenied, FeaRunPermit, FeaRunRequest,
};

#[derive(Default)]
struct DenyingHost(AtomicU64);
impl FeaRunAdmission for DenyingHost {
    fn admit(&self, _: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(FeaRunDenied::SignInRequired)
    }
}

#[test]
fn malformed_models_keep_domain_diagnostics_before_lease_admission() {
    let _guard = analysis_test_guard();
    let host = Arc::new(DenyingHost::default());
    let _admission = fea_run_admission::replace_fea_run_admission(Some(host.clone())).unwrap();

    let mut heat = sample_model();
    heat.sequence.steps = vec![ProcedureStep::thermal_transient("invalid-heat")];
    heat.signals.clear();
    heat.excitations.clear();
    heat.excitation_cases.clear();
    append_constant_excitation(
        &mut heat,
        "invalid_heat",
        "thermal_core",
        ConstantExcitation::HeatGeneration(f64::INFINITY),
    );
    let error = analysis_run_thermal_with_options_op(
        &heat,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .unwrap_err();
    assert_eq!(error.error_code, "RM.FEA.RUN_THERMAL.SOLVER_MODEL_INVALID");
    assert!(error.message.contains("signal"));

    let mut boundary = sample_model();
    boundary.sequence.steps = vec![ProcedureStep::thermal_transient("invalid-boundary")];
    boundary.constraints.clear();
    boundary.boundaries = vec![runmat_analysis_core::AnalysisBoundary {
        boundary_id: "invalid_convection".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
            "thermal_open_wall",
        )),
        operator: runmat_analysis_core::BoundaryOperator::ThermalConvection {
            ambient_temperature: runmat_analysis_core::PrescribedValue::Constant(Temperature::new(
                293.15,
            )),
            coefficient: runmat_analysis_core::HeatTransferCoefficient::new(f64::NAN),
        },
    }];
    let error = analysis_run_thermal_with_options_op(
        &boundary,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .unwrap_err();
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_THERMAL.INVALID_THERMAL_BOUNDARY"
    );

    let mut voltage = sample_model();
    voltage.sequence.steps = vec![ProcedureStep::structural_transient("invalid-voltage")];
    set_model_electro_coupling(
        &mut voltage,
        TestElectroThermalCoupling {
            reference_temperature_k: Temperature::new(293.15),
            applied_voltage_v: f64::NAN,
            base_electrical_conductivity_s_per_m: 3.5e7,
            resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(4.0e-4),
        },
    );
    let error = analysis_run_transient_with_options_op(
        &voltage,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .unwrap_err();
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_TRANSIENT.SOLVER_MODEL_INVALID"
    );
    assert!(error.message.contains("signal"));
    assert_eq!(
        host.0.load(Ordering::SeqCst),
        0,
        "invalid local domains must not consult billing"
    );

    let mut executed = false;
    let denied = fea_run_admission::entrypoints::run_model(
        &sample_model(),
        AnalysisOperation::LinearStatic,
        &OperationContext::new(None, None),
        || {
            executed = true;
            Ok(())
        },
    )
    .expect_err("valid local domain requires an admitted lease before execution");
    assert_eq!(denied.error_code, "RM.FEA.SignInRequired");
    assert_eq!(host.0.load(Ordering::SeqCst), 1);
    assert!(
        !executed,
        "a denied local lease must prevent numerical execution"
    );
}
