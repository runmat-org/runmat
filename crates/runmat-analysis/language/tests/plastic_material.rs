pub mod support;

use runmat_analysis_core::MaterialPlasticModel;
use runmat_analysis_language::*;
use support::{source_module, TestHost};

fn check(plastic: &str, elastic: &str) -> CheckedFeaProgram {
    let source = format!("material steel {{ {elastic}\n plastic {{ {plastic}\n }}\n }}");
    FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", &source, 1),
        },
    })
}

const ELASTIC: &str = "elastic {\n youngs_modulus = 200 GPa\n poisson_ratio = 0.3\n }";
const LAW: &str = "law = \"j2_isotropic_linear_hardening\"\n yield_stress = 250 MPa\n isotropic_hardening_modulus = 1 GPa";

#[test]
fn authored_j2_lowers_dimensional_parameters_without_inference() {
    let program = check(LAW, ELASTIC);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    assert_eq!(program.materials.len(), 1);
    assert_eq!(
        program.materials.values().next().unwrap().plastic,
        Some(MaterialPlasticModel::J2IsotropicLinearHardening {
            yield_stress_pa: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::YieldStress,
                _,
            >::new(250e6),
            isotropic_hardening_modulus_pa: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::HardeningModulus,
                _,
            >::new(1e9),
        })
    );
    let perfect = check(&LAW.replace("1 GPa", "0 Pa"), ELASTIC);
    assert!(perfect.is_valid(), "{:#?}", perfect.diagnostics);
}

#[test]
fn authored_j2_rejects_proxy_unknown_law_missing_values_and_wrong_dimensions() {
    for plastic in [
        "yield_strain = 0.001\n hardening_modulus_ratio = 0.1\n saturation_exponent = 1",
        &LAW.replace("j2_isotropic_linear_hardening", "j2_kinematic_hardening"),
        &LAW.replace("law = \"j2_isotropic_linear_hardening\"\n ", ""),
        &LAW.replace("250 MPa", "250"),
        &LAW.replace("1 GPa", "1 m"),
        &LAW.replace("250 MPa", "0 Pa"),
        &LAW.replace("1 GPa", "-1 Pa"),
        &format!("{LAW}\n yield_strain = 0.001"),
        &format!("{LAW}\n law = \"j2_isotropic_linear_hardening\""),
        "law = \"j2_isotropic_linear_hardening\"\n yield_stress = 250 MPa",
    ] {
        let program = check(plastic, ELASTIC);
        assert!(!program.is_valid(), "unexpectedly accepted {plastic}");
        assert!(
            program.materials.is_empty(),
            "invalid law must not be published"
        );
    }
    let missing_elasticity = check(LAW, "");
    assert!(!missing_elasticity.is_valid());
    assert!(missing_elasticity.materials.is_empty());
}
