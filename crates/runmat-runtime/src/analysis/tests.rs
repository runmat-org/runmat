use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::{
    fs,
    path::{Path, PathBuf},
};

use chrono::Utc;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, ApiDeviceInfo, GpuTensorHandle, HostTensorOwned,
    HostTensorView,
};
use runmat_analysis_core::{
    AnalysisFieldValues, AnalysisModel, AnalysisModelId, AnalysisStep, AnalysisStepKind,
    BoundaryCondition, BoundaryConditionKind, CfdSolveFamily, ConductivityFrequencyPoint,
    ElectromagneticDomain, EvidenceConfidence, LoadCase, LoadKind, MaterialAssignment,
    MaterialElectricalModel, MaterialMechanicalModel, MaterialModel, MaterialThermalModel,
    ReferenceFrame,
};
use runmat_analysis_fea::{
    fea_acoustic_frequency_response_field_id, fea_cht_energy_residual_field_id,
    fea_cht_fluid_temperature_field_id, fea_cht_interface_heat_flux_field_id,
    fea_cht_interface_temperature_jump_field_id, fea_cht_solid_temperature_field_id,
    fea_electro_thermal_temperature_field_id, fea_electro_thermal_thermal_residual_field_id,
    fea_fsi_coupling_iteration_count_field_id, fea_fsi_fluid_pressure_field_id,
    fea_fsi_fluid_velocity_field_id, fea_fsi_interface_displacement_field_id,
    fea_fsi_interface_pressure_field_id, fea_fsi_interface_residual_field_id,
    fea_fsi_interface_traction_field_id, fea_fsi_structural_displacement_field_id,
    fea_modal_mode_shape_field_id, fea_nonlinear_contact_gap_field_id,
    fea_nonlinear_contact_pressure_field_id, fea_nonlinear_equivalent_plastic_strain_field_id,
    fea_nonlinear_load_factor_field_id, fea_nonlinear_plastic_strain_field_id,
    fea_nonlinear_residual_norm_field_id, fea_nonlinear_von_mises_field_id,
    fea_thermal_boundary_heat_flux_field_id, fea_thermal_heat_flux_field_id,
    fea_thermal_heat_source_field_id, fea_thermal_temperature_gradient_field_id,
    fea_thermo_mechanical_coupling_residual_field_id, fea_thermo_mechanical_displacement_field_id,
    fea_thermo_mechanical_temperature_field_id, fea_thermo_mechanical_thermal_strain_field_id,
    fea_thermo_mechanical_thermal_stress_field_id, fea_thermo_mechanical_von_mises_field_id,
    fea_transient_acceleration_field_id, fea_transient_kinetic_energy_field_id,
    fea_transient_residual_norm_field_id, fea_transient_strain_energy_field_id,
    fea_transient_velocity_field_id, fea_transient_von_mises_field_id, ComputeBackend,
    FeaProgressPhase, FeaProgressStatus, FEA_FIELD_ACOUSTIC_PARTICLE_VELOCITY,
    FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE, FEA_FIELD_ACOUSTIC_PRESSURE_REAL,
    FEA_FIELD_CFD_PRESSURE, FEA_FIELD_CFD_RESIDUAL_CONTINUITY, FEA_FIELD_CFD_RESIDUAL_MOMENTUM,
    FEA_FIELD_CFD_REYNOLDS_NUMBER, FEA_FIELD_CFD_VELOCITY, FEA_FIELD_CFD_VORTICITY,
    FEA_FIELD_CFD_WALL_SHEAR_STRESS, FEA_FIELD_CHT_FLUID_PRESSURE, FEA_FIELD_CHT_FLUID_VELOCITY,
    FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY, FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD,
    FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_POTENTIAL, FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT,
    FEA_FIELD_EM_CURRENT_DENSITY_REAL, FEA_FIELD_EM_ELECTRIC_FIELD_REAL,
    FEA_FIELD_EM_ELECTRIC_FLUX_DENSITY_REAL, FEA_FIELD_EM_ENERGY_DENSITY,
    FEA_FIELD_EM_MAGNETIC_FIELD_REAL, FEA_FIELD_EM_MAGNETIC_FLUX_DENSITY_MAGNITUDE,
    FEA_FIELD_EM_MAGNETIC_FLUX_DENSITY_REAL, FEA_FIELD_EM_POYNTING_VECTOR_REAL,
    FEA_FIELD_EM_RESIDUAL_REAL, FEA_FIELD_EM_VECTOR_POTENTIAL_IMAG,
    FEA_FIELD_EM_VECTOR_POTENTIAL_REAL, FEA_FIELD_MODAL_EIGENVALUE, FEA_FIELD_MODAL_FREQUENCY_HZ,
    FEA_FIELD_MODAL_MODAL_MASS, FEA_FIELD_MODAL_MODAL_STIFFNESS, FEA_FIELD_MODAL_M_ORTHOGONALITY,
    FEA_FIELD_MODAL_PARTICIPATION_FACTOR, FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION,
    FEA_FIELD_MODAL_RESIDUAL_NORM, FEA_FIELD_STRUCTURAL_DISPLACEMENT,
    FEA_FIELD_STRUCTURAL_EQUATION_SCALE, FEA_FIELD_STRUCTURAL_REACTION_FORCE,
    FEA_FIELD_STRUCTURAL_RESIDUAL_NORM, FEA_FIELD_STRUCTURAL_STRAIN, FEA_FIELD_STRUCTURAL_STRESS,
    FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY, FEA_FIELD_STRUCTURAL_VON_MISES,
};
use runmat_geometry_core::{
    GeometryAsset, GeometrySource, MaterialEvidence, MaterialEvidenceConfidence, MeshDescriptor,
    MeshKind, Region, RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh,
    TessellationProfile, UnitSystem,
};

use super::*;

const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";
const SIMPLE_STEP: &str = "ISO-10303-21;\nHEADER;\nFILE_NAME('Assembly_A');\nENDSEC;\nDATA;\n#10=PRODUCT('Bracket_A','',(#1));\nENDSEC;\nEND-ISO-10303-21;\n";

fn analysis_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn sample_analysis_run_prep_context() -> AnalysisRunPrepContext {
    AnalysisRunPrepContext {
        prepared_mesh_count: 1,
        prepared_node_count: 16,
        prepared_element_count: 20,
        mapped_region_count: 3,
        min_scaled_jacobian: 0.86,
        mean_aspect_ratio: 1.5,
        inverted_element_count: 0,
        mapped_load_count: 1,
        mapped_bc_count: 3,
        layout_seed: 29,
        topology_dof_multiplier: 1.2,
        topology_bandwidth_estimate: 4,
        mapped_region_participation_ratio: 0.9,
        topology_surface_patch_ratio: 0.35,
        topology_volume_core_ratio: 0.55,
        topology_mixed_family_ratio: 0.05,
        topology_region_span_mean: 5.0,
        topology_region_block_count: 3,
        topology_region_mesh_mean: 4.0,
        topology_region_mesh_variance: 0.5,
        topology_triangle_family_ratio: 0.2,
        topology_quad_family_ratio: 0.3,
        topology_tet_family_ratio: 0.25,
        topology_hex_family_ratio: 0.25,
        coordinate_span_x_m: 2.4,
        coordinate_span_y_m: 0.6,
        coordinate_span_z_m: 0.4,
        coordinate_active_dimension_count: 3,
        coordinate_characteristic_length_m: 0.2,
        element_geometry_node_count: 4,
        element_geometry_edge_count: 5,
        mean_element_edge_length_m: 0.2,
        mean_element_area_m2: 0.04,
        element_geometry_coverage_ratio: 1.0,
        reference_element_coordinates_m: [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.2, 0.0]],
        reference_element_area_m2: 0.04,
        control_volume_cell_count: 15,
        control_volume_face_count: 19,
        control_volume_internal_face_count: 11,
        control_volume_boundary_face_count: 8,
        control_volume_connectivity_coverage_ratio: 1.0,
    }
}

fn sample_model() -> AnalysisModel {
    AnalysisModel {
        model_id: AnalysisModelId("beam_model".to_string()),
        geometry_id: "geo:beam".to_string(),
        geometry_revision: 1,
        units: UnitSystem::Meter,
        frame: ReferenceFrame::Global,
        materials: vec![MaterialModel {
            material_id: "mat_steel".to_string(),
            name: "Steel".to_string(),
            mechanical: MaterialMechanicalModel {
                youngs_modulus_pa: 200e9,
                poisson_ratio: 0.3,
            },
            thermal: MaterialThermalModel {
                reference_temperature_k: 293.15,
                modulus_temp_coeff_per_k: -2.5e-4,
                ..MaterialThermalModel::default()
            },
            acoustic: None,
            electrical: None,
            plastic: None,
        }],
        material_assignments: Vec::new(),
        thermo_mechanical: None,
        electro_thermal: None,
        electromagnetic: None,
        cfd: None,
        interfaces: Vec::new(),
        boundary_conditions: vec![BoundaryCondition {
            bc_id: "bc_root".to_string(),
            region_id: "root".to_string(),
            kind: BoundaryConditionKind::Fixed,
        }],
        loads: vec![LoadCase {
            load_id: "load_tip".to_string(),
            region_id: "tip".to_string(),
            kind: LoadKind::Force {
                fx: 0.0,
                fy: -1000.0,
                fz: 0.0,
            },
        }],
        steps: vec![AnalysisStep {
            step_id: "step_static".to_string(),
            kind: AnalysisStepKind::Static,
        }],
    }
}

fn sample_model_with_material_assignment_mismatch() -> AnalysisModel {
    let mut model = sample_model();
    model.materials.push(MaterialModel {
        material_id: "mat_polymer".to_string(),
        name: "Polymer".to_string(),
        mechanical: MaterialMechanicalModel {
            youngs_modulus_pa: 3.2e9,
            poisson_ratio: 0.37,
        },
        thermal: MaterialThermalModel {
            reference_temperature_k: 293.15,
            modulus_temp_coeff_per_k: -7.0e-4,
            ..MaterialThermalModel::default()
        },
        acoustic: None,
        electrical: None,
        plastic: None,
    });
    model.material_assignments = vec![MaterialAssignment {
        region_id: "tip".to_string(),
        expected_material_id: "mat_steel".to_string(),
        assigned_material_id: "mat_polymer".to_string(),
        confidence: EvidenceConfidence::Verified,
    }];
    model
}

fn sample_cfd_domain(
    solve_family: CfdSolveFamily,
    enabled: bool,
) -> runmat_analysis_core::CfdDomain {
    runmat_analysis_core::CfdDomain {
        enabled,
        solve_family,
        reference_density_kg_per_m3: 1.225,
        dynamic_viscosity_pa_s: 1.81e-5,
        inlet_velocity_m_per_s: 5.0,
        turbulence_intensity: 0.06,
        time_profile: Vec::new(),
    }
}

fn sample_cfd_boundary_conditions(inlet_velocity_m_per_s: f64) -> Vec<BoundaryCondition> {
    vec![
        BoundaryCondition {
            bc_id: "bc_cfd_inlet".to_string(),
            region_id: "fluid_inlet".to_string(),
            kind: BoundaryConditionKind::CfdInletVelocity {
                velocity_m_per_s: inlet_velocity_m_per_s,
            },
        },
        BoundaryCondition {
            bc_id: "bc_cfd_outlet".to_string(),
            region_id: "fluid_outlet".to_string(),
            kind: BoundaryConditionKind::CfdOutletPressure { pressure_pa: 0.0 },
        },
        BoundaryCondition {
            bc_id: "bc_cfd_wall_upper".to_string(),
            region_id: "fluid_wall_upper".to_string(),
            kind: BoundaryConditionKind::CfdNoSlipWall,
        },
        BoundaryCondition {
            bc_id: "bc_cfd_wall_lower".to_string(),
            region_id: "fluid_wall_lower".to_string(),
            kind: BoundaryConditionKind::CfdNoSlipWall,
        },
    ]
}

fn sample_cht_model() -> AnalysisModel {
    let mut model = sample_model();
    model.steps = vec![
        AnalysisStep {
            step_id: "cht_flow".to_string(),
            kind: AnalysisStepKind::Cfd,
        },
        AnalysisStep {
            step_id: "cht_thermal".to_string(),
            kind: AnalysisStepKind::Thermal,
        },
    ];
    model.cfd = Some(sample_cfd_domain(CfdSolveFamily::Transient, true));
    set_model_thermo_coupling(
        &mut model,
        ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 60.0,
            thermal_expansion_coefficient: 1.2e-5,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: vec![ThermoRegionTemperatureDelta {
                region_id: "tip".to_string(),
                temperature_delta_k: 70.0,
            }],
            time_profile: vec![
                ThermoTimeProfilePoint {
                    normalized_time: 0.0,
                    scale: 0.5,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 1.0,
                    scale: 1.0,
                },
            ],
        },
    );
    model
}

fn sample_fsi_model() -> AnalysisModel {
    let mut model = sample_model();
    model.steps = vec![
        AnalysisStep {
            step_id: "fsi_structure".to_string(),
            kind: AnalysisStepKind::Transient,
        },
        AnalysisStep {
            step_id: "fsi_flow".to_string(),
            kind: AnalysisStepKind::Cfd,
        },
    ];
    model.cfd = Some(runmat_analysis_core::CfdDomain {
        enabled: true,
        solve_family: CfdSolveFamily::Transient,
        reference_density_kg_per_m3: 1.225,
        dynamic_viscosity_pa_s: 1.81e-5,
        inlet_velocity_m_per_s: 4.0,
        turbulence_intensity: 0.06,
        time_profile: vec![
            runmat_analysis_core::CfdTimeProfilePoint {
                normalized_time: 0.0,
                inlet_scale: 0.6,
            },
            runmat_analysis_core::CfdTimeProfilePoint {
                normalized_time: 1.0,
                inlet_scale: 1.0,
            },
        ],
    });
    model
}

fn set_model_thermo_coupling(model: &mut AnalysisModel, coupling: ThermoMechanicalCouplingOptions) {
    model.thermo_mechanical = Some(runmat_analysis_core::ThermoMechanicalDomain {
        enabled: coupling.enabled,
        reference_temperature_k: coupling.reference_temperature_k,
        applied_temperature_delta_k: coupling.applied_temperature_delta_k,
        field_artifact_id: coupling.field_artifact_id,
        field_source: coupling
            .field_source
            .map(|source| runmat_analysis_core::ThermoFieldSource {
                source_id: source.source_id,
                revision: source.revision,
                interpolation_mode: source.interpolation_mode.map(|mode| match mode {
                    ThermoFieldInterpolationMode::Linear => {
                        runmat_analysis_core::ThermoFieldInterpolationMode::Linear
                    }
                    ThermoFieldInterpolationMode::Step => {
                        runmat_analysis_core::ThermoFieldInterpolationMode::Step
                    }
                }),
                expected_region_ids: source.expected_region_ids,
            }),
        region_temperature_deltas: coupling
            .region_temperature_deltas
            .into_iter()
            .map(|delta| runmat_analysis_core::ThermoRegionTemperatureDelta {
                region_id: delta.region_id,
                temperature_delta_k: delta.temperature_delta_k,
            })
            .collect(),
        time_profile: coupling
            .time_profile
            .into_iter()
            .map(|point| runmat_analysis_core::ThermoTimeProfilePoint {
                normalized_time: point.normalized_time,
                scale: point.scale,
            })
            .collect(),
    });
}

fn set_model_electro_coupling(model: &mut AnalysisModel, coupling: ElectroThermalCouplingOptions) {
    for material in &mut model.materials {
        material.electrical = Some(runmat_analysis_core::MaterialElectricalModel {
            reference_temperature_k: coupling.reference_temperature_k,
            conductivity_s_per_m: coupling.base_electrical_conductivity_s_per_m,
            resistive_heating_coefficient: coupling.resistive_heating_coefficient,
            relative_permittivity: 1.0,
            relative_permeability: 1.0,
            conductivity_frequency_response: Vec::new(),
        });
    }
    model.electro_thermal = Some(runmat_analysis_core::ElectroThermalDomain {
        enabled: coupling.enabled,
        reference_temperature_k: coupling.reference_temperature_k,
        applied_voltage_v: coupling.applied_voltage_v,
        region_conductivity_scales: coupling
            .region_conductivity_scales
            .into_iter()
            .map(
                |scale| runmat_analysis_core::ElectroRegionConductivityScale {
                    region_id: scale.region_id,
                    conductivity_scale: scale.conductivity_scale,
                },
            )
            .collect(),
        time_profile: coupling
            .time_profile
            .into_iter()
            .map(|point| runmat_analysis_core::ElectroTimeProfilePoint {
                normalized_time: point.normalized_time,
                current_scale: point.current_scale,
            })
            .collect(),
    });
}

fn set_model_plasticity(model: &mut AnalysisModel, plasticity: PlasticityConstitutiveOptions) {
    if !plasticity.enabled {
        return;
    }
    for material in &mut model.materials {
        material.plastic = Some(runmat_analysis_core::MaterialPlasticModel {
            yield_strain: plasticity.yield_strain,
            hardening_modulus_ratio: plasticity.hardening_modulus_ratio,
            saturation_exponent: plasticity.saturation_exponent,
        });
    }
}

fn set_model_contact(model: &mut AnalysisModel, contact: ContactInterfaceOptions) {
    if !contact.enabled {
        return;
    }
    model.interfaces = vec![runmat_analysis_core::AnalysisInterface {
        interface_id: "contact_1".to_string(),
        primary_region_id: "root".to_string(),
        secondary_region_id: "tip".to_string(),
        kind: runmat_analysis_core::AnalysisInterfaceKind::Contact(
            runmat_analysis_core::ContactInterfaceModel {
                penalty_stiffness_scale: contact.penalty_stiffness_scale,
                max_penetration_ratio: contact.max_penetration_ratio,
                friction_coefficient: contact.friction_coefficient,
            },
        ),
    }];
}

fn sample_geometry_asset() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo:beam".to_string(),
        source: GeometrySource {
            path: "/fixtures/beam.stl".to_string(),
            sha256: "hash-beam".to_string(),
            importer_version: "stl/v1".to_string(),
        },
        source_geometry: SourceGeometry {
            kind: SourceGeometryKind::Mesh,
            assembly: None,
            material_evidence: Vec::new(),
        },
        tessellation_profile: TessellationProfile::default(),
        units: UnitSystem::Meter,
        revision: 2,
        meshes: vec![MeshDescriptor {
            mesh_id: "mesh_1".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 3,
            element_count: 1,
        }],
        surface_meshes: vec![SurfaceMesh::new(
            "mesh_1",
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0, 1, 2]],
        )],
        regions: vec![Region {
            region_id: "region_default".to_string(),
            name: "Default Region".to_string(),
            tag: Some("mesh_default".to_string()),
            cad_ownership: None,
        }],
        region_entity_mappings: vec![RegionEntityMapping::all_faces(
            "region_default",
            "mesh_1",
            1,
        )],
        diagnostics: Vec::new(),
    }
}

fn sample_step_like_geometry_asset() -> GeometryAsset {
    let mut asset = sample_geometry_asset();
    asset.source_geometry.kind = SourceGeometryKind::Cad;
    asset.source_geometry.material_evidence = vec![MaterialEvidence {
        source_key: "STEP:MATERIAL".to_string(),
        normalized_key: "material_name".to_string(),
        value: "Aluminum 6061".to_string(),
        confidence: MaterialEvidenceConfidence::High,
        unit_basis: None,
        assumptions: vec!["imported".to_string()],
    }];
    asset.regions = vec![
        Region {
            region_id: "region_root".to_string(),
            name: "Base_Mount".to_string(),
            tag: Some("fixed".to_string()),
            cad_ownership: None,
        },
        Region {
            region_id: "region_tip".to_string(),
            name: "Tip_Load".to_string(),
            tag: Some("load".to_string()),
            cad_ownership: None,
        },
    ];
    asset.region_entity_mappings = vec![
        RegionEntityMapping::all_faces("region_root", "mesh_1", 1),
        RegionEntityMapping::all_faces("region_tip", "mesh_1", 1),
    ];
    asset
}

fn sample_linear_static_study_spec() -> AnalysisStudySpec {
    AnalysisStudySpec {
        study_id: "study_linear_static_001".to_string(),
        geometry: sample_geometry_asset(),
        create_model_intent: AnalysisCreateModelIntentSpec {
            model_id: "study_model_linear_static_001".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
            prep_context: None,
        },
        model: None,
        run_kind: AnalysisRunKind::LinearStatic,
        backend: ComputeBackend::Cpu,
        linear_static_run_options: None,
        modal_run_options: None,
        acoustic_run_options: None,
        thermal_run_options: None,
        transient_run_options: None,
        cfd_run_options: None,
        cht_run_options: None,
        fsi_run_options: None,
        nonlinear_run_options: None,
        electromagnetic_run_options: None,
    }
}

fn sample_electromagnetic_study_spec() -> AnalysisStudySpec {
    AnalysisStudySpec {
        study_id: "study_electromagnetic_001".to_string(),
        geometry: sample_geometry_asset(),
        create_model_intent: AnalysisCreateModelIntentSpec {
            model_id: "study_model_electromagnetic_001".to_string(),
            profile: AnalysisCreateModelProfile::ElectromagneticStatic,
            prep_context: None,
        },
        model: None,
        run_kind: AnalysisRunKind::Electromagnetic,
        backend: ComputeBackend::Cpu,
        linear_static_run_options: None,
        modal_run_options: None,
        acoustic_run_options: None,
        thermal_run_options: None,
        transient_run_options: None,
        cfd_run_options: None,
        cht_run_options: None,
        fsi_run_options: None,
        nonlinear_run_options: None,
        electromagnetic_run_options: None,
    }
}

#[test]
fn fea_document_resolves_study_geometry_and_run_options() {
    let _guard = analysis_test_guard();
    let tmp = tempfile::tempdir().expect("tempdir should be created");
    fs::write(tmp.path().join("part.stl"), TRIANGLE_STL).expect("fixture geometry should write");
    let input = r#"
version: 1
kind: study
id: bracket_static
geometry:
  path: part.stl
  units: meter
model:
  profile: linear_static_structural
run:
  backend: cpu
  options:
    deterministic_mode: true
    precision_mode: fp64
    preconditioner_mode: jacobi
    quality_policy: strict
"#;

    let resolved = pollster::block_on(parse_and_resolve_fea_document(input, tmp.path()))
        .expect("FEA study document should resolve");

    let FeaResolvedDocument::Study(spec) = resolved else {
        panic!("expected resolved study");
    };
    assert_eq!(spec.study_id, "bracket_static");
    assert_eq!(spec.geometry.units, UnitSystem::Meter);
    assert!(spec.geometry.source.path.ends_with("part.stl"));
    assert_eq!(
        spec.create_model_intent.profile,
        AnalysisCreateModelProfile::LinearStaticStructural
    );
    assert_eq!(spec.create_model_intent.model_id, "bracket_static_model");
    assert_eq!(spec.run_kind, AnalysisRunKind::LinearStatic);
    assert_eq!(spec.backend, ComputeBackend::Cpu);
    assert!(spec.model.is_none());

    let options = spec
        .linear_static_run_options
        .expect("linear static options should parse");
    assert!(options.deterministic_mode);
    assert_eq!(options.precision_mode, PrecisionMode::Fp64);
    assert_eq!(options.preconditioner_mode, PreconditionerMode::Jacobi);
    assert_eq!(options.quality_policy, QualityPolicy::Strict);
}

#[test]
fn fea_document_rejects_legacy_run_kind_profile_mismatch() {
    let _guard = analysis_test_guard();
    let tmp = tempfile::tempdir().expect("tempdir should be created");
    fs::write(tmp.path().join("part.stl"), TRIANGLE_STL).expect("fixture geometry should write");
    let input = r#"
version: 1
kind: study
id: bracket_thermal
geometry:
  path: part.stl
  units: meter
model:
  profile: thermal_standalone
run:
  kind: linear_static
  backend: cpu
"#;

    let err = pollster::block_on(parse_and_resolve_fea_document(input, tmp.path()))
        .expect_err("mismatched legacy run.kind should fail");

    assert!(err.contains("run.kind"));
    assert!(err.contains("model.profile"));
}

#[test]
fn fea_document_resolves_explicit_model_and_sweep() {
    let _guard = analysis_test_guard();
    let tmp = tempfile::tempdir().expect("tempdir should be created");
    fs::write(tmp.path().join("assembly.step"), SIMPLE_STEP)
        .expect("fixture geometry should write");
    let input = r#"
version: 1
kind: sweep
id: bracket_sweep
fail_fast: false
studies:
  - version: 1
    id: bracket_static_a
    geometry:
      path: assembly.step
      units: millimeter
    model:
      id: bracket_model
      profile: linear_static_structural
      defaults: none
      frame: global
    regions:
      bracket:
        selector: "name:Bracket_A"
    materials:
      aluminum:
        name: Aluminum 6061
        mechanical:
          youngs_modulus_pa: 69000000000.0
          poisson_ratio: 0.33
    material_assignments:
      - region: bracket
        material: aluminum
    boundary_conditions:
      - id: fixed_bracket
        region: bracket
        kind: fixed
    loads:
      - id: pressure_load
        region: bracket
        type: pressure
        magnitude_pa: 1200.0
    steps:
      - id: static_step
        kind: static
    run:
      backend: cpu
"#;

    let resolved = pollster::block_on(parse_and_resolve_fea_document(input, tmp.path()))
        .expect("FEA sweep document should resolve");

    let FeaResolvedDocument::Sweep(sweep) = resolved else {
        panic!("expected resolved sweep");
    };
    assert_eq!(sweep.sweep_id, "bracket_sweep");
    assert!(!sweep.fail_fast);
    assert_eq!(sweep.studies.len(), 1);

    let study = &sweep.studies[0];
    assert_eq!(study.study_id, "bracket_static_a");
    assert_eq!(study.create_model_intent.model_id, "bracket_model");
    assert_eq!(study.geometry.units, UnitSystem::Millimeter);
    let model = study.model.as_ref().expect("explicit model should resolve");
    assert_eq!(model.model_id.0, "bracket_model");
    assert_eq!(model.materials.len(), 1);
    assert_eq!(model.material_assignments.len(), 1);
    assert_eq!(model.material_assignments[0].region_id, "region_1");
    assert_eq!(model.boundary_conditions.len(), 1);
    assert_eq!(model.loads.len(), 1);
    assert_eq!(model.steps.len(), 1);
}

#[test]
fn analysis_create_model_returns_v1_envelope() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "model_from_geo".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
            prep_context: None,
        },
        OperationContext::new(Some("trace-create-1".to_string()), None),
    )
    .expect("create model should pass");

    assert_eq!(envelope.operation, "fea.create_model");
    assert_eq!(envelope.op_version, "fea.create_model/v1");
    assert_eq!(envelope.data.model_id.0, "model_from_geo");
    assert_eq!(envelope.data.geometry_id, "geo:beam");
    assert_eq!(envelope.data.geometry_revision, 2);
    assert_eq!(envelope.data.units, UnitSystem::Meter);
    assert_eq!(envelope.data.frame, ReferenceFrame::Global);
    assert!(!envelope.data.materials.is_empty());
    assert!(!envelope.data.boundary_conditions.is_empty());
    assert!(!envelope.data.loads.is_empty());
    assert!(!envelope.data.steps.is_empty());
    assert_eq!(envelope.data.steps[0].kind, AnalysisStepKind::Static);
}

#[test]
fn transient_run_option_presets_are_ordered_for_cost_vs_accuracy() {
    let coarse = AnalysisTransientRunOptions::coarse();
    let balanced = AnalysisTransientRunOptions::balanced();
    let production = AnalysisTransientRunOptions::production_recommended();
    let high_accuracy = AnalysisTransientRunOptions::high_accuracy();

    assert!(coarse.step_count < balanced.step_count);
    assert!(balanced.step_count < high_accuracy.step_count);

    assert!(coarse.tolerance > balanced.tolerance);
    assert!(balanced.tolerance > high_accuracy.tolerance);

    assert!(coarse.time_step_s > balanced.time_step_s);
    assert!(balanced.time_step_s > high_accuracy.time_step_s);

    assert_eq!(production.quality_policy, QualityPolicy::Balanced);
    assert!(production.deterministic_mode);
    assert_eq!(production.precision_mode, PrecisionMode::Fp64);
    assert_eq!(production.dt_bucket_rel_tolerance, 0.01);
}

#[test]
fn modal_run_option_presets_are_ordered_for_cost_vs_accuracy() {
    let coarse = AnalysisModalRunOptions::coarse();
    let balanced = AnalysisModalRunOptions::balanced();
    let high_accuracy = AnalysisModalRunOptions::high_accuracy();

    assert!(coarse.mode_count < balanced.mode_count);
    assert!(balanced.mode_count < high_accuracy.mode_count);
    assert!(coarse.residual_warn_threshold > balanced.residual_warn_threshold);
    assert!(balanced.residual_warn_threshold > high_accuracy.residual_warn_threshold);
}

#[test]
fn nonlinear_run_option_presets_are_ordered_for_cost_vs_accuracy() {
    let coarse = AnalysisNonlinearRunOptions::coarse();
    let balanced = AnalysisNonlinearRunOptions::balanced();
    let production = AnalysisNonlinearRunOptions::production_recommended();
    let high_accuracy = AnalysisNonlinearRunOptions::high_accuracy();

    assert!(coarse.increment_count < balanced.increment_count);
    assert!(balanced.increment_count <= production.increment_count);
    assert!(production.increment_count <= high_accuracy.increment_count);

    assert!(coarse.max_newton_iters < balanced.max_newton_iters);
    assert!(balanced.max_newton_iters <= production.max_newton_iters);
    assert!(production.max_newton_iters <= high_accuracy.max_newton_iters);

    assert!(coarse.tolerance > balanced.tolerance);
    assert!(balanced.tolerance >= production.tolerance);
    assert!(production.tolerance >= high_accuracy.tolerance);

    assert_eq!(production.quality_policy, QualityPolicy::Balanced);
    assert!(production.deterministic_mode);
    assert_eq!(production.precision_mode, PrecisionMode::Fp64);
    assert!(production.line_search);
    assert!(production.max_line_search_backtracks >= balanced.max_line_search_backtracks);
}

#[test]
fn analysis_create_model_maps_invalid_intent_error() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let err = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "   ".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect_err("create model should fail");

    assert_eq!(err.error_code, "RM.FEA.CREATE_MODEL.INVALID_INTENT");
    assert_eq!(err.operation, "fea.create_model");
    assert_eq!(err.op_version, "fea.create_model/v1");
}

#[test]
fn analysis_create_model_supports_nonlinear_profile_template() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "nonlinear_model".to_string(),
            profile: AnalysisCreateModelProfile::NonlinearStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("nonlinear profile should be supported");

    assert_eq!(envelope.data.model_id.0, "nonlinear_model");
    assert_eq!(envelope.data.steps[0].kind, AnalysisStepKind::Nonlinear);
    assert_eq!(
        envelope.data.loads[0].load_id,
        "load_default_nonlinear_force"
    );
}

#[test]
fn analysis_create_model_accepts_prep_context_and_validates_model() {
    let _guard = analysis_test_guard();
    let _prep_guard = crate::geometry::prep_artifact_test_guard();
    let geometry = sample_step_like_geometry_asset();
    let prep = crate::geometry::geometry_prep_for_analysis_op(
        &geometry,
        crate::geometry::GeometryPrepForAnalysisSpec::default(),
        OperationContext::new(None, None),
    )
    .expect("prep for analysis should succeed");

    let created = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "prep_model".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
            prep_context: Some(AnalysisCreateModelPrepContext {
                source_geometry_id: prep.data.prep.provenance.source_geometry_id.clone(),
                source_geometry_revision: prep.data.prep.provenance.source_geometry_revision,
                region_mappings: prep.data.prep.region_mappings.clone(),
            }),
        },
        OperationContext::new(None, None),
    )
    .expect("create model with prep context should succeed");

    analysis_validate(
        &created.data,
        geometry.units,
        &ReferenceFrame::Global,
        OperationContext::new(None, None),
    )
    .expect("prep-aware created model should validate");
    assert_eq!(created.data.boundary_conditions[0].region_id, "region_root");
    assert_eq!(created.data.loads[0].region_id, "region_tip");
    assert!(created
        .data
        .material_assignments
        .iter()
        .all(|assignment| assignment.confidence
            == runmat_analysis_core::EvidenceConfidence::Verified));
}

#[test]
fn analysis_create_model_rejects_mismatched_prep_context() {
    let _guard = analysis_test_guard();
    let geometry = sample_step_like_geometry_asset();
    let error = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "bad_prep_model".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
            prep_context: Some(AnalysisCreateModelPrepContext {
                source_geometry_id: "geo:other".to_string(),
                source_geometry_revision: geometry.revision,
                region_mappings: Vec::new(),
            }),
        },
        OperationContext::new(None, None),
    )
    .expect_err("mismatched prep context should fail");
    assert_eq!(error.error_code, "RM.FEA.CREATE_MODEL.PREP_MISMATCH");
}

#[test]
fn analysis_create_model_supports_transient_profile_template() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "transient_model".to_string(),
            profile: AnalysisCreateModelProfile::TransientStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("transient profile should be supported");

    assert_eq!(envelope.data.model_id.0, "transient_model");
    assert_eq!(envelope.data.steps[0].kind, AnalysisStepKind::Transient);
    assert_eq!(
        envelope.data.loads[0].load_id,
        "load_default_transient_force"
    );
}

#[test]
fn analysis_create_model_supports_modal_profile_template() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("modal profile should be supported");

    assert_eq!(envelope.data.model_id.0, "modal_model");
    assert_eq!(envelope.data.steps[0].kind, AnalysisStepKind::Modal);
    assert_eq!(envelope.data.loads[0].load_id, "load_default_modal_seed");
}

#[test]
fn analysis_create_model_supports_acoustic_harmonic_profile_template() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "acoustic_harmonic_model".to_string(),
            profile: AnalysisCreateModelProfile::AcousticHarmonic,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("acoustic harmonic profile should be supported");

    assert_eq!(envelope.data.model_id.0, "acoustic_harmonic_model");
    assert_eq!(envelope.data.steps[0].kind, AnalysisStepKind::Modal);
    assert_eq!(
        envelope.data.loads[0].load_id,
        "load_default_acoustic_harmonic_seed"
    );
}

#[test]
fn analysis_create_model_supports_electromagnetic_profile_template() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "electromagnetic_profile_model".to_string(),
            profile: AnalysisCreateModelProfile::ElectromagneticStatic,
            prep_context: None,
        },
        OperationContext::new(Some("trace-create-em-profile".to_string()), None),
    )
    .expect("electromagnetic profile model creation should succeed");

    assert_eq!(
        envelope.data.steps[0].kind,
        AnalysisStepKind::Electromagnetic
    );
    let domain = envelope
        .data
        .electromagnetic
        .as_ref()
        .expect("electromagnetic domain should be populated");
    assert!(domain.enabled);
    assert_eq!(domain.reference_frequency_hz, 60.0);
    assert_eq!(domain.applied_current_a, 100.0);
}

#[test]
fn analysis_create_model_supports_cfd_steady_profile_template() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "cfd_steady_model".to_string(),
            profile: AnalysisCreateModelProfile::CfdSteadyState,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("cfd steady profile should be supported");

    assert_eq!(envelope.data.steps[0].kind, AnalysisStepKind::Cfd);
    let cfd = envelope
        .data
        .cfd
        .as_ref()
        .expect("cfd domain should be populated");
    assert_eq!(cfd.solve_family, CfdSolveFamily::SteadyState);
    assert!(cfd.time_profile.is_empty());
}

#[test]
fn analysis_create_model_supports_cfd_transient_profile_template() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "cfd_transient_model".to_string(),
            profile: AnalysisCreateModelProfile::CfdTransient,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("cfd transient profile should be supported");

    assert_eq!(envelope.data.steps[0].kind, AnalysisStepKind::Cfd);
    let cfd = envelope
        .data
        .cfd
        .as_ref()
        .expect("cfd domain should be populated");
    assert_eq!(cfd.solve_family, CfdSolveFamily::Transient);
    assert_eq!(cfd.time_profile.len(), 2);
    assert_eq!(cfd.time_profile[0].normalized_time, 0.0);
    assert_eq!(cfd.time_profile[1].normalized_time, 1.0);
}

#[test]
fn analysis_create_model_supports_cht_coupled_profile_template() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "cht_coupled_model".to_string(),
            profile: AnalysisCreateModelProfile::ChtCoupled,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("cht coupled profile should be supported");

    assert_eq!(envelope.data.steps.len(), 2);
    assert!(envelope
        .data
        .steps
        .iter()
        .any(|step| step.kind == AnalysisStepKind::Cfd));
    assert!(envelope
        .data
        .steps
        .iter()
        .any(|step| step.kind == AnalysisStepKind::Thermal));
    let cfd = envelope
        .data
        .cfd
        .as_ref()
        .expect("cfd domain should be populated for cht profile");
    assert_eq!(cfd.solve_family, CfdSolveFamily::Transient);
    assert_eq!(cfd.time_profile.len(), 2);
    assert!(
        envelope
            .data
            .thermo_mechanical
            .as_ref()
            .expect("thermo-mechanical domain should be populated")
            .enabled
    );
}

#[test]
fn analysis_create_model_supports_fsi_coupled_profile_template() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "fsi_coupled_model".to_string(),
            profile: AnalysisCreateModelProfile::FsiCoupled,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("fsi coupled profile should be supported");

    assert_eq!(envelope.data.steps.len(), 2);
    assert!(envelope
        .data
        .steps
        .iter()
        .any(|step| step.kind == AnalysisStepKind::Transient));
    assert!(envelope
        .data
        .steps
        .iter()
        .any(|step| step.kind == AnalysisStepKind::Cfd));
    assert_eq!(envelope.data.loads[0].load_id, "load_default_fsi_seed");
    let cfd = envelope
        .data
        .cfd
        .as_ref()
        .expect("cfd domain should be populated for fsi profile");
    assert_eq!(cfd.solve_family, CfdSolveFamily::Transient);
    assert_eq!(cfd.time_profile.len(), 2);
    assert!(envelope.data.thermo_mechanical.is_none());
}

#[test]
fn analysis_validate_study_reports_invalid_study_id() {
    let _guard = analysis_test_guard();
    let root = temp_artifact_root("validate-study-evidence");
    let _ = fs::remove_dir_all(&root);
    let env_guard = EnvVarRestoreGuard {
        key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
    };
    std::env::set_var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT", root.display().to_string());
    let mut spec = sample_linear_static_study_spec();
    spec.study_id = "   ".to_string();

    let envelope = analysis_validate_study_op(&spec, OperationContext::new(None, None))
        .expect("study validation should return typed output");

    assert_eq!(envelope.operation, "fea.validate_study");
    assert_eq!(envelope.op_version, "fea.validate_study/v1");
    assert!(!envelope.data.valid);
    assert!(envelope
        .data
        .issue_codes
        .iter()
        .any(|code| code == "RM.FEA.STUDY.ID_EMPTY"));
    assert!(envelope.data.issues.iter().any(|issue| {
        issue.code == "RM.FEA.STUDY.ID_EMPTY"
            && issue.message.contains("study_id must be non-empty")
    }));
    assert!(envelope
        .data
        .evidence_artifact_path
        .ends_with("validate.json"));
    drop(env_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_validate_study_rejects_unused_electromagnetic_options() {
    let _guard = analysis_test_guard();
    let mut spec = sample_linear_static_study_spec();
    spec.electromagnetic_run_options = Some(AnalysisElectromagneticRunOptions::default());

    let envelope = analysis_validate_study_op(&spec, OperationContext::new(None, None))
        .expect("study validation should return typed output");

    assert!(!envelope.data.valid);
    assert!(envelope
        .data
        .issue_codes
        .iter()
        .any(|code| code == "RM.FEA.STUDY.RUN_OPTIONS_KIND_MISMATCH"));
    assert!(envelope.data.issues.iter().any(|issue| {
        issue.code == "RM.FEA.STUDY.RUN_OPTIONS_KIND_MISMATCH"
            && issue.message.contains("solver selected by model.profile")
    }));
}

#[test]
fn analysis_validate_study_rejects_invalid_electromagnetic_options() {
    let _guard = analysis_test_guard();
    let mut spec = sample_electromagnetic_study_spec();
    spec.electromagnetic_run_options = Some(AnalysisElectromagneticRunOptions {
        sweep_enabled: true,
        sweep_frequency_hz: vec![60.0, -10.0],
        residual_target: 0.0,
        harmonic_tolerance: f64::NAN,
        harmonic_max_iterations: 0,
        ..AnalysisElectromagneticRunOptions::default()
    });

    let envelope = analysis_validate_study_op(&spec, OperationContext::new(None, None))
        .expect("study validation should return typed output");

    assert!(!envelope.data.valid);
    assert!(envelope
        .data
        .issue_codes
        .iter()
        .any(|code| code == "RM.FEA.STUDY.ELECTROMAGNETIC_RESIDUAL_TARGET_INVALID"));
    assert!(envelope
        .data
        .issue_codes
        .iter()
        .any(|code| code == "RM.FEA.STUDY.ELECTROMAGNETIC_HARMONIC_TOLERANCE_INVALID"));
    assert!(envelope
        .data
        .issue_codes
        .iter()
        .any(|code| code == "RM.FEA.STUDY.ELECTROMAGNETIC_HARMONIC_MAX_ITERATIONS_INVALID"));
    assert!(envelope
        .data
        .issue_codes
        .iter()
        .any(|code| code == "RM.FEA.STUDY.ELECTROMAGNETIC_SWEEP_FREQUENCY_INVALID"));
    assert!(envelope.data.issues.iter().any(|issue| {
        issue.code == "RM.FEA.STUDY.ELECTROMAGNETIC_SWEEP_FREQUENCY_INVALID"
            && issue
                .message
                .contains("must contain finite positive values")
    }));
}

#[test]
fn analysis_plan_study_returns_canonical_linear_static_sequence() {
    let _guard = analysis_test_guard();
    let root = temp_artifact_root("plan-study-evidence");
    let _ = fs::remove_dir_all(&root);
    let env_guard = EnvVarRestoreGuard {
        key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
    };
    std::env::set_var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT", root.display().to_string());
    let spec = sample_linear_static_study_spec();

    let envelope = analysis_plan_study_op(&spec, OperationContext::new(None, None))
        .expect("study plan should succeed");

    assert_eq!(envelope.operation, "fea.plan_study");
    assert_eq!(envelope.op_version, "fea.plan_study/v1");
    assert_eq!(envelope.data.study_id, spec.study_id);
    assert_eq!(envelope.data.model_id, spec.create_model_intent.model_id);
    assert_eq!(envelope.data.run_operation, "fea.run_linear_static");
    assert_eq!(envelope.data.run_op_version, "fea.run_linear_static/v1");
    assert!(envelope.data.electromagnetic_run_options.is_none());
    assert_eq!(
        envelope.data.operation_sequence,
        vec![
            "fea.create_model/v1".to_string(),
            "fea.validate/v1".to_string(),
            "fea.run_linear_static/v1".to_string(),
        ]
    );
    assert!(envelope.data.study_fingerprint.starts_with("sha256:"));
    assert!(envelope.data.evidence_artifact_path.ends_with("plan.json"));
    drop(env_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_plan_study_surfaces_electromagnetic_run_operation_and_options() {
    let _guard = analysis_test_guard();
    let mut spec = sample_electromagnetic_study_spec();
    spec.electromagnetic_run_options = Some(AnalysisElectromagneticRunOptions {
        sweep_enabled: true,
        sweep_frequency_hz: vec![40.0, 60.0, 120.0],
        ..AnalysisElectromagneticRunOptions::default()
    });

    let envelope = analysis_plan_study_op(&spec, OperationContext::new(None, None))
        .expect("electromagnetic study plan should succeed");

    assert_eq!(envelope.data.run_operation, "fea.run_electromagnetic");
    assert_eq!(envelope.data.run_op_version, "fea.run_electromagnetic/v1");
    assert_eq!(
        envelope.data.electromagnetic_run_options,
        spec.electromagnetic_run_options
    );
}

#[test]
fn analysis_plan_study_sweep_returns_typed_plan_entries() {
    let _guard = analysis_test_guard();
    let root = temp_artifact_root("plan-study-sweep-evidence");
    let _ = fs::remove_dir_all(&root);
    let env_guard = EnvVarRestoreGuard {
        key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
    };
    std::env::set_var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT", root.display().to_string());
    let sweep_spec = AnalysisStudySweepSpec {
        sweep_id: "study_sweep_plan_001".to_string(),
        studies: vec![
            sample_linear_static_study_spec(),
            sample_electromagnetic_study_spec(),
        ],
        fail_fast: true,
    };

    let envelope = analysis_plan_study_sweep_op(&sweep_spec, OperationContext::new(None, None))
        .expect("study sweep plan should succeed");

    assert_eq!(envelope.operation, "fea.plan_study_sweep");
    assert_eq!(envelope.op_version, "fea.plan_study_sweep/v1");
    assert_eq!(envelope.data.sweep_id, "study_sweep_plan_001");
    assert_eq!(envelope.data.study_count, 2);
    assert_eq!(envelope.data.planned_count, 2);
    assert_eq!(envelope.data.failed_count, 0);
    assert!(envelope.data.failure_entries.is_empty());
    assert_eq!(envelope.data.plan_entries.len(), 2);
    assert!(envelope
        .data
        .plan_entries
        .iter()
        .all(|entry| entry.study_fingerprint.starts_with("sha256:")));
    assert!(envelope
        .data
        .plan_entries
        .iter()
        .any(|entry| entry.run_kind == AnalysisRunKind::LinearStatic));
    assert!(envelope
        .data
        .plan_entries
        .iter()
        .any(|entry| entry.run_kind == AnalysisRunKind::Electromagnetic));
    assert!(envelope.data.evidence_artifact_path.ends_with("plan.json"));
    assert!(PathBuf::from(&envelope.data.evidence_artifact_path).exists());
    drop(env_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_plan_study_sweep_rejects_empty_study_set() {
    let _guard = analysis_test_guard();
    let spec = AnalysisStudySweepSpec {
        sweep_id: "study_sweep_plan_empty".to_string(),
        studies: Vec::new(),
        fail_fast: true,
    };

    let err = analysis_plan_study_sweep_op(&spec, OperationContext::new(None, None))
        .expect_err("empty sweep plan should be rejected");
    assert_eq!(err.operation, "fea.plan_study_sweep");
    assert_eq!(err.op_version, "fea.plan_study_sweep/v1");
    assert_eq!(err.error_code, "RM.FEA.PLAN_STUDY_SWEEP.INVALID_SPEC");
}

#[test]
fn analysis_plan_study_sweep_can_continue_on_study_failure() {
    let _guard = analysis_test_guard();
    let mut invalid = sample_linear_static_study_spec();
    invalid.study_id = "   ".to_string();
    let spec = AnalysisStudySweepSpec {
        sweep_id: "study_sweep_plan_continue".to_string(),
        studies: vec![sample_linear_static_study_spec(), invalid],
        fail_fast: false,
    };

    let envelope = analysis_plan_study_sweep_op(&spec, OperationContext::new(None, None))
        .expect("continue-on-failure sweep planning should succeed");

    assert_eq!(envelope.data.study_count, 2);
    assert_eq!(envelope.data.planned_count, 1);
    assert_eq!(envelope.data.failed_count, 1);
    assert_eq!(envelope.data.plan_entries.len(), 1);
    assert_eq!(envelope.data.failure_entries.len(), 1);
    assert_eq!(envelope.data.failure_entries[0].study_index, 1);
    assert_eq!(
        envelope.data.failure_entries[0].error_code,
        "RM.FEA.PLAN_STUDY.INVALID_SPEC"
    );
}

#[test]
fn analysis_run_study_executes_linear_static_path() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("run-study-evidence");
    let _ = fs::remove_dir_all(&root);
    let env_guard = EnvVarRestoreGuard {
        key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
    };
    std::env::set_var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT", root.display().to_string());
    let spec = sample_linear_static_study_spec();

    let envelope = analysis_run_study_op(&spec, OperationContext::new(None, None))
        .expect("study run should succeed");

    assert_eq!(envelope.operation, "fea.run_study");
    assert_eq!(envelope.op_version, "fea.run_study/v1");
    assert_eq!(envelope.data.study_id, spec.study_id);
    assert_eq!(envelope.data.model_id, spec.create_model_intent.model_id);
    assert_eq!(envelope.data.run_kind, AnalysisRunKind::LinearStatic);
    assert_eq!(envelope.data.backend, ComputeBackend::Cpu);
    assert!(envelope.data.electromagnetic_run_options.is_none());
    assert_eq!(envelope.data.run_operation, "fea.run_linear_static");
    assert_eq!(envelope.data.run_op_version, "fea.run_linear_static/v1");
    assert_eq!(
        envelope.data.operation_sequence,
        vec![
            "fea.create_model/v1".to_string(),
            "fea.validate/v1".to_string(),
            "fea.run_linear_static/v1".to_string(),
        ]
    );
    assert!(envelope.data.study_fingerprint.starts_with("sha256:"));
    assert!(envelope.data.run_id.starts_with("run_"));
    assert!(envelope.data.evidence_artifact_path.ends_with("run.json"));

    let persisted = storage::load_run_result(&envelope.data.run_id)
        .expect("run load should succeed")
        .expect("run should be persisted");
    assert_eq!(persisted.run_id, envelope.data.run_id);
    assert_eq!(persisted.run_status, envelope.data.run_status);
    assert_eq!(persisted.publishable, envelope.data.publishable);
    assert_eq!(
        persisted.solver_convergence,
        envelope.data.solver_convergence
    );
    assert_eq!(persisted.result_quality, envelope.data.result_quality);
    assert_eq!(persisted.quality_reasons, envelope.data.quality_reasons);
    assert_eq!(persisted.provenance, envelope.data.provenance);
    drop(env_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_run_study_honors_electromagnetic_run_options() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let mut spec = sample_electromagnetic_study_spec();
    spec.electromagnetic_run_options = Some(AnalysisElectromagneticRunOptions {
        sweep_enabled: true,
        sweep_frequency_hz: vec![30.0, 60.0, 120.0],
        residual_target: 5.0e-7,
        harmonic_tolerance: 1.2345e-4,
        harmonic_max_iterations: 64,
        ..AnalysisElectromagneticRunOptions::default()
    });

    let envelope = analysis_run_study_op(&spec, OperationContext::new(None, None))
        .expect("electromagnetic study run should succeed");

    assert_eq!(envelope.operation, "fea.run_study");
    assert_eq!(envelope.op_version, "fea.run_study/v1");
    assert_eq!(envelope.data.run_kind, AnalysisRunKind::Electromagnetic);
    assert_eq!(envelope.data.backend, ComputeBackend::Cpu);
    assert_eq!(
        envelope.data.electromagnetic_run_options,
        spec.electromagnetic_run_options
    );
    assert_eq!(envelope.data.run_operation, "fea.run_electromagnetic");
    assert_eq!(envelope.data.run_op_version, "fea.run_electromagnetic/v1");

    let persisted = storage::load_run_result(&envelope.data.run_id)
        .expect("run load should succeed")
        .expect("run should be persisted");
    let em_payload = persisted
        .electromagnetic_results
        .as_ref()
        .expect("electromagnetic payload should be present");
    assert_eq!(em_payload.sweep_frequency_hz.len(), 3);
    assert_eq!(em_payload.sweep_peak_flux_density.len(), 3);
    assert_eq!(em_payload.sweep_solve_quality.len(), 3);
    let harmonic_diag = persisted
        .run
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_EM_HARMONIC_COUPLING")
        .expect("harmonic coupling diagnostic should be present");
    assert!(harmonic_diag.message.contains("tolerance=0.00012345"));
    assert!(harmonic_diag.message.contains("iterations="));
    assert_eq!(
        persisted.solver_convergence,
        envelope.data.solver_convergence
    );
    assert_eq!(persisted.result_quality, envelope.data.result_quality);
    assert_eq!(persisted.quality_reasons, envelope.data.quality_reasons);
    assert_eq!(persisted.provenance, envelope.data.provenance);
}

#[test]
fn analysis_run_study_emits_default_electromagnetic_options_when_unspecified() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let spec = sample_electromagnetic_study_spec();

    let envelope = analysis_run_study_op(&spec, OperationContext::new(None, None))
        .expect("electromagnetic study run should succeed");

    assert_eq!(envelope.data.run_kind, AnalysisRunKind::Electromagnetic);
    assert_eq!(
        envelope.data.electromagnetic_run_options,
        Some(AnalysisElectromagneticRunOptions::default())
    );
    assert_eq!(envelope.data.run_operation, "fea.run_electromagnetic");
    assert_eq!(envelope.data.run_op_version, "fea.run_electromagnetic/v1");
}

#[test]
fn analysis_run_study_sweep_executes_multiple_studies() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("run-study-sweep-evidence");
    let _ = fs::remove_dir_all(&root);
    let env_guard = EnvVarRestoreGuard {
        key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
    };
    std::env::set_var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT", root.display().to_string());
    let linear = sample_linear_static_study_spec();
    let electromagnetic = sample_electromagnetic_study_spec();
    let sweep_spec = AnalysisStudySweepSpec {
        sweep_id: "study_sweep_001".to_string(),
        studies: vec![linear, electromagnetic],
        fail_fast: true,
    };

    let envelope = analysis_run_study_sweep_op(&sweep_spec, OperationContext::new(None, None))
        .expect("study sweep should succeed");

    assert_eq!(envelope.operation, "fea.run_study_sweep");
    assert_eq!(envelope.op_version, "fea.run_study_sweep/v1");
    assert_eq!(envelope.data.sweep_id, "study_sweep_001");
    assert_eq!(envelope.data.study_count, 2);
    assert_eq!(envelope.data.success_count, 2);
    assert_eq!(envelope.data.failed_count, 0);
    assert!(envelope.data.failure_entries.is_empty());
    assert_eq!(envelope.data.run_entries.len(), 2);
    assert!(envelope
        .data
        .run_entries
        .iter()
        .any(|entry| entry.run_kind == AnalysisRunKind::LinearStatic));
    assert!(envelope
        .data
        .run_entries
        .iter()
        .any(|entry| entry.run_kind == AnalysisRunKind::Electromagnetic));
    assert!(envelope
        .data
        .run_entries
        .iter()
        .all(|entry| entry.run_id.starts_with("run_")));
    assert!(envelope.data.evidence_artifact_path.ends_with("run.json"));
    drop(env_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_run_study_sweep_rejects_empty_study_set() {
    let _guard = analysis_test_guard();
    let spec = AnalysisStudySweepSpec {
        sweep_id: "study_sweep_empty".to_string(),
        studies: Vec::new(),
        fail_fast: true,
    };

    let err = analysis_run_study_sweep_op(&spec, OperationContext::new(None, None))
        .expect_err("empty sweep should be rejected");
    assert_eq!(err.operation, "fea.run_study_sweep");
    assert_eq!(err.op_version, "fea.run_study_sweep/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_STUDY_SWEEP.INVALID_SPEC");
}

#[test]
fn analysis_run_study_sweep_fail_fast_returns_error_on_invalid_study() {
    let _guard = analysis_test_guard();
    let mut invalid = sample_linear_static_study_spec();
    invalid.study_id = "   ".to_string();
    let spec = AnalysisStudySweepSpec {
        sweep_id: "study_sweep_fail_fast".to_string(),
        studies: vec![sample_linear_static_study_spec(), invalid],
        fail_fast: true,
    };

    let err = analysis_run_study_sweep_op(&spec, OperationContext::new(None, None))
        .expect_err("fail-fast sweep should return error");
    assert_eq!(err.operation, "fea.run_study_sweep");
    assert_eq!(err.op_version, "fea.run_study_sweep/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_STUDY_SWEEP.STUDY_FAILED");
}

#[test]
fn analysis_run_study_sweep_can_continue_on_study_failure() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let mut invalid = sample_linear_static_study_spec();
    invalid.study_id = "   ".to_string();
    let spec = AnalysisStudySweepSpec {
        sweep_id: "study_sweep_continue".to_string(),
        studies: vec![sample_linear_static_study_spec(), invalid],
        fail_fast: false,
    };

    let envelope = analysis_run_study_sweep_op(&spec, OperationContext::new(None, None))
        .expect("continue-on-failure sweep should succeed");

    assert_eq!(envelope.data.study_count, 2);
    assert_eq!(envelope.data.success_count, 1);
    assert_eq!(envelope.data.failed_count, 1);
    assert_eq!(envelope.data.run_entries.len(), 1);
    assert_eq!(envelope.data.failure_entries.len(), 1);
    assert_eq!(envelope.data.failure_entries[0].study_index, 1);
    assert_eq!(
        envelope.data.failure_entries[0].error_code,
        "RM.FEA.RUN_STUDY.INVALID_SPEC"
    );
    assert!(envelope.data.failure_entries[0].study_id.trim().is_empty());
}

#[test]
fn analysis_validate_study_sweep_reports_valid_entries_and_persists_artifact() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("validate-study-sweep-evidence");
    let _ = fs::remove_dir_all(&root);
    let env_guard = EnvVarRestoreGuard {
        key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
    };
    std::env::set_var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT", root.display().to_string());
    let spec = AnalysisStudySweepSpec {
        sweep_id: "study_sweep_validate_001".to_string(),
        studies: vec![
            sample_linear_static_study_spec(),
            sample_electromagnetic_study_spec(),
        ],
        fail_fast: true,
    };

    let envelope = analysis_validate_study_sweep_op(&spec, OperationContext::new(None, None))
        .expect("study sweep validation should succeed");

    assert_eq!(envelope.operation, "fea.validate_study_sweep");
    assert_eq!(envelope.op_version, "fea.validate_study_sweep/v1");
    assert_eq!(envelope.data.sweep_id, "study_sweep_validate_001");
    assert!(envelope.data.valid);
    assert!(envelope.data.issue_codes.is_empty());
    assert_eq!(envelope.data.study_entries.len(), 2);
    assert!(envelope.data.study_entries.iter().all(|entry| entry.valid));
    assert!(envelope
        .data
        .study_entries
        .iter()
        .all(|entry| entry.issue_codes.is_empty() && entry.issues.is_empty()));
    assert!(envelope
        .data
        .evidence_artifact_path
        .ends_with("validate.json"));
    assert!(PathBuf::from(&envelope.data.evidence_artifact_path).exists());

    drop(env_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_validate_study_sweep_reports_sweep_and_study_issue_details() {
    let _guard = analysis_test_guard();
    let mut invalid = sample_linear_static_study_spec();
    invalid.study_id = "   ".to_string();
    let spec = AnalysisStudySweepSpec {
        sweep_id: "   ".to_string(),
        studies: vec![invalid],
        fail_fast: true,
    };

    let envelope = analysis_validate_study_sweep_op(&spec, OperationContext::new(None, None))
        .expect("study sweep validation should return a typed payload");

    assert_eq!(envelope.operation, "fea.validate_study_sweep");
    assert_eq!(envelope.op_version, "fea.validate_study_sweep/v1");
    assert!(!envelope.data.valid);
    assert_eq!(
        envelope.data.issue_codes,
        vec!["RM.FEA.STUDY_SWEEP.ID_EMPTY".to_string()]
    );
    assert_eq!(envelope.data.study_entries.len(), 1);
    assert!(!envelope.data.study_entries[0].valid);
    assert!(envelope.data.study_entries[0]
        .issue_codes
        .iter()
        .any(|code| code == "RM.FEA.STUDY.ID_EMPTY"));
    assert!(envelope.data.study_entries[0]
        .issues
        .iter()
        .any(|issue| issue.code == "RM.FEA.STUDY.ID_EMPTY" && !issue.message.is_empty()));
}

#[test]
fn analysis_create_model_infers_materials_and_assignments_from_geometry_evidence() {
    let _guard = analysis_test_guard();
    let geometry = sample_step_like_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "model_from_step_like".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("create model should succeed");

    assert!(envelope
        .data
        .materials
        .iter()
        .any(|material| material.material_id == "mat_aluminum"));
    assert_eq!(envelope.data.material_assignments.len(), 2);
    assert!(envelope
        .data
        .material_assignments
        .iter()
        .all(|assignment| assignment.assigned_material_id == "mat_aluminum"));
    assert_eq!(
        envelope.data.boundary_conditions[0].region_id,
        "region_root"
    );
    assert_eq!(envelope.data.loads[0].region_id, "region_tip");
}

#[test]
fn analysis_validate_returns_typed_envelope() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let context =
        OperationContext::new(Some("trace-a1".to_string()), Some("request-a1".to_string()));
    let envelope = analysis_validate(&model, UnitSystem::Meter, &ReferenceFrame::Global, context)
        .expect("validation should pass");

    assert_eq!(envelope.operation, "fea.validate");
    assert_eq!(envelope.op_version, "fea.validate/v1");
    assert!(envelope.data.valid);
    assert_eq!(envelope.trace_id.as_deref(), Some("trace-a1"));
}

#[test]
fn analysis_validate_maps_typed_error_code() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.materials.clear();
    let context = OperationContext::new(None, None);
    let error = analysis_validate(&model, UnitSystem::Meter, &ReferenceFrame::Global, context)
        .expect_err("validation should fail");

    assert_eq!(error.error_code, "RM.FEA.VALIDATE.MISSING_MATERIALS");
    assert_eq!(error.operation, "fea.validate");
    assert_eq!(error.op_version, "fea.validate/v1");
}

#[test]
fn analysis_run_linear_static_returns_typed_envelope() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let context =
        OperationContext::new(Some("trace-a2".to_string()), Some("request-a2".to_string()));
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Balanced,
            prep_context: Some(sample_analysis_run_prep_context()),
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        context,
    )
    .expect("run should pass");

    assert_eq!(envelope.operation, "fea.run_linear_static");
    assert_eq!(envelope.op_version, "fea.run_linear_static/v1");
    assert_eq!(envelope.data.run.backend, ComputeBackend::Cpu);
    assert!(!envelope
        .data
        .run
        .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .expect("structural displacement field should be present")
        .is_empty());
    assert_eq!(envelope.data.run_status, RunStatus::Publishable);
    assert!(envelope.data.publishable);
    assert!(envelope.data.modal_results.is_none());
    assert_eq!(envelope.data.solver_convergence, QualityGate::Pass);
    assert!(envelope.data.provenance.deterministic_mode);
    assert_eq!(envelope.data.provenance.precision_mode, "fp64");
    assert_eq!(envelope.data.provenance.solver_method, "matrix_free_pcg");
    assert_eq!(envelope.data.provenance.preconditioner, "jacobi");
    assert_eq!(envelope.data.provenance.quality_policy, "balanced");
    assert_eq!(envelope.data.provenance.solver_device_apply_k_ratio, 0.0);
    assert_eq!(envelope.data.provenance.solver_host_sync_count, 0);
}

#[test]
fn analysis_run_linear_static_with_thermo_mechanical_coupling_reports_fields() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    set_model_thermo_coupling(
        &mut model,
        ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 65.0,
            thermal_expansion_coefficient: 1.2e-5,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: Vec::new(),
            time_profile: Vec::new(),
        },
    );

    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Balanced,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-linear-thermo-fields".to_string()), None),
    )
    .expect("thermo-mechanical linear static run should pass");

    assert!(envelope
        .data
        .run
        .field(&fea_thermo_mechanical_temperature_field_id(0))
        .is_some());
    assert!(envelope
        .data
        .run
        .field(&fea_thermo_mechanical_thermal_strain_field_id(0))
        .is_some());
    assert!(envelope
        .data
        .run
        .field(&fea_thermo_mechanical_thermal_stress_field_id(0))
        .is_some());
    assert!(envelope
        .data
        .run
        .field(&fea_thermo_mechanical_displacement_field_id(0))
        .is_some());
    assert!(envelope
        .data
        .run
        .field(&fea_thermo_mechanical_von_mises_field_id(0))
        .is_some());
    assert!(envelope
        .data
        .run
        .field(&fea_thermo_mechanical_coupling_residual_field_id(0))
        .is_some());

    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("thermo-mechanical linear static results should be queryable");
    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("thermo-mechanical descriptor should be present")
    };
    for field_id in [
        fea_thermo_mechanical_temperature_field_id(0),
        fea_thermo_mechanical_von_mises_field_id(0),
        fea_thermo_mechanical_coupling_residual_field_id(0),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Scalar);
        assert_eq!(descriptor.component_count, None);
    }
    let displacement_descriptor = descriptor(&fea_thermo_mechanical_displacement_field_id(0));
    assert_eq!(displacement_descriptor.kind, AnalysisFieldKind::Vector);
    assert_eq!(displacement_descriptor.component_count, Some(3));
    for field_id in [
        fea_thermo_mechanical_thermal_strain_field_id(0),
        fea_thermo_mechanical_thermal_stress_field_id(0),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Tensor);
        assert_eq!(descriptor.component_count, Some(6));
    }
}

#[test]
fn analysis_run_linear_static_persists_artifacts_through_runtime_filesystem_provider() {
    let _guard = analysis_test_guard();
    let _provider_lock = runmat_filesystem::provider_override_lock();
    storage::reset_artifact_store_for_tests();
    let sandbox_root = temp_artifact_root("artifact-provider");
    let _ = fs::remove_dir_all(&sandbox_root);
    let provider = Arc::new(
        runmat_filesystem::SandboxFsProvider::new(sandbox_root.clone())
            .expect("sandbox filesystem provider should be created"),
    );
    let _provider_guard = runmat_filesystem::replace_provider(provider);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: PathBuf::from("/fea-artifacts"),
    })
    .expect("configure provider-backed filesystem artifact store");

    let model = sample_model();
    let envelope = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-provider-artifact".to_string()), None),
    )
    .expect("run should pass");
    let artifact_path = PathBuf::from("/fea-artifacts")
        .join("runs")
        .join(format!("{}.json", envelope.data.run_id));

    let bytes =
        runmat_filesystem::read(&artifact_path).expect("artifact should be provider-readable");
    assert!(!bytes.is_empty());
    let persisted = storage::load_run_result(&envelope.data.run_id)
        .expect("provider-backed artifact load should succeed")
        .expect("run artifact should exist");
    assert_eq!(persisted.run_id, envelope.data.run_id);
    assert!(sandbox_root
        .join("fea-artifacts")
        .join("runs")
        .join(format!("{}.json", envelope.data.run_id))
        .exists());

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&sandbox_root);
}

#[test]
fn analysis_run_linear_static_cancellation_maps_normalized_error() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let cancelled = Arc::new(AtomicBool::new(true));
    let _interrupt_guard = crate::interrupt::replace_interrupt(Some(cancelled));
    let model = sample_model();

    let err = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-cancel".to_string()), None),
    )
    .expect_err("cancelled run should fail");

    assert_eq!(err.error_code, "RM.FEA.RUN_LINEAR_STATIC.CANCELLED");
    assert_eq!(err.error_type, OperationErrorType::Cancelled);
    assert_eq!(err.severity, OperationErrorSeverity::Warning);
    assert!(!err.retryable);
    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_run_linear_static_emits_solver_and_artifact_progress_events() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_for_handler = Arc::clone(&events);
    let _progress_guard = replace_fea_progress_handler(Some(Arc::new(move |event| {
        events_for_handler
            .lock()
            .expect("progress event lock should not be poisoned")
            .push(event);
    })));
    let model = sample_model();

    analysis_run_linear_static_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-progress".to_string()), None),
    )
    .expect("run should pass");

    let events = events
        .lock()
        .expect("progress event lock should not be poisoned");
    assert!(events.iter().any(|event| {
        event.operation == "fea.run_linear_static"
            && event.phase == FeaProgressPhase::Solve
            && event.status == FeaProgressStatus::Started
    }));
    assert!(events.iter().any(|event| {
        event.operation == "fea.run_linear_static"
            && event.phase == FeaProgressPhase::ArtifactPersistence
            && event.status == FeaProgressStatus::Completed
    }));
    assert!(events.iter().any(|event| {
        event.operation == "fea.run_linear_static"
            && event.phase == FeaProgressPhase::Complete
            && event.status == FeaProgressStatus::Completed
    }));

    storage::reset_artifact_store_for_tests();
}

#[test]
fn gpu_run_without_provider_records_fallback_event() {
    let _guard = analysis_test_guard();
    let _guard = runmat_accelerate_api::ThreadProviderGuard::set(None);
    let model = sample_model();
    let envelope = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Gpu,
        OperationContext::new(None, None),
    )
    .expect("run should pass");

    if envelope.data.provenance.solver_backend == "cpu_reference" {
        assert!(envelope
            .data
            .provenance
            .fallback_events
            .iter()
            .any(|event| event.starts_with("SOLVER_BACKEND_FALLBACK")));
        assert_eq!(envelope.data.provenance.solver_device_apply_k_ratio, 0.0);
        assert!(matches!(
            envelope
                .data
                .run
                .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
                .expect("structural displacement field should be present")
                .values,
            AnalysisFieldValues::HostF64(_)
        ));
    } else {
        assert_eq!(envelope.data.provenance.solver_backend, "runtime_tensor");
    }
}

#[test]
fn gpu_run_with_provider_emits_device_refs() {
    let _guard = analysis_test_guard();
    static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1000);

    struct AnalysisTestProvider;

    impl AccelProvider for AnalysisTestProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Ok(GpuTensorHandle {
                shape: host.shape.to_vec(),
                device_id: 7,
                buffer_id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            })
        }

        fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move {
                Ok(HostTensorOwned {
                    data: vec![0.0; h.shape.iter().product()],
                    shape: h.shape.clone(),
                    storage: runmat_accelerate_api::handle_storage(h),
                })
            })
        }

        fn free(&self, _h: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "analysis-test-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            7
        }

        fn device_info_struct(&self) -> ApiDeviceInfo {
            ApiDeviceInfo {
                device_id: 7,
                name: "analysis-test-provider".to_string(),
                vendor: "runmat-tests".to_string(),
                memory_bytes: None,
                backend: Some("test_gpu".to_string()),
            }
        }
    }

    static PROVIDER: AnalysisTestProvider = AnalysisTestProvider;
    let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));

    let model = sample_model();
    let envelope = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Gpu,
        OperationContext::new(None, None),
    )
    .expect("run should pass");

    assert!(!envelope
        .data
        .provenance
        .fallback_events
        .iter()
        .any(|event| event.starts_with("BACKEND_NO_PROVIDER")
            || event.starts_with("BACKEND_UPLOAD_FAILED")));
    assert!(matches!(
        envelope.data.provenance.solver_backend.as_str(),
        "runtime_tensor" | "cpu_reference"
    ));
    if envelope.data.provenance.solver_backend == "cpu_reference" {
        assert!(envelope
            .data
            .provenance
            .fallback_events
            .iter()
            .any(|event| event.starts_with("SOLVER_BACKEND_FALLBACK")));
    }
    assert!(
        (0.0..=1.0).contains(&envelope.data.provenance.solver_device_apply_k_ratio),
        "ratio must be in [0,1]"
    );
    assert!(matches!(
        envelope
            .data
            .run
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .expect("structural displacement field should be present")
            .values,
        AnalysisFieldValues::DeviceRef(_)
    ));
    assert!(matches!(
        envelope
            .data
            .run
            .field(FEA_FIELD_STRUCTURAL_VON_MISES)
            .expect("structural von Mises field should be present")
            .values,
        AnalysisFieldValues::DeviceRef(_)
    ));
}

#[test]
fn analysis_results_returns_filtered_fields_and_metadata() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let run = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-results-1".to_string()), None),
    )
    .expect("run should pass");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: vec![FEA_FIELD_STRUCTURAL_DISPLACEMENT.to_string()],
            include_field_values: true,

            include_diagnostics: false,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
            include_electromagnetic_results: true,
        },
        OperationContext::new(Some("trace-results-2".to_string()), None),
    )
    .expect("results should pass");

    assert_eq!(results.operation, "fea.results");
    assert_eq!(results.op_version, "fea.results/v1");
    assert_eq!(results.data.fields.len(), 1);
    assert_eq!(
        results.data.fields[0].field_id,
        FEA_FIELD_STRUCTURAL_DISPLACEMENT
    );
    assert!(results.data.diagnostics.is_none());
    assert_eq!(results.data.summary.field_count, 1);
    assert_eq!(results.data.summary.mode_count, 0);
    assert!(results.data.summary.available_mode_indices.is_empty());
    assert_eq!(results.data.summary.min_frequency_hz, None);
    assert_eq!(results.data.summary.max_frequency_hz, None);
    assert_eq!(results.data.summary.max_modal_residual_norm, None);
    assert_eq!(results.data.summary.first_mode_converged, None);
    assert_eq!(results.data.summary.snapshot_count, 0);
    assert_eq!(results.data.summary.time_start_s, None);
    assert_eq!(results.data.summary.time_end_s, None);
    assert_eq!(results.data.summary.max_transient_residual_norm, None);
    assert_eq!(results.data.summary.final_step_converged, None);
}

#[test]
fn analysis_results_describes_structural_l2_fields() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let run = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-results-structural-fields-1".to_string()), None),
    )
    .expect("run should pass");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: vec![
                FEA_FIELD_STRUCTURAL_STRAIN.to_string(),
                FEA_FIELD_STRUCTURAL_STRESS.to_string(),
                FEA_FIELD_STRUCTURAL_REACTION_FORCE.to_string(),
                FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY.to_string(),
                FEA_FIELD_STRUCTURAL_RESIDUAL_NORM.to_string(),
                FEA_FIELD_STRUCTURAL_EQUATION_SCALE.to_string(),
            ],
            include_field_values: false,
            include_diagnostics: false,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
            include_electromagnetic_results: true,
        },
        OperationContext::new(Some("trace-results-structural-fields-2".to_string()), None),
    )
    .expect("results should pass");

    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("descriptor should be present")
    };

    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_STRAIN).kind,
        AnalysisFieldKind::Tensor
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_STRAIN).component_count,
        Some(6)
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_STRESS).kind,
        AnalysisFieldKind::Tensor
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_STRESS).component_count,
        Some(6)
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_REACTION_FORCE).kind,
        AnalysisFieldKind::Vector
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_REACTION_FORCE).component_count,
        Some(3)
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY).kind,
        AnalysisFieldKind::Scalar
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_RESIDUAL_NORM).kind,
        AnalysisFieldKind::Scalar
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_EQUATION_SCALE).kind,
        AnalysisFieldKind::Scalar
    );
}

#[test]
fn analysis_results_unknown_field_maps_typed_error() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let run = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-results-3".to_string()), None),
    )
    .expect("run should pass");

    let err = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: vec!["strain_energy".to_string()],
            include_field_values: true,

            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
            include_electromagnetic_results: true,
        },
        OperationContext::new(Some("trace-results-4".to_string()), None),
    )
    .expect_err("results should fail");

    assert_eq!(err.operation, "fea.results");
    assert_eq!(err.op_version, "fea.results/v1");
    assert_eq!(err.error_code, "RM.FEA.RESULTS.FIELD_NOT_FOUND");
}

#[test]
fn analysis_results_by_run_id_roundtrip_works() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let model = sample_model();
    let run = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-results-by-id-run".to_string()), None),
    )
    .expect("run should pass");

    let fetched = analysis_results_by_run_id_op(
        &run.data.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-results-by-id-fetch".to_string()), None),
    )
    .expect("results by id should pass");

    assert_eq!(fetched.operation, "fea.results");
    assert_eq!(fetched.op_version, "fea.results/v1");
    assert_eq!(fetched.data.summary.field_count, 2);
    assert_eq!(fetched.data.summary.mode_count, 0);
    assert!(fetched.data.summary.available_mode_indices.is_empty());
    assert_eq!(fetched.data.summary.min_frequency_hz, None);
    assert_eq!(fetched.data.summary.max_frequency_hz, None);
    assert_eq!(fetched.data.summary.max_modal_residual_norm, None);
    assert_eq!(fetched.data.summary.first_mode_converged, None);
    assert_eq!(fetched.data.summary.snapshot_count, 0);
    assert_eq!(fetched.data.summary.time_start_s, None);
    assert_eq!(fetched.data.summary.time_end_s, None);
    assert_eq!(fetched.data.summary.max_transient_residual_norm, None);
    assert_eq!(fetched.data.summary.final_step_converged, None);

    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_results_by_run_id_missing_maps_typed_error() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let err = analysis_results_by_run_id_op(
        "run_missing",
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-results-by-id-missing".to_string()), None),
    )
    .expect_err("missing run id should fail");

    assert_eq!(err.error_code, "RM.FEA.RESULTS.RUN_NOT_FOUND");
    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_results_compare_reports_typed_deltas() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();

    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let baseline = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            max_newton_iters: 1,
            line_search: false,
            ..AnalysisNonlinearRunOptions::balanced()
        },
        OperationContext::new(None, None),
    )
    .expect("baseline nonlinear run should succeed");
    let candidate = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions::production_recommended(),
        OperationContext::new(None, None),
    )
    .expect("candidate nonlinear run should succeed");

    let compare = analysis_results_compare_op(
        AnalysisResultsCompareQuery {
            baseline_run_id: baseline.data.run_id.clone(),
            candidate_run_id: candidate.data.run_id.clone(),
        },
        OperationContext::new(None, None),
    )
    .expect("compare operation should succeed");

    assert_eq!(compare.operation, "fea.results_compare");
    assert_eq!(compare.op_version, "fea.results_compare/v1");
    assert!(compare.data.failed_increment_delta.is_some());
    assert!(compare.data.max_iteration_delta.is_some());
    assert!(compare.data.solve_ms_delta.is_some());

    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_trends_summarizes_recent_nonlinear_runs() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();

    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    for _ in 0..4 {
        let _ = analysis_run_nonlinear_op(
            &model,
            ComputeBackend::Cpu,
            OperationContext::new(None, None),
        )
        .expect("nonlinear run should persist for trends");
    }

    let trends = analysis_trends_op(
        AnalysisTrendsQuery { window_size: 3 },
        OperationContext::new(None, None),
    )
    .expect("trends should succeed");

    assert_eq!(trends.operation, "fea.trends");
    assert_eq!(trends.op_version, "fea.trends/v1");
    let nonlinear = trends
        .data
        .summaries
        .iter()
        .find(|summary| summary.run_kind == AnalysisRunKind::Nonlinear)
        .expect("nonlinear trend summary should exist");
    assert_eq!(nonlinear.sample_count, 3);
    assert!(nonlinear.median_solve_ms.is_some());
    assert!(nonlinear.p95_solve_ms.is_some());
    assert!(nonlinear.failed_increment_rate.is_some());
    assert!(nonlinear.thermo_coupling_enabled_rate.is_none());
    assert!(nonlinear.thermo_transient_warn_rate.is_none());
    assert!(nonlinear.thermo_nonlinear_warn_rate.is_none());
    assert!(nonlinear.thermo_spread_breach_rate.is_none());
    assert!(nonlinear.thermo_heterogeneity_breach_rate.is_none());
    assert!(nonlinear.electro_thermal_coupling_enabled_rate.is_none());
    assert!(nonlinear.electro_transient_warn_rate.is_none());
    assert!(nonlinear.electro_nonlinear_warn_rate.is_none());
    assert!(nonlinear.plastic_nonlinear_warn_rate.is_none());
    assert!(nonlinear.contact_nonlinear_warn_rate.is_none());

    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_trends_classifies_acoustic_runs_separately() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();

    let geometry = sample_geometry_asset();
    let acoustic_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "acoustic_trend_model".to_string(),
            profile: AnalysisCreateModelProfile::AcousticHarmonic,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("acoustic model should be created");
    for _ in 0..3 {
        let _ = analysis_run_acoustic_op(
            &acoustic_model.data,
            ComputeBackend::Cpu,
            OperationContext::new(None, None),
        )
        .expect("acoustic run should persist for trends");
    }

    let trends = analysis_trends_op(
        AnalysisTrendsQuery { window_size: 2 },
        OperationContext::new(None, None),
    )
    .expect("trends should succeed");

    let acoustic = trends
        .data
        .summaries
        .iter()
        .find(|summary| summary.run_kind == AnalysisRunKind::Acoustic)
        .expect("acoustic trend summary should exist");
    assert_eq!(acoustic.sample_count, 2);
    assert!(acoustic.median_solve_ms.is_some());
    assert!(acoustic.p95_solve_ms.is_some());

    let modal = trends
        .data
        .summaries
        .iter()
        .find(|summary| summary.run_kind == AnalysisRunKind::Modal);
    assert!(
        modal.is_none(),
        "acoustic runs should not classify as modal"
    );

    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_trends_classifies_cfd_runs_separately() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();

    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "cfd_1".to_string(),
        kind: AnalysisStepKind::Cfd,
    }];
    model.cfd = Some(sample_cfd_domain(CfdSolveFamily::SteadyState, true));
    for _ in 0..3 {
        let _ = analysis_run_cfd_op(
            &model,
            ComputeBackend::Cpu,
            OperationContext::new(None, None),
        )
        .expect("cfd run should persist for trends");
    }

    let trends = analysis_trends_op(
        AnalysisTrendsQuery { window_size: 2 },
        OperationContext::new(None, None),
    )
    .expect("trends should succeed");

    let cfd = trends
        .data
        .summaries
        .iter()
        .find(|summary| summary.run_kind == AnalysisRunKind::Cfd)
        .expect("cfd trend summary should exist");
    assert_eq!(cfd.sample_count, 2);
    assert!(cfd.median_solve_ms.is_some());
    assert!(cfd.p95_solve_ms.is_some());
    assert!(cfd.failed_increment_rate.is_none());

    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_trends_classifies_cht_runs_separately() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();

    let model = sample_cht_model();
    for _ in 0..3 {
        let _ = analysis_run_cht_op(
            &model,
            ComputeBackend::Cpu,
            OperationContext::new(None, None),
        )
        .expect("cht run should persist for trends");
    }

    let trends = analysis_trends_op(
        AnalysisTrendsQuery { window_size: 2 },
        OperationContext::new(None, None),
    )
    .expect("trends should succeed");

    let cht = trends
        .data
        .summaries
        .iter()
        .find(|summary| summary.run_kind == AnalysisRunKind::Cht)
        .expect("cht trend summary should exist");
    assert_eq!(cht.sample_count, 2);
    assert!(cht.median_solve_ms.is_some());
    assert!(cht.p95_solve_ms.is_some());
    assert!(cht.failed_increment_rate.is_none());

    let cfd = trends
        .data
        .summaries
        .iter()
        .find(|summary| summary.run_kind == AnalysisRunKind::Cfd);
    assert!(cfd.is_none(), "cht runs should not classify as cfd");

    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_trends_classifies_fsi_runs_separately() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();

    let model = sample_fsi_model();
    for _ in 0..3 {
        let _ = analysis_run_fsi_op(
            &model,
            ComputeBackend::Cpu,
            OperationContext::new(None, None),
        )
        .expect("fsi run should persist for trends");
    }

    let trends = analysis_trends_op(
        AnalysisTrendsQuery { window_size: 2 },
        OperationContext::new(None, None),
    )
    .expect("trends should succeed");

    let fsi = trends
        .data
        .summaries
        .iter()
        .find(|summary| summary.run_kind == AnalysisRunKind::Fsi)
        .expect("fsi trend summary should exist");
    assert_eq!(fsi.sample_count, 2);
    assert!(fsi.median_solve_ms.is_some());
    assert!(fsi.p95_solve_ms.is_some());
    assert!(fsi.failed_increment_rate.is_none());

    let transient = trends
        .data
        .summaries
        .iter()
        .find(|summary| summary.run_kind == AnalysisRunKind::Transient);
    assert!(
        transient.is_none(),
        "fsi runs should not classify as transient"
    );

    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_results_summary_surfaces_thermo_transient_metrics() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    set_model_thermo_coupling(
        &mut model,
        ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 65.0,
            thermal_expansion_coefficient: 1.2e-5,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: Vec::new(),
            time_profile: Vec::new(),
        },
    );
    set_model_electro_coupling(
        &mut model,
        ElectroThermalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_voltage_v: 36.0,
            base_electrical_conductivity_s_per_m: 3.5e7,
            resistive_heating_coefficient: 4.0e-4,
            region_conductivity_scales: Vec::new(),
            time_profile: Vec::new(),
        },
    );

    let run = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions::default(),
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    assert_eq!(results.data.summary.thermo_coupling_enabled, Some(true));
    assert!(results.data.summary.thermo_coupling_fingerprint.is_some());
    assert!(results
        .data
        .summary
        .thermo_constitutive_temperature_factor
        .is_some());
    assert!(results
        .data
        .summary
        .thermo_effective_modulus_scale
        .is_some());
    assert!(results
        .data
        .summary
        .thermo_constitutive_material_spread_ratio
        .is_some());
    assert!(results
        .data
        .summary
        .thermo_assignment_heterogeneity_index
        .is_some());
    assert!(results.data.summary.thermo_transient_severity.is_some());
    assert!(results.data.summary.thermo_nonlinear_severity.is_none());
    assert_eq!(
        results.data.summary.electro_thermal_coupling_enabled,
        Some(true)
    );
    assert!(results
        .data
        .summary
        .electro_thermal_coupling_fingerprint
        .is_some());
    assert!(results.data.summary.electro_joule_heating_scale.is_some());
    assert!(results
        .data
        .summary
        .electro_conductivity_spread_ratio
        .is_some());
    assert!(results.data.summary.electro_transient_severity.is_some());
    assert!(results.data.summary.electro_nonlinear_severity.is_none());
    assert!(results.data.summary.plastic_nonlinear_severity.is_none());
    assert!(results.data.summary.contact_nonlinear_severity.is_none());
    assert!(run
        .data
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_POTENTIAL)
        .is_some());
    assert!(run
        .data
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD)
        .is_some());
    assert!(run
        .data
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY)
        .is_some());
    assert!(run
        .data
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT)
        .is_some());
    let electric_field = run
        .data
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD)
        .expect("electro-thermal electric field should be present");
    let current_density = run
        .data
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY)
        .expect("electro-thermal current density should be present");
    let joule_heat = run
        .data
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT)
        .expect("electro-thermal Joule heat should be present");
    assert_eq!(electric_field.shape.len(), 2);
    assert_eq!(electric_field.shape[1], 3);
    assert_eq!(current_density.shape, electric_field.shape);
    assert_eq!(joule_heat.shape, vec![electric_field.shape[0]]);
    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("electro-thermal descriptor should be present")
    };
    assert_eq!(
        descriptor(FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_POTENTIAL).kind,
        AnalysisFieldKind::Scalar
    );
    for field_id in [
        FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD,
        FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY,
    ] {
        let descriptor = descriptor(field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Vector);
        assert_eq!(descriptor.component_count, Some(3));
    }
    assert_eq!(
        descriptor(FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT).kind,
        AnalysisFieldKind::Scalar
    );
    assert_eq!(
        descriptor(FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT).component_count,
        None
    );
    let transient = run
        .data
        .transient_results
        .as_ref()
        .expect("transient results should be present");
    assert_eq!(
        transient.electro_thermal_temperature_snapshots.len(),
        transient.time_points_s.len()
    );
    assert_eq!(
        transient.electro_thermal_thermal_residual_snapshots.len(),
        transient.time_points_s.len()
    );
    assert_eq!(
        transient.electro_thermal_temperature_snapshots[0].field_id,
        fea_electro_thermal_temperature_field_id(0)
    );
    assert_eq!(
        transient.electro_thermal_thermal_residual_snapshots[0].field_id,
        fea_electro_thermal_thermal_residual_field_id(0)
    );
    for field_id in [
        fea_electro_thermal_temperature_field_id(0),
        fea_electro_thermal_thermal_residual_field_id(0),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Scalar);
        assert_eq!(descriptor.component_count, None);
    }
}

#[test]
fn analysis_run_transient_rejects_invalid_electro_thermal_voltage() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_invalid_electro_voltage".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    set_model_electro_coupling(
        &mut model,
        ElectroThermalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_voltage_v: f64::NAN,
            base_electrical_conductivity_s_per_m: 3.5e7,
            resistive_heating_coefficient: 4.0e-4,
            region_conductivity_scales: Vec::new(),
            time_profile: Vec::new(),
        },
    );

    let err = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("transient run should reject invalid electro-thermal voltage");

    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_TRANSIENT.INVALID_ELECTRO_THERMAL_OPTIONS"
    );
    assert_eq!(
        err.context.get("applied_voltage_v").map(String::as_str),
        Some("NaN")
    );
}

#[test]
fn analysis_run_transient_rejects_invalid_electro_thermal_conductivity_scale() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_invalid_electro_conductivity".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    set_model_electro_coupling(
        &mut model,
        ElectroThermalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_voltage_v: 36.0,
            base_electrical_conductivity_s_per_m: 3.5e7,
            resistive_heating_coefficient: 4.0e-4,
            region_conductivity_scales: vec![ElectroRegionConductivityScale {
                region_id: "tip".to_string(),
                conductivity_scale: 0.0,
            }],
            time_profile: Vec::new(),
        },
    );

    let err = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("transient run should reject invalid electro-thermal conductivity scale");

    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_TRANSIENT.INVALID_ELECTRO_THERMAL_OPTIONS"
    );
    assert_eq!(
        err.context.get("conductivity_scale").map(String::as_str),
        Some("0")
    );
}

#[test]
fn analysis_run_transient_rejects_unmapped_electro_thermal_region() {
    let _guard = analysis_test_guard();
    let mut model = sample_model_with_material_assignment_mismatch();
    model.steps = vec![AnalysisStep {
        step_id: "transient_unmapped_electro_region".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    set_model_electro_coupling(
        &mut model,
        ElectroThermalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_voltage_v: 36.0,
            base_electrical_conductivity_s_per_m: 3.5e7,
            resistive_heating_coefficient: 4.0e-4,
            region_conductivity_scales: vec![ElectroRegionConductivityScale {
                region_id: "not_a_model_region".to_string(),
                conductivity_scale: 1.0,
            }],
            time_profile: Vec::new(),
        },
    );

    let err = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("transient run should reject unmapped electro-thermal regions");

    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_TRANSIENT.INVALID_ELECTRO_THERMAL_OPTIONS"
    );
    assert_eq!(
        err.context.get("region_id").map(String::as_str),
        Some("not_a_model_region")
    );
}

#[test]
fn analysis_results_summary_surfaces_thermo_nonlinear_metrics() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    set_model_thermo_coupling(
        &mut model,
        ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 80.0,
            thermal_expansion_coefficient: 1.2e-5,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: Vec::new(),
            time_profile: Vec::new(),
        },
    );
    set_model_electro_coupling(
        &mut model,
        ElectroThermalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_voltage_v: 82.0,
            base_electrical_conductivity_s_per_m: 2.6e7,
            resistive_heating_coefficient: 6.0e-4,
            region_conductivity_scales: Vec::new(),
            time_profile: Vec::new(),
        },
    );

    let run = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions::production_recommended(),
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    assert_eq!(results.data.summary.thermo_coupling_enabled, Some(true));
    assert!(results.data.summary.thermo_coupling_fingerprint.is_some());
    assert!(results
        .data
        .summary
        .thermo_constitutive_temperature_factor
        .is_some());
    assert!(results
        .data
        .summary
        .thermo_effective_modulus_scale
        .is_some());
    assert!(results
        .data
        .summary
        .thermo_constitutive_material_spread_ratio
        .is_some());
    assert!(results
        .data
        .summary
        .thermo_assignment_heterogeneity_index
        .is_some());
    assert!(results.data.summary.thermo_nonlinear_severity.is_some());
    assert!(results.data.summary.thermo_transient_severity.is_some());
    assert_eq!(
        results.data.summary.electro_thermal_coupling_enabled,
        Some(true)
    );
    assert!(results
        .data
        .summary
        .electro_thermal_coupling_fingerprint
        .is_some());
    assert!(results.data.summary.electro_joule_heating_scale.is_some());
    assert!(results
        .data
        .summary
        .electro_conductivity_spread_ratio
        .is_some());
    assert!(results.data.summary.electro_nonlinear_severity.is_some());
    assert!(results.data.summary.electro_transient_severity.is_some());
    assert!(results.data.summary.plastic_nonlinear_severity.is_none());
    assert!(results.data.summary.contact_nonlinear_severity.is_none());
    assert!(run
        .data
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_POTENTIAL)
        .is_some());
    assert!(run
        .data
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD)
        .is_some());
    assert!(run
        .data
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY)
        .is_some());
    assert!(run
        .data
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT)
        .is_some());
    let electric_field = run
        .data
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD)
        .expect("electro-thermal electric field should be present");
    let current_density = run
        .data
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY)
        .expect("electro-thermal current density should be present");
    let joule_heat = run
        .data
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT)
        .expect("electro-thermal Joule heat should be present");
    assert_eq!(electric_field.shape.len(), 2);
    assert_eq!(electric_field.shape[1], 3);
    assert_eq!(current_density.shape, electric_field.shape);
    assert_eq!(joule_heat.shape, vec![electric_field.shape[0]]);
    let nonlinear = run
        .data
        .nonlinear_results
        .as_ref()
        .expect("nonlinear results should be present");
    assert_eq!(
        nonlinear.electro_thermal_temperature_snapshots.len(),
        nonlinear.load_factors.len()
    );
    assert_eq!(
        nonlinear.electro_thermal_thermal_residual_snapshots.len(),
        nonlinear.load_factors.len()
    );
    assert_eq!(
        nonlinear.electro_thermal_temperature_snapshots[0].field_id,
        fea_electro_thermal_temperature_field_id(0)
    );
    assert_eq!(
        nonlinear.electro_thermal_thermal_residual_snapshots[0].field_id,
        fea_electro_thermal_thermal_residual_field_id(0)
    );
    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("nonlinear electro-thermal descriptor should be present")
    };
    for field_id in [
        FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_POTENTIAL.to_string(),
        FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT.to_string(),
        fea_electro_thermal_temperature_field_id(0),
        fea_electro_thermal_thermal_residual_field_id(0),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Scalar);
        assert_eq!(descriptor.component_count, None);
    }
    for field_id in [
        FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD,
        FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY,
    ] {
        let descriptor = descriptor(field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Vector);
        assert_eq!(descriptor.component_count, Some(3));
    }
}

#[test]
fn analysis_trends_handles_mixed_schema_and_noisy_samples() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("trends-mixed-schema");
    let _ = fs::remove_dir_all(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .expect("configure filesystem artifact store");

    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let run = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("seed nonlinear run should succeed");

    let run_path = root.join("runs").join(format!("{}.json", run.data.run_id));
    let raw = fs::read_to_string(&run_path).expect("read wrapped artifact");
    let wrapped: serde_json::Value = serde_json::from_str(&raw).expect("parse wrapped artifact");
    let mut legacy = wrapped
        .get("run")
        .cloned()
        .expect("wrapped artifact should have run payload");
    legacy["run_id"] = serde_json::json!(format!("{}_legacy", run.data.run_id));
    fs::write(
        root.join("runs")
            .join(format!("{}_legacy.json", run.data.run_id)),
        serde_json::to_vec_pretty(&legacy).expect("encode legacy artifact"),
    )
    .expect("write legacy artifact");

    let trends = analysis_trends_op(
        AnalysisTrendsQuery { window_size: 8 },
        OperationContext::new(None, None),
    )
    .expect("trends should succeed on mixed schema artifacts");
    let nonlinear = trends
        .data
        .summaries
        .iter()
        .find(|summary| summary.run_kind == AnalysisRunKind::Nonlinear)
        .expect("nonlinear summary should be present");
    assert!(nonlinear.sample_count >= 2);
    assert!(nonlinear.p95_solve_ms.unwrap_or(0.0) >= nonlinear.median_solve_ms.unwrap_or(0.0));

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

fn temp_artifact_root(test_name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "runmat-analysis-tests-{}-{}",
        test_name,
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ))
}

struct EnvVarRestoreGuard {
    key: &'static str,
    previous: Option<String>,
}

impl Drop for EnvVarRestoreGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.as_ref() {
            std::env::set_var(self.key, previous);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

fn scoped_thermo_field_artifact_root(root: &Path) -> EnvVarRestoreGuard {
    const KEY: &str = "RUNMAT_THERMO_FIELD_ARTIFACT_ROOT";
    let previous = std::env::var(KEY).ok();
    std::env::set_var(KEY, root.display().to_string());
    EnvVarRestoreGuard { key: KEY, previous }
}

#[test]
fn analysis_results_by_run_id_legacy_nonlinear_artifacts_remain_loadable() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("legacy-loadable");
    let _ = fs::remove_dir_all(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .expect("configure filesystem artifact store");

    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let run = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should succeed");
    let run_id = run.data.run_id.clone();
    let run_path = root.join("runs").join(format!("{run_id}.json"));

    let mut legacy_value = serde_json::to_value(&run.data).expect("serialize nonlinear run");
    let nonlinear = legacy_value
        .get_mut("nonlinear_results")
        .and_then(|value| value.as_object_mut())
        .expect("nonlinear results should be object");
    nonlinear.remove("increment_norms");
    nonlinear.remove("iteration_counts");
    nonlinear.remove("failed_increments");
    nonlinear.remove("line_search_backtracks");
    nonlinear.remove("max_line_search_backtracks_per_increment");
    nonlinear.remove("tangent_rebuild_count");
    nonlinear.remove("iteration_spike_count");
    nonlinear.remove("convergence_stall_count");
    nonlinear.remove("backtrack_burst_count");
    fs::write(
        &run_path,
        serde_json::to_vec_pretty(&legacy_value).expect("encode legacy artifact"),
    )
    .expect("write legacy artifact");

    let fetched = analysis_results_by_run_id_op(
        &run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("legacy nonlinear artifact should still load");
    assert_eq!(fetched.data.summary.failed_increment_count, Some(0));
    assert_eq!(
        fetched.data.summary.nonlinear_iteration_spike_count,
        Some(0)
    );

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_results_by_run_id_future_artifact_extra_fields_are_ignored() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("future-extra-fields");
    let _ = fs::remove_dir_all(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .expect("configure filesystem artifact store");

    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let run = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should succeed");
    let run_id = run.data.run_id.clone();
    let run_path = root.join("runs").join(format!("{run_id}.json"));

    let mut wrapped = serde_json::json!({
        "schema_version": "analysis_run_artifact/v1",
        "created_at": Utc::now().to_rfc3339(),
        "op_version": "fea.run_nonlinear/v1",
        "run": run.data,
        "future_metadata": {
            "schema_hint": "analysis_run_artifact/v2",
            "opaque": [1, 2, 3]
        }
    });
    wrapped["run"]["nonlinear_results"]["future_spatial_difficulty"] =
        serde_json::json!([0.1, 0.2, 0.3]);
    fs::write(
        &run_path,
        serde_json::to_vec_pretty(&wrapped).expect("encode future artifact"),
    )
    .expect("write future artifact");

    let fetched = analysis_results_by_run_id_op(
        &run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("future nonlinear artifact should still load");
    assert!(fetched.data.summary.increment_count > 0);
    assert!(fetched.data.summary.max_nonlinear_iteration_count.is_some());

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_artifacts_record_family_specific_op_versions_for_coupled_runs() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("family-op-version");
    let _ = fs::remove_dir_all(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .expect("configure filesystem artifact store");

    let mut cfd_model = sample_model();
    cfd_model.steps[0].kind = AnalysisStepKind::Cfd;
    cfd_model.cfd = Some(sample_cfd_domain(CfdSolveFamily::SteadyState, true));
    let cfd = analysis_run_cfd_op(
        &cfd_model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("cfd run should succeed");

    let cht = analysis_run_cht_op(
        &sample_cht_model(),
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("cht run should succeed");

    let fsi = analysis_run_fsi_op(
        &sample_fsi_model(),
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("fsi run should succeed");
    let acoustic_model = analysis_create_model_op(
        &sample_geometry_asset(),
        AnalysisCreateModelIntentSpec {
            model_id: "acoustic_op_version_model".to_string(),
            profile: AnalysisCreateModelProfile::AcousticHarmonic,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("acoustic model should be created");
    let acoustic = analysis_run_acoustic_op(
        &acoustic_model.data,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("acoustic run should succeed");

    let read_op_version = |run_id: &str| -> String {
        let path = root.join("runs").join(format!("{run_id}.json"));
        let raw = fs::read_to_string(path).expect("read wrapped artifact");
        let wrapped: serde_json::Value =
            serde_json::from_str(&raw).expect("parse wrapped artifact");
        wrapped
            .get("op_version")
            .and_then(|value| value.as_str())
            .expect("wrapped artifact should include op_version")
            .to_string()
    };

    assert_eq!(read_op_version(&cfd.data.run_id), "fea.run_cfd/v1");
    assert_eq!(read_op_version(&cht.data.run_id), "fea.run_cht/v1");
    assert_eq!(read_op_version(&fsi.data.run_id), "fea.run_fsi/v1");
    assert_eq!(
        read_op_version(&acoustic.data.run_id),
        "fea.run_acoustic/v1"
    );

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_artifact_retention_prunes_old_runs_per_kind() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("retention-prune");
    let _ = fs::remove_dir_all(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .expect("configure filesystem artifact store");
    std::env::set_var("RUNMAT_FEA_ARTIFACT_MAX_RUNS_PER_KIND", "2");
    std::env::remove_var("RUNMAT_FEA_ARTIFACT_MAX_RUNS");

    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let mut run_ids = Vec::new();
    for _ in 0..5 {
        let run = analysis_run_nonlinear_op(
            &model,
            ComputeBackend::Cpu,
            OperationContext::new(None, None),
        )
        .expect("nonlinear run should succeed");
        run_ids.push(run.data.run_id.clone());
    }

    let run_dir = root.join("runs");
    let kept_files = fs::read_dir(&run_dir)
        .expect("read run dir")
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("json"))
        .count();
    assert!(kept_files <= 2);
    assert!(storage::load_run_result(&run_ids[0])
        .expect("load pruned result")
        .is_none());
    assert!(
        storage::load_run_result(run_ids.last().expect("latest run id"))
            .expect("load latest result")
            .is_some()
    );

    std::env::remove_var("RUNMAT_FEA_ARTIFACT_MAX_RUNS_PER_KIND");
    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_results_by_run_id_filesystem_replay_is_stable() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("filesystem-replay");
    let _ = fs::remove_dir_all(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .expect("configure filesystem artifact store");

    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let run = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should succeed");

    let first = analysis_results_by_run_id_op(
        &run.data.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("load first replay");
    let second = analysis_results_by_run_id_op(
        &run.data.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("load second replay");

    assert_eq!(first.data.summary, second.data.summary);
    assert_eq!(first.data.run_status, second.data.run_status);
    assert_eq!(first.data.publishable, second.data.publishable);
    assert_eq!(first.data.quality_reasons, second.data.quality_reasons);

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn requested_preconditioner_fallback_is_recorded() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Amg,
            quality_policy: QualityPolicy::Balanced,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-preconditioner-fallback".to_string()), None),
    )
    .expect("run should succeed");

    assert_eq!(envelope.data.provenance.preconditioner, "jacobi");
    assert!(envelope
        .data
        .provenance
        .fallback_events
        .iter()
        .any(|event| event.starts_with("SOLVER_PRECONDITIONER_FALLBACK")));
}

#[test]
fn ilu_preconditioner_request_is_honored_without_fallback() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Ilu,
            quality_policy: QualityPolicy::Balanced,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-preconditioner-ilu".to_string()), None),
    )
    .expect("run should succeed");

    assert_eq!(envelope.data.provenance.preconditioner, "ilu0");
    assert!(!envelope
        .data
        .provenance
        .fallback_events
        .iter()
        .any(|event| event.starts_with("SOLVER_PRECONDITIONER_FALLBACK")));
}

#[test]
fn quality_policy_exploratory_allows_publishable_warn_path() {
    let _guard = analysis_test_guard();
    let model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::MultiMaterialAssembly,
    );
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Exploratory,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-quality-policy-exploratory".to_string()), None),
    )
    .expect("run should succeed");

    assert!(envelope.data.publishable);
    assert_eq!(envelope.data.run_status, RunStatus::Publishable);
    assert!(envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::MaterialAssignmentConflict));
    assert_eq!(envelope.data.provenance.quality_policy, "exploratory");
}

#[test]
fn quality_policy_balanced_allows_publishable_with_quality_reasons() {
    let _guard = analysis_test_guard();

    struct UploadFailProvider;

    impl AccelProvider for UploadFailProvider {
        fn upload(&self, _host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Err(anyhow::anyhow!("forced-upload-failure"))
        }

        fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move {
                Ok(HostTensorOwned {
                    data: vec![0.0; h.shape.iter().product()],
                    shape: h.shape.clone(),
                    storage: runmat_accelerate_api::handle_storage(h),
                })
            })
        }

        fn free(&self, _h: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "upload-fail-provider".to_string()
        }
    }

    static PROVIDER: UploadFailProvider = UploadFailProvider;
    let _provider_guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));

    let model = sample_model();
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Gpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Balanced,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-quality-policy-balanced".to_string()), None),
    )
    .expect("run should succeed");

    assert_eq!(envelope.data.solver_convergence, QualityGate::Pass);
    assert_eq!(envelope.data.result_quality, QualityGate::Pass);
    assert!(envelope.data.publishable);
    assert_eq!(envelope.data.run_status, RunStatus::Publishable);
    assert!(envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::FieldPromotionFallback));
    assert_eq!(envelope.data.provenance.quality_policy, "balanced");
}

#[test]
fn quality_policy_strict_rejects_publishable_with_quality_reasons() {
    let _guard = analysis_test_guard();

    struct UploadFailProvider;

    impl AccelProvider for UploadFailProvider {
        fn upload(&self, _host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Err(anyhow::anyhow!("forced-upload-failure"))
        }

        fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move {
                Ok(HostTensorOwned {
                    data: vec![0.0; h.shape.iter().product()],
                    shape: h.shape.clone(),
                    storage: runmat_accelerate_api::handle_storage(h),
                })
            })
        }

        fn free(&self, _h: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "upload-fail-provider".to_string()
        }
    }

    static PROVIDER: UploadFailProvider = UploadFailProvider;
    let _provider_guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));

    let model = sample_model();
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Gpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Strict,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-quality-policy-strict".to_string()), None),
    )
    .expect("run should succeed");

    assert_eq!(envelope.data.solver_convergence, QualityGate::Pass);
    assert_eq!(envelope.data.result_quality, QualityGate::Pass);
    assert!(!envelope.data.publishable);
    assert_eq!(envelope.data.run_status, RunStatus::Degraded);
    assert!(envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::FieldPromotionFallback));
    assert_eq!(envelope.data.provenance.quality_policy, "strict");
}

#[test]
fn analysis_run_modal_rejects_models_without_modal_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_modal_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("modal run should fail for missing modal step");

    assert_eq!(err.operation, "fea.run_modal");
    assert_eq!(err.op_version, "fea.run_modal/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_MODAL.INVALID_MODEL");
}

#[test]
fn analysis_run_acoustic_rejects_models_without_modal_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_acoustic_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("acoustic run should fail for missing acoustic harmonic step marker");

    assert_eq!(err.operation, "fea.run_acoustic");
    assert_eq!(err.op_version, "fea.run_acoustic/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_ACOUSTIC.INVALID_MODEL");
}

#[test]
fn analysis_run_acoustic_rejects_models_without_acoustic_source() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let mut acoustic_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "acoustic_model_missing_source".to_string(),
            profile: AnalysisCreateModelProfile::AcousticHarmonic,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("acoustic model should be created")
    .data;
    acoustic_model.loads = vec![LoadCase {
        load_id: "load_structural_force_not_acoustic_source".to_string(),
        region_id: "region_default".to_string(),
        kind: LoadKind::Force {
            fx: 1.0,
            fy: 0.0,
            fz: 0.0,
        },
    }];

    let err = analysis_run_acoustic_op(
        &acoustic_model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("acoustic run should reject models without acoustic pressure sources");

    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_ACOUSTIC.MISSING_ACOUSTIC_SOURCE"
    );
}

#[test]
fn analysis_run_acoustic_rejects_models_without_acoustic_boundary() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let mut acoustic_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "acoustic_model_missing_boundary".to_string(),
            profile: AnalysisCreateModelProfile::AcousticHarmonic,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("acoustic model should be created")
    .data;
    acoustic_model.boundary_conditions = vec![BoundaryCondition {
        bc_id: "bc_structural_fixed_not_acoustic_boundary".to_string(),
        region_id: "region_default".to_string(),
        kind: BoundaryConditionKind::Fixed,
    }];

    let err = analysis_run_acoustic_op(
        &acoustic_model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("acoustic run should reject models without acoustic boundary conditions");

    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_ACOUSTIC.MISSING_ACOUSTIC_BOUNDARY"
    );
}

#[test]
fn analysis_run_transient_rejects_models_without_transient_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("transient run should fail for missing transient step");

    assert_eq!(err.operation, "fea.run_transient");
    assert_eq!(err.op_version, "fea.run_transient/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_TRANSIENT.INVALID_MODEL");
}

#[test]
fn analysis_run_cfd_rejects_models_without_cfd_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_cfd_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("cfd run should fail for missing cfd step");

    assert_eq!(err.operation, "fea.run_cfd");
    assert_eq!(err.op_version, "fea.run_cfd/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_CFD.INVALID_MODEL");
}

#[test]
fn analysis_run_cfd_rejects_model_without_cfd_domain() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps[0].kind = AnalysisStepKind::Cfd;
    let err = analysis_run_cfd_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("cfd run should fail when cfd domain is missing");

    assert_eq!(err.operation, "fea.run_cfd");
    assert_eq!(err.op_version, "fea.run_cfd/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_CFD.INVALID_MODEL");
}

#[test]
fn analysis_run_cfd_rejects_disabled_cfd_domain() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps[0].kind = AnalysisStepKind::Cfd;
    model.cfd = Some(sample_cfd_domain(CfdSolveFamily::SteadyState, false));
    let err = analysis_run_cfd_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("cfd run should fail for disabled cfd domain");

    assert_eq!(err.operation, "fea.run_cfd");
    assert_eq!(err.op_version, "fea.run_cfd/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_CFD.INVALID_OPTIONS");
}

#[test]
fn analysis_run_thermal_rejects_models_without_thermal_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_thermal_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("thermal run should fail for missing thermal step");

    assert_eq!(err.operation, "fea.run_thermal");
    assert_eq!(err.op_version, "fea.run_thermal/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_THERMAL.INVALID_MODEL");
}

#[test]
fn analysis_run_thermal_rejects_invalid_thermal_material_values() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "thermal_invalid_material".to_string(),
        kind: AnalysisStepKind::Thermal,
    }];
    model.materials[0].thermal.conductivity_w_per_mk = 0.0;

    let err = analysis_run_thermal_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("thermal run should reject nonpositive thermal material values");

    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_THERMAL.INVALID_THERMAL_MATERIAL"
    );
}

#[test]
fn analysis_run_thermal_rejects_invalid_heat_source_values() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "thermal_invalid_source".to_string(),
        kind: AnalysisStepKind::Thermal,
    }];
    model.loads = vec![LoadCase {
        load_id: "load_invalid_heat_source".to_string(),
        region_id: "thermal_core".to_string(),
        kind: LoadKind::HeatSource {
            volumetric_w_per_m3: f64::INFINITY,
        },
    }];

    let err = analysis_run_thermal_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("thermal run should reject non-finite heat source values");

    assert_eq!(err.error_code, "RM.FEA.RUN_THERMAL.INVALID_THERMAL_SOURCE");
}

#[test]
fn analysis_run_thermal_rejects_invalid_thermal_boundary_values() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "thermal_invalid_boundary".to_string(),
        kind: AnalysisStepKind::Thermal,
    }];
    model.boundary_conditions = vec![BoundaryCondition {
        bc_id: "bc_invalid_convection".to_string(),
        region_id: "thermal_open_wall".to_string(),
        kind: BoundaryConditionKind::ThermalConvection {
            ambient_temperature_k: 293.15,
            coefficient_w_per_m2k: f64::NAN,
        },
    }];

    let err = analysis_run_thermal_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("thermal run should reject non-finite thermal boundary values");

    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_THERMAL.INVALID_THERMAL_BOUNDARY"
    );
}

#[test]
fn analysis_run_cht_rejects_models_without_cfd_step() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "thermal_1".to_string(),
        kind: AnalysisStepKind::Thermal,
    }];
    set_model_thermo_coupling(
        &mut model,
        ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 50.0,
            thermal_expansion_coefficient: 1.2e-5,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: Vec::new(),
            time_profile: Vec::new(),
        },
    );
    let err = analysis_run_cht_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("cht run should fail for missing cfd step");

    assert_eq!(err.operation, "fea.run_cht");
    assert_eq!(err.op_version, "fea.run_cht/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_CHT.INVALID_MODEL");
}

#[test]
fn analysis_run_cht_rejects_models_without_thermal_step() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "cfd_1".to_string(),
        kind: AnalysisStepKind::Cfd,
    }];
    model.cfd = Some(sample_cfd_domain(CfdSolveFamily::Transient, true));
    let err = analysis_run_cht_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("cht run should fail for missing thermal step");

    assert_eq!(err.operation, "fea.run_cht");
    assert_eq!(err.op_version, "fea.run_cht/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_CHT.INVALID_MODEL");
}

#[test]
fn analysis_run_cht_rejects_invalid_cfd_domain_parameters() {
    let _guard = analysis_test_guard();
    let mut model = sample_cht_model();
    model
        .cfd
        .as_mut()
        .expect("cht model should include cfd")
        .dynamic_viscosity_pa_s = 0.0;
    let err = analysis_run_cht_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("cht run should fail for invalid cfd domain values");

    assert_eq!(err.operation, "fea.run_cht");
    assert_eq!(err.op_version, "fea.run_cht/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_CHT.INVALID_OPTIONS");
}

#[test]
fn analysis_run_cht_rejects_contact_interface_mapping() {
    let _guard = analysis_test_guard();
    let mut model = sample_cht_model();
    set_model_contact(
        &mut model,
        ContactInterfaceOptions {
            enabled: true,
            penalty_stiffness_scale: 1.0,
            max_penetration_ratio: 0.01,
            friction_coefficient: 0.0,
        },
    );
    let err = analysis_run_cht_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("cht run should fail for contact interface mapping");

    assert_eq!(err.operation, "fea.run_cht");
    assert_eq!(err.op_version, "fea.run_cht/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_CHT.INVALID_INTERFACE_MAPPING");
}

#[test]
fn analysis_run_cht_uses_authored_conjugate_heat_transfer_interface() {
    let _guard = analysis_test_guard();
    let mut model = sample_cht_model();
    model.interfaces = vec![runmat_analysis_core::AnalysisInterface {
        interface_id: "cht_channel_slab_interface".to_string(),
        primary_region_id: "fluid_channel".to_string(),
        secondary_region_id: "solid_slab".to_string(),
        kind: runmat_analysis_core::AnalysisInterfaceKind::ConjugateHeatTransfer(
            runmat_analysis_core::ConjugateHeatTransferInterfaceModel {
                thermal_conductance_w_per_m2k: 750.0,
                contact_resistance_m2k_per_w: 0.0,
                relaxation_factor: 0.5,
            },
        ),
    }];

    let envelope = analysis_run_cht_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisChtRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            time_step_s: 1.0e-3,
            step_count: 4,
            max_linear_iters: 64,
            tolerance: 1.0e-8,
            residual_warn_threshold: 1.0e-4,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(None, None),
    )
    .expect("cht run should accept authored conjugate heat-transfer interface");

    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_CHT_INTERFACE_CLOSURE"
            && diag.message.contains("interface_conductance_w_per_m2k=750")
    }));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_CHT_COUPLING"
            && diag.message.contains("authored_interface_count=1")));
}

#[test]
fn analysis_run_fsi_rejects_models_without_transient_step() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "cfd_1".to_string(),
        kind: AnalysisStepKind::Cfd,
    }];
    model.cfd = Some(sample_cfd_domain(CfdSolveFamily::Transient, true));
    let err = analysis_run_fsi_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("fsi run should fail for missing transient step");

    assert_eq!(err.operation, "fea.run_fsi");
    assert_eq!(err.op_version, "fea.run_fsi/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_FSI.INVALID_MODEL");
}

#[test]
fn analysis_run_fsi_rejects_invalid_cfd_domain_parameters() {
    let _guard = analysis_test_guard();
    let mut model = sample_fsi_model();
    model
        .cfd
        .as_mut()
        .expect("fsi model should include cfd")
        .dynamic_viscosity_pa_s = 0.0;
    let err = analysis_run_fsi_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("fsi run should fail for invalid cfd domain values");

    assert_eq!(err.operation, "fea.run_fsi");
    assert_eq!(err.op_version, "fea.run_fsi/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_FSI.INVALID_OPTIONS");
}

#[test]
fn analysis_run_fsi_rejects_contact_interface_mapping() {
    let _guard = analysis_test_guard();
    let mut model = sample_fsi_model();
    set_model_contact(
        &mut model,
        ContactInterfaceOptions {
            enabled: true,
            penalty_stiffness_scale: 1.0,
            max_penetration_ratio: 0.01,
            friction_coefficient: 0.0,
        },
    );
    let err = analysis_run_fsi_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("fsi run should fail for contact interface mapping");

    assert_eq!(err.operation, "fea.run_fsi");
    assert_eq!(err.op_version, "fea.run_fsi/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_FSI.INVALID_INTERFACE_MAPPING");
}

#[test]
fn analysis_run_fsi_uses_authored_fluid_structure_interface() {
    let _guard = analysis_test_guard();
    let mut model = sample_fsi_model();
    model.interfaces = vec![runmat_analysis_core::AnalysisInterface {
        interface_id: "fsi_pipe_plate_interface".to_string(),
        primary_region_id: "fluid_pipe".to_string(),
        secondary_region_id: "plate_wall".to_string(),
        kind: runmat_analysis_core::AnalysisInterfaceKind::FluidStructure(
            runmat_analysis_core::FluidStructureInterfaceModel {
                normal_stiffness_pa_per_m: 8.0e8,
                damping_ratio: 0.04,
                relaxation_factor: 0.5,
            },
        ),
    }];

    let envelope = analysis_run_fsi_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisFsiRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            time_step_s: 1.0e-3,
            step_count: 4,
            max_linear_iters: 64,
            tolerance: 1.0e-8,
            residual_warn_threshold: 1.0e-4,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(None, None),
    )
    .expect("fsi run should accept authored fluid-structure interface");

    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_FSI_INTERFACE_CLOSURE"
            && diag
                .message
                .contains("interface_stiffness_pa_per_m=800000000")
    }));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_FSI_COUPLING"
            && diag.message.contains("authored_interface_count=1")));
}

#[test]
fn analysis_run_electromagnetic_rejects_models_without_em_step() {
    let model = sample_model();
    let err = analysis_run_electromagnetic_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-em-run-missing-step".to_string()), None),
    )
    .expect_err("electromagnetic run should fail without electromagnetic step");
    assert_eq!(err.operation, "fea.run_electromagnetic");
    assert_eq!(err.op_version, "fea.run_electromagnetic/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_ELECTROMAGNETIC.REQUIRES_STEP");
}

fn configure_valid_em_authoring(model: &mut AnalysisModel) {
    for material in &mut model.materials {
        if material.electrical.is_none() {
            material.electrical = Some(MaterialElectricalModel::default());
        }
    }
    model.electromagnetic = Some(ElectromagneticDomain {
        enabled: true,
        reference_frequency_hz: 60.0,
        applied_current_a: 120.0,
    });
    model.boundary_conditions = vec![
        BoundaryCondition {
            bc_id: "em_ground".to_string(),
            region_id: "em_region".to_string(),
            kind: BoundaryConditionKind::VectorPotentialGround,
        },
        BoundaryCondition {
            bc_id: "em_insulation".to_string(),
            region_id: "em_region".to_string(),
            kind: BoundaryConditionKind::MagneticInsulation,
        },
    ];
    model.loads = vec![LoadCase {
        load_id: "em_source".to_string(),
        region_id: "em_region".to_string(),
        kind: LoadKind::CoilCurrent {
            current_a: 120.0,
            phase_rad: 0.0,
            amplitude_scale: 1.0,
        },
    }];
}

#[test]
fn analysis_run_electromagnetic_rejects_missing_electrical_material() {
    let mut model = sample_model();
    model.steps[0].kind = AnalysisStepKind::Electromagnetic;
    configure_valid_em_authoring(&mut model);
    for material in &mut model.materials {
        material.electrical = None;
    }

    let err = analysis_run_electromagnetic_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-em-run-missing-material".to_string()), None),
    )
    .expect_err("electromagnetic run should reject missing electrical material");
    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_ELECTROMAGNETIC.MISSING_ELECTROMAGNETIC_MATERIAL"
    );
}

#[test]
fn analysis_run_electromagnetic_rejects_missing_current_source() {
    let mut model = sample_model();
    model.steps[0].kind = AnalysisStepKind::Electromagnetic;
    configure_valid_em_authoring(&mut model);
    model.loads = vec![LoadCase {
        load_id: "structural_force".to_string(),
        region_id: "em_region".to_string(),
        kind: LoadKind::Force {
            fx: 1.0,
            fy: 0.0,
            fz: 0.0,
        },
    }];

    let err = analysis_run_electromagnetic_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-em-run-missing-source".to_string()), None),
    )
    .expect_err("electromagnetic run should reject missing current source");
    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_ELECTROMAGNETIC.MISSING_ELECTROMAGNETIC_SOURCE"
    );
}

#[test]
fn analysis_run_electromagnetic_rejects_missing_em_boundary() {
    let mut model = sample_model();
    model.steps[0].kind = AnalysisStepKind::Electromagnetic;
    configure_valid_em_authoring(&mut model);
    model.boundary_conditions = vec![BoundaryCondition {
        bc_id: "structural_fixed".to_string(),
        region_id: "em_region".to_string(),
        kind: BoundaryConditionKind::Fixed,
    }];

    let err = analysis_run_electromagnetic_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-em-run-missing-boundary".to_string()), None),
    )
    .expect_err("electromagnetic run should reject missing EM boundary");
    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_ELECTROMAGNETIC.MISSING_ELECTROMAGNETIC_BOUNDARY"
    );
}

#[test]
fn analysis_run_electromagnetic_static_contract_emits_typed_payload() {
    let mut model = sample_model();
    model.steps[0].kind = AnalysisStepKind::Electromagnetic;
    model.materials[0].electrical = Some(MaterialElectricalModel {
        reference_temperature_k: 293.15,
        conductivity_s_per_m: 5.8e7,
        resistive_heating_coefficient: 0.0039,
        relative_permittivity: 3.2,
        relative_permeability: 1.8,
        conductivity_frequency_response: vec![
            ConductivityFrequencyPoint {
                frequency_hz: 40.0,
                conductivity_scale: 1.06,
                dispersive_loss_scale: Some(0.02),
                relative_permittivity_scale: Some(1.04),
                relative_permeability_scale: Some(1.02),
            },
            ConductivityFrequencyPoint {
                frequency_hz: 60.0,
                conductivity_scale: 1.0,
                dispersive_loss_scale: Some(0.03),
                relative_permittivity_scale: Some(1.0),
                relative_permeability_scale: Some(1.0),
            },
            ConductivityFrequencyPoint {
                frequency_hz: 240.0,
                conductivity_scale: 0.91,
                dispersive_loss_scale: Some(0.04),
                relative_permittivity_scale: Some(0.96),
                relative_permeability_scale: Some(0.98),
            },
        ],
    });
    model.electromagnetic = Some(ElectromagneticDomain {
        enabled: true,
        reference_frequency_hz: 60.0,
        applied_current_a: 120.0,
    });
    configure_valid_em_authoring(&mut model);
    let envelope = analysis_run_electromagnetic_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-em-run-static".to_string()), None),
    )
    .expect("electromagnetic run should return static EM payload");
    assert_eq!(envelope.operation, "fea.run_electromagnetic");
    assert_eq!(envelope.op_version, "fea.run_electromagnetic/v1");
    assert_ne!(envelope.data.run_status, RunStatus::Rejected);
    assert_eq!(
        envelope.data.run.solver_method,
        "electromagnetic_edge_curl_curl_harmonic_block_bicgstab"
    );
    assert!(envelope.data.electromagnetic_results.is_some());
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_EM_STATIC"));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_EM_HARMONIC_COUPLING"
            && diag.message.contains("formulation=edge_curl_curl_harmonic")
            && diag.message.contains("edge_dof_count=")
    }));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_EM_MAXWELL_EDGE_TOPOLOGY"
            && diag.message.contains("incidence_element_count=")
            && diag.message.contains("incidence_orientation_count=")
            && diag
                .message
                .contains("incidence_operator_pair_coverage_ratio=")
    }));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_EM_FORMULATION"
            && diag
                .message
                .contains("formulation_family=frequency_domain_maxwell")
            && diag
                .message
                .contains("active_formulation=full_wave_harmonic")
            && diag
                .message
                .contains("includes_magnetostatic_curl_curl=true")
            && diag
                .message
                .contains("includes_magnetoquasistatic_eddy_current=true")
            && diag
                .message
                .contains("includes_full_wave_displacement_current=true")
            && diag.message.contains("formulation_coverage_ratio=1")
            && diag
                .message
                .contains("material_frequency_response_coverage_ratio=")
    }));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_EM_SOURCE_ENERGY"
            && diag.message.contains("source_region_coverage_ratio=")
            && diag
                .message
                .contains("source_region_energy_consistency_ratio=")
            && diag.message.contains("energy_imbalance_ratio=")
            && diag.message.contains("boundary_energy_ratio=")
    }));
    let em_cost_diag = envelope
        .data
        .run
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_EM_COST")
        .expect("EM cost diagnostic must be present");
    assert!(em_cost_diag.message.contains("prepared_build_ms="));
    assert!(em_cost_diag.message.contains("solve_ms="));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_EM_SWEEP"));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_EM_KNOWN_ANSWER"
            && diag.message.contains("basis=homogeneous_current_line")
            && diag
                .message
                .contains("homogeneous_material_residual_ratio=")
            && diag
                .message
                .contains("source_energy_consistency_residual_ratio=")
            && diag.message.contains("gauge_anchor_residual_ratio=")
            && diag.message.contains("known_answer_coverage_ratio=")
    }));
    let em_payload = envelope
        .data
        .electromagnetic_results
        .as_ref()
        .expect("electromagnetic payload expected");
    assert_eq!(em_payload.sweep_frequency_hz.len(), 1);
    assert_eq!(em_payload.sweep_peak_flux_density.len(), 1);
    assert_eq!(em_payload.sweep_solve_quality.len(), 1);
    assert!(em_payload.resonance_peak_frequency_hz.is_some());
    assert!(em_payload.resonance_peak_flux_density.is_some());
    assert_eq!(
        em_payload.vector_potential_real.field_id,
        FEA_FIELD_EM_VECTOR_POTENTIAL_REAL
    );
    assert_eq!(
        em_payload.vector_potential_imag.field_id,
        FEA_FIELD_EM_VECTOR_POTENTIAL_IMAG
    );
    assert_eq!(em_payload.vector_potential_real.shape.len(), 2);
    assert_eq!(em_payload.vector_potential_real.shape[1], 3);
    assert_eq!(
        em_payload.vector_potential_imag.shape,
        em_payload.vector_potential_real.shape
    );
    match &em_payload.vector_potential_real.values {
        AnalysisFieldValues::HostF64(values) => {
            assert_eq!(
                values.len(),
                em_payload
                    .vector_potential_real
                    .shape
                    .iter()
                    .product::<usize>()
            );
        }
        AnalysisFieldValues::DeviceRef(_) => {
            panic!("CPU EM vector potential should be returned as host values")
        }
    }
    assert_eq!(
        em_payload.magnetic_flux_density_real.field_id,
        FEA_FIELD_EM_MAGNETIC_FLUX_DENSITY_REAL
    );
    assert_eq!(
        em_payload.current_density_real.field_id,
        FEA_FIELD_EM_CURRENT_DENSITY_REAL
    );
    assert_eq!(
        em_payload.electric_field_real.field_id,
        FEA_FIELD_EM_ELECTRIC_FIELD_REAL
    );
    assert_eq!(
        em_payload.energy_density.field_id,
        FEA_FIELD_EM_ENERGY_DENSITY
    );
    assert_eq!(
        em_payload.residual_real.field_id,
        FEA_FIELD_EM_RESIDUAL_REAL
    );
    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("em results should be queryable");
    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("EM descriptor should be present")
    };
    for field_id in [
        FEA_FIELD_EM_VECTOR_POTENTIAL_IMAG,
        FEA_FIELD_EM_MAGNETIC_FLUX_DENSITY_REAL,
        FEA_FIELD_EM_MAGNETIC_FIELD_REAL,
        FEA_FIELD_EM_CURRENT_DENSITY_REAL,
        FEA_FIELD_EM_ELECTRIC_FIELD_REAL,
        FEA_FIELD_EM_ELECTRIC_FLUX_DENSITY_REAL,
        FEA_FIELD_EM_POYNTING_VECTOR_REAL,
    ] {
        let descriptor = descriptor(field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Vector);
        assert_eq!(descriptor.component_count, Some(3));
    }
    assert_eq!(
        descriptor(FEA_FIELD_EM_MAGNETIC_FLUX_DENSITY_MAGNITUDE).kind,
        AnalysisFieldKind::Scalar
    );
    assert_eq!(
        descriptor(FEA_FIELD_EM_MAGNETIC_FLUX_DENSITY_MAGNITUDE).component_count,
        None
    );
    assert_eq!(
        descriptor(FEA_FIELD_EM_ENERGY_DENSITY).kind,
        AnalysisFieldKind::Scalar
    );
    assert_eq!(
        descriptor(FEA_FIELD_EM_RESIDUAL_REAL).kind,
        AnalysisFieldKind::Scalar
    );
    let em_diag = envelope
        .data
        .run
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_EM_STATIC")
        .expect("EM static diagnostic must be present");
    assert!(em_diag.message.contains("relative_permittivity_mean="));
    assert!(em_diag.message.contains("relative_permeability_mean="));
    assert!(em_diag
        .message
        .contains("conductivity_frequency_scale_mean="));
    assert!(em_diag
        .message
        .contains("conductivity_frequency_response_coverage_ratio="));
    assert!(em_diag
        .message
        .contains("relative_permittivity_frequency_scale_mean="));
    assert!(em_diag
        .message
        .contains("relative_permittivity_frequency_response_coverage_ratio="));
    assert!(em_diag
        .message
        .contains("relative_permeability_frequency_scale_mean="));
    assert!(em_diag
        .message
        .contains("relative_permeability_frequency_response_coverage_ratio="));
    assert!(em_diag.message.contains("dispersive_loss_scale_mean="));
    assert!(em_diag
        .message
        .contains("dispersive_phase_attenuation_mean="));
    assert!(em_diag
        .message
        .contains("dispersive_phase_conductivity_attenuation_ratio="));
}

#[test]
fn analysis_run_electromagnetic_sweep_emits_resonance_metrics() {
    let mut model = sample_model();
    model.steps[0].kind = AnalysisStepKind::Electromagnetic;
    model.materials[0].electrical = Some(MaterialElectricalModel {
        reference_temperature_k: 293.15,
        conductivity_s_per_m: 5.8e7,
        resistive_heating_coefficient: 0.0039,
        relative_permittivity: 3.2,
        relative_permeability: 1.8,
        conductivity_frequency_response: vec![
            ConductivityFrequencyPoint {
                frequency_hz: 40.0,
                conductivity_scale: 1.06,
                dispersive_loss_scale: Some(0.02),
                relative_permittivity_scale: Some(1.04),
                relative_permeability_scale: Some(1.02),
            },
            ConductivityFrequencyPoint {
                frequency_hz: 60.0,
                conductivity_scale: 1.0,
                dispersive_loss_scale: Some(0.03),
                relative_permittivity_scale: Some(1.0),
                relative_permeability_scale: Some(1.0),
            },
            ConductivityFrequencyPoint {
                frequency_hz: 240.0,
                conductivity_scale: 0.91,
                dispersive_loss_scale: Some(0.04),
                relative_permittivity_scale: Some(0.96),
                relative_permeability_scale: Some(0.98),
            },
        ],
    });
    model.electromagnetic = Some(ElectromagneticDomain {
        enabled: true,
        reference_frequency_hz: 60.0,
        applied_current_a: 120.0,
    });
    configure_valid_em_authoring(&mut model);
    let envelope = analysis_run_electromagnetic_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisElectromagneticRunOptions {
            sweep_enabled: true,
            sweep_frequency_hz: vec![20.0, 40.0, 60.0, 120.0, 240.0],
            residual_target: 5.0e-7,
            harmonic_tolerance: 1.2345e-4,
            harmonic_max_iterations: 64,
            ..AnalysisElectromagneticRunOptions::default()
        },
        OperationContext::new(Some("trace-em-run-sweep".to_string()), None),
    )
    .expect("electromagnetic sweep run should succeed");
    let payload = envelope
        .data
        .electromagnetic_results
        .as_ref()
        .expect("electromagnetic payload expected");
    assert_eq!(payload.sweep_frequency_hz.len(), 5);
    assert_eq!(payload.sweep_peak_flux_density.len(), 5);
    assert_eq!(payload.sweep_solve_quality.len(), 5);
    assert!(payload.resonance_peak_frequency_hz.is_some());
    assert!(payload.resonance_peak_flux_density.is_some());
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .filter(|diag| diag.code == "FEA_EM_SWEEP")
        .any(|diag| diag.message.contains("sweep_count=5")));
    let harmonic_diag = envelope
        .data
        .run
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_EM_HARMONIC_COUPLING")
        .expect("harmonic coupling diagnostic should be present");
    assert!(harmonic_diag.message.contains("tolerance=0.00012345"));
    assert!(harmonic_diag.message.contains("iterations="));
}

#[test]
fn analysis_run_electromagnetic_rejects_invalid_harmonic_controls() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps[0].kind = AnalysisStepKind::Electromagnetic;
    model.electromagnetic = Some(ElectromagneticDomain {
        enabled: true,
        reference_frequency_hz: 60.0,
        applied_current_a: 120.0,
    });
    let err = analysis_run_electromagnetic_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisElectromagneticRunOptions {
            harmonic_max_iterations: 0,
            ..AnalysisElectromagneticRunOptions::default()
        },
        OperationContext::new(
            Some("trace-em-run-invalid-harmonic-controls".to_string()),
            None,
        ),
    )
    .expect_err("electromagnetic run should reject zero harmonic_max_iterations");
    assert_eq!(err.operation, "fea.run_electromagnetic");
    assert_eq!(err.op_version, "fea.run_electromagnetic/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_ELECTROMAGNETIC.INVALID_OPTIONS");
}

#[test]
fn analysis_run_thermal_returns_temperature_payload() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "thermal_1".to_string(),
        kind: AnalysisStepKind::Thermal,
    }];
    set_model_thermo_coupling(
        &mut model,
        ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 60.0,
            thermal_expansion_coefficient: 1.2e-5,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: vec![ThermoRegionTemperatureDelta {
                region_id: "tip".to_string(),
                temperature_delta_k: 70.0,
            }],
            time_profile: vec![
                ThermoTimeProfilePoint {
                    normalized_time: 0.0,
                    scale: 0.5,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 1.0,
                    scale: 1.0,
                },
            ],
        },
    );
    let run = analysis_run_thermal_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisThermalRunOptions {
            step_count: 6,
            ..AnalysisThermalRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect("thermal run should succeed");

    assert_eq!(run.operation, "fea.run_thermal");
    assert_eq!(run.op_version, "fea.run_thermal/v1");
    assert!(run.data.thermal_results.is_some());
    assert!(run.data.transient_results.is_none());
    assert!(run.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_THERMAL_HEAT_BALANCE"
            && diag.message.contains("input_heat=")
            && diag.message.contains("boundary_heat=")
            && diag.message.contains("stored_energy=")
            && diag.message.contains("numerical_loss=")
            && diag.message.contains("heat_balance_residual_ratio=")
    }));
    assert!(run.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_THERMAL_FIELD_RECOVERY"
            && diag.message.contains("recovery_dimensions=")
            && diag.message.contains("boundary_face_count=6")
    }));
    assert!(run.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_THERMAL_KNOWN_ANSWER"
            && diag.message.contains("slab_linear_profile_rms_ratio=")
            && diag.message.contains("slab_monotonic_edge_fraction=")
            && diag.message.contains("lumped_response_error_ratio=")
            && diag.message.contains("source_response_sign_alignment=")
    }));
    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("fea.results should return thermal payload");
    let thermal = results
        .data
        .thermal_results
        .as_ref()
        .expect("thermal results must be present");
    assert_eq!(thermal.time_points_s.len(), 6);
    assert_eq!(thermal.temperature_snapshots.len(), 6);
    assert_eq!(thermal.temperature_gradient_snapshots.len(), 6);
    assert_eq!(thermal.heat_flux_snapshots.len(), 6);
    assert_eq!(thermal.heat_source_snapshots.len(), 6);
    assert_eq!(thermal.boundary_heat_flux_snapshots.len(), 6);
    assert_eq!(thermal.temperature_gradient_snapshots[0].shape.len(), 2);
    assert_eq!(thermal.temperature_gradient_snapshots[0].shape[1], 3);
    assert_eq!(
        thermal.heat_flux_snapshots[0].shape,
        thermal.temperature_gradient_snapshots[0].shape
    );
    assert_eq!(
        thermal.heat_source_snapshots[0].shape,
        vec![thermal.temperature_gradient_snapshots[0].shape[0]]
    );
    assert_eq!(thermal.boundary_heat_flux_snapshots[0].shape, vec![6]);
    let field_ids = results
        .data
        .field_descriptors
        .iter()
        .map(|descriptor| descriptor.field_id.as_str())
        .collect::<Vec<_>>();
    assert!(field_ids.contains(&fea_thermal_temperature_gradient_field_id(0).as_str()));
    assert!(field_ids.contains(&fea_thermal_heat_flux_field_id(0).as_str()));
    assert!(field_ids.contains(&fea_thermal_heat_source_field_id(0).as_str()));
    assert!(field_ids.contains(&fea_thermal_boundary_heat_flux_field_id(0).as_str()));
    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("thermal field descriptor should be present")
    };
    assert_eq!(
        descriptor(&fea_thermal_heat_flux_field_id(0)).kind,
        AnalysisFieldKind::Vector
    );
    assert_eq!(
        descriptor(&fea_thermal_heat_flux_field_id(0)).component_count,
        Some(3)
    );
    assert_eq!(
        descriptor(&fea_thermal_boundary_heat_flux_field_id(0)).kind,
        AnalysisFieldKind::Scalar
    );
    assert_eq!(
        descriptor(&fea_thermal_boundary_heat_flux_field_id(0)).component_count,
        None
    );
    assert_eq!(results.data.summary.snapshot_count, 6);
}

#[test]
fn analysis_run_thermal_balanced_degrades_on_high_constitutive_spread() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "thermal_1".to_string(),
        kind: AnalysisStepKind::Thermal,
    }];
    model.materials.push(MaterialModel {
        material_id: "mat_poly_high_k".to_string(),
        name: "High K Composite".to_string(),
        mechanical: MaterialMechanicalModel {
            youngs_modulus_pa: 5.0e9,
            poisson_ratio: 0.33,
        },
        thermal: MaterialThermalModel {
            reference_temperature_k: 293.15,
            conductivity_w_per_mk: 1200.0,
            specific_heat_j_per_kgk: 160.0,
            ..MaterialThermalModel::default()
        },
        acoustic: None,
        electrical: None,
        plastic: None,
    });
    set_model_thermo_coupling(
        &mut model,
        ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 80.0,
            thermal_expansion_coefficient: 1.2e-5,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: vec![
                ThermoRegionTemperatureDelta {
                    region_id: "tip".to_string(),
                    temperature_delta_k: 95.0,
                },
                ThermoRegionTemperatureDelta {
                    region_id: "root".to_string(),
                    temperature_delta_k: 60.0,
                },
            ],
            time_profile: vec![
                ThermoTimeProfilePoint {
                    normalized_time: 0.0,
                    scale: 0.3,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 1.0,
                    scale: 1.0,
                },
            ],
        },
    );

    let run = analysis_run_thermal_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisThermalRunOptions {
            quality_policy: QualityPolicy::Balanced,
            step_count: 8,
            ..AnalysisThermalRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect("thermal run should execute");

    assert!(!run.data.publishable);
    assert_eq!(run.data.run_status, RunStatus::Degraded);
    assert!(run
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::ThermalConstitutiveSpreadHigh));
}

#[test]
fn analysis_run_nonlinear_rejects_models_without_nonlinear_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("nonlinear run should reject models without nonlinear step");
    assert_eq!(err.operation, "fea.run_nonlinear");
    assert_eq!(err.op_version, "fea.run_nonlinear/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_NONLINEAR.INVALID_MODEL");
}

#[test]
fn analysis_run_nonlinear_returns_native_nonlinear_result() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let envelope = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            increment_count: 16,
            ..AnalysisNonlinearRunOptions::balanced()
        },
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should succeed");

    assert_eq!(envelope.operation, "fea.run_nonlinear");
    assert_eq!(envelope.op_version, "fea.run_nonlinear/v1");
    let nonlinear = envelope
        .data
        .nonlinear_results
        .as_ref()
        .expect("nonlinear payload should exist");
    assert_eq!(nonlinear.method, NonlinearMethod::IncrementalNewtonRaphson);
    assert_eq!(nonlinear.load_factors.len(), 16);
    assert_eq!(nonlinear.load_factors.len(), nonlinear.residual_norms.len());
    assert_eq!(
        nonlinear.residual_norms.len(),
        nonlinear.increment_norms.len()
    );
    assert_eq!(
        nonlinear.residual_norms.len(),
        nonlinear.iteration_counts.len()
    );
    assert_eq!(
        nonlinear.load_factors.len(),
        nonlinear.von_mises_snapshots.len()
    );
    assert_eq!(
        nonlinear.load_factors.len(),
        nonlinear.plastic_strain_snapshots.len()
    );
    assert_eq!(
        nonlinear.load_factors.len(),
        nonlinear.equivalent_plastic_strain_snapshots.len()
    );
    assert_eq!(
        nonlinear.load_factors.len(),
        nonlinear.contact_pressure_snapshots.len()
    );
    assert_eq!(
        nonlinear.load_factors.len(),
        nonlinear.contact_gap_snapshots.len()
    );
    assert_eq!(
        nonlinear.load_factors.len(),
        nonlinear.load_factor_snapshots.len()
    );
    assert_eq!(
        nonlinear.load_factors.len(),
        nonlinear.residual_norm_snapshots.len()
    );
    assert_eq!(
        nonlinear.von_mises_snapshots[0].field_id,
        fea_nonlinear_von_mises_field_id(0)
    );
    assert_eq!(
        nonlinear.plastic_strain_snapshots[0].field_id,
        fea_nonlinear_plastic_strain_field_id(0)
    );
    assert_eq!(
        nonlinear.equivalent_plastic_strain_snapshots[0].field_id,
        fea_nonlinear_equivalent_plastic_strain_field_id(0)
    );
    assert_eq!(
        nonlinear.contact_pressure_snapshots[0].field_id,
        fea_nonlinear_contact_pressure_field_id(0)
    );
    assert_eq!(
        nonlinear.contact_gap_snapshots[0].field_id,
        fea_nonlinear_contact_gap_field_id(0)
    );
    assert_eq!(
        nonlinear.load_factor_snapshots[0].field_id,
        fea_nonlinear_load_factor_field_id(0)
    );
    assert_eq!(
        nonlinear.residual_norm_snapshots[0].field_id,
        fea_nonlinear_residual_norm_field_id(0)
    );
    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("nonlinear results should be queryable");
    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("nonlinear descriptor should be present")
    };
    for field_id in [
        fea_nonlinear_von_mises_field_id(0),
        fea_nonlinear_equivalent_plastic_strain_field_id(0),
        fea_nonlinear_contact_pressure_field_id(0),
        fea_nonlinear_contact_gap_field_id(0),
        fea_nonlinear_load_factor_field_id(0),
        fea_nonlinear_residual_norm_field_id(0),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Scalar);
        assert_eq!(descriptor.component_count, None);
    }
    let plastic_strain_descriptor = descriptor(&fea_nonlinear_plastic_strain_field_id(0));
    assert_eq!(plastic_strain_descriptor.kind, AnalysisFieldKind::Tensor);
    assert_eq!(plastic_strain_descriptor.component_count, Some(6));
    assert!(nonlinear.tangent_rebuild_count > 0);
    assert!(nonlinear.iteration_spike_count <= nonlinear.load_factors.len());
    assert!(nonlinear.max_line_search_backtracks_per_increment > 0);
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_NONLINEAR_CONVERGENCE"));
}

#[test]
fn analysis_run_nonlinear_strict_rejects_iteration_cap_exhaustion() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];

    let envelope = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            quality_policy: QualityPolicy::Strict,
            max_newton_iters: 1,
            line_search: false,
            ..AnalysisNonlinearRunOptions::balanced()
        },
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should produce envelope");

    assert_eq!(envelope.data.run_status, RunStatus::Degraded);
    assert!(!envelope.data.publishable);
    assert!(envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::NonlinearIncrementFailure));
}

#[test]
fn analysis_run_nonlinear_rejects_missing_prep_artifact_reference() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];

    let error = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            prep_artifact_id: Some("prep:missing".to_string()),
            ..AnalysisNonlinearRunOptions::production_recommended()
        },
        OperationContext::new(None, None),
    )
    .expect_err("missing prep artifact reference should fail");
    assert_eq!(error.error_code, "RM.FEA.RUN_PREP.NOT_FOUND");
}

#[test]
fn analysis_run_nonlinear_rejects_mismatched_prep_artifact_reference() {
    let _guard = analysis_test_guard();
    let _prep_guard = crate::geometry::prep_artifact_test_guard();
    let geometry = sample_step_like_geometry_asset();
    let prep = crate::geometry::geometry_prep_for_analysis_op(
        &geometry,
        crate::geometry::GeometryPrepForAnalysisSpec::default(),
        OperationContext::new(None, None),
    )
    .expect("prep should succeed");

    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];

    let error = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            prep_artifact_id: Some(prep.data.prep_artifact_id.clone()),
            ..AnalysisNonlinearRunOptions::production_recommended()
        },
        OperationContext::new(None, None),
    )
    .expect_err("mismatched prep artifact reference should fail");
    assert_eq!(error.error_code, "RM.FEA.RUN_PREP.MISMATCH");
}

#[test]
fn analysis_run_nonlinear_rejects_stale_prep_artifact_when_newer_revision_exists() {
    let _guard = analysis_test_guard();
    let _prep_guard = crate::geometry::prep_artifact_test_guard();
    crate::geometry::reset_prep_artifact_store_for_tests();
    crate::geometry::configure_prep_artifacts(crate::geometry::GeometryPrepArtifactConfig {
        require_latest_revision: Some(true),
        ..crate::geometry::GeometryPrepArtifactConfig::default()
    })
    .expect("prep artifact config should be configurable");

    let mut geometry_v1 = sample_step_like_geometry_asset();
    geometry_v1.revision = 1;
    let mut geometry_v2 = geometry_v1.clone();
    geometry_v2.revision = 2;

    let prep_v1 = crate::geometry::geometry_prep_for_analysis_op(
        &geometry_v1,
        crate::geometry::GeometryPrepForAnalysisSpec::default(),
        OperationContext::new(None, None),
    )
    .expect("prep v1 should succeed");
    let _prep_v2 = crate::geometry::geometry_prep_for_analysis_op(
        &geometry_v2,
        crate::geometry::GeometryPrepForAnalysisSpec::default(),
        OperationContext::new(None, None),
    )
    .expect("prep v2 should succeed");

    let created = analysis_create_model_op(
        &geometry_v1,
        AnalysisCreateModelIntentSpec {
            model_id: "stale_prep_model".to_string(),
            profile: AnalysisCreateModelProfile::NonlinearStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("create model should succeed");

    let error = analysis_run_nonlinear_with_options_op(
        &created.data,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            prep_artifact_id: Some(prep_v1.data.prep_artifact_id),
            ..AnalysisNonlinearRunOptions::production_recommended()
        },
        OperationContext::new(None, None),
    )
    .expect_err("stale prep artifact should fail");
    assert_eq!(error.error_code, "RM.FEA.RUN_PREP.STALE");

    let health = crate::geometry::geometry_prep_artifact_health_op(
        crate::geometry::GeometryPrepArtifactHealthQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("prep health should be queryable");
    assert!(health.data.metrics.stale_reject_count >= 1);

    crate::geometry::reset_prep_artifact_store_for_tests();
}

#[test]
fn nonlinear_quality_policy_diverges_for_increment_failures() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];

    let run_with_policy = |quality_policy| {
        analysis_run_nonlinear_with_options_op(
            &model,
            ComputeBackend::Cpu,
            AnalysisNonlinearRunOptions {
                quality_policy,
                max_newton_iters: 1,
                line_search: false,
                max_line_search_backtracks: 0,
                ..AnalysisNonlinearRunOptions::balanced()
            },
            OperationContext::new(None, None),
        )
        .expect("nonlinear run should return envelope")
    };

    let exploratory = run_with_policy(QualityPolicy::Exploratory);
    let balanced = run_with_policy(QualityPolicy::Balanced);
    let strict = run_with_policy(QualityPolicy::Strict);

    assert!(exploratory.data.publishable);
    assert_eq!(exploratory.data.run_status, RunStatus::Publishable);
    assert!(balanced
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::NonlinearIncrementFailure));
    assert!(!balanced.data.publishable);
    assert_eq!(balanced.data.run_status, RunStatus::Degraded);

    assert!(strict
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::NonlinearIncrementFailure));
    assert!(!strict.data.publishable);
    assert_eq!(strict.data.run_status, RunStatus::Degraded);
}

#[test]
fn nonlinear_balanced_degrades_when_thermo_mechanical_severity_is_high() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    for material in &mut model.materials {
        material.thermal.expansion_coefficient_per_k = 1.0e-3;
    }
    set_model_thermo_coupling(
        &mut model,
        ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 90.0,
            thermal_expansion_coefficient: 1.0e-3,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: Vec::new(),
            time_profile: Vec::new(),
        },
    );

    let run = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            quality_policy: QualityPolicy::Balanced,
            ..AnalysisNonlinearRunOptions::production_recommended()
        },
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should return envelope");

    assert!(!run.data.publishable);
    assert_eq!(run.data.run_status, RunStatus::Degraded);
    assert!(run
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::ThermoMechanicalNonlinearStress));

    let nonlinear = run
        .data
        .nonlinear_results
        .as_ref()
        .expect("nonlinear results should be present");
    assert_eq!(
        nonlinear.thermo_mechanical_temperature_snapshots.len(),
        nonlinear.load_factors.len()
    );
    assert_eq!(
        nonlinear.thermo_mechanical_thermal_strain_snapshots.len(),
        nonlinear.load_factors.len()
    );
    assert_eq!(
        nonlinear.thermo_mechanical_thermal_stress_snapshots.len(),
        nonlinear.load_factors.len()
    );
    assert_eq!(
        nonlinear.thermo_mechanical_displacement_snapshots.len(),
        nonlinear.load_factors.len()
    );
    assert_eq!(
        nonlinear.thermo_mechanical_von_mises_snapshots.len(),
        nonlinear.load_factors.len()
    );
    assert_eq!(
        nonlinear
            .thermo_mechanical_coupling_residual_snapshots
            .len(),
        nonlinear.load_factors.len()
    );
    assert_eq!(
        nonlinear.thermo_mechanical_temperature_snapshots[0].field_id,
        fea_thermo_mechanical_temperature_field_id(0)
    );
    assert_eq!(
        nonlinear.thermo_mechanical_thermal_strain_snapshots[0].field_id,
        fea_thermo_mechanical_thermal_strain_field_id(0)
    );
    assert_eq!(
        nonlinear.thermo_mechanical_thermal_stress_snapshots[0].field_id,
        fea_thermo_mechanical_thermal_stress_field_id(0)
    );
    assert_eq!(
        nonlinear.thermo_mechanical_displacement_snapshots[0].field_id,
        fea_thermo_mechanical_displacement_field_id(0)
    );
    assert_eq!(
        nonlinear.thermo_mechanical_von_mises_snapshots[0].field_id,
        fea_thermo_mechanical_von_mises_field_id(0)
    );
    assert_eq!(
        nonlinear.thermo_mechanical_coupling_residual_snapshots[0].field_id,
        fea_thermo_mechanical_coupling_residual_field_id(0)
    );

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("thermo-mechanical nonlinear results should be queryable");
    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("nonlinear thermo-mechanical descriptor should be present")
    };
    for field_id in [
        fea_thermo_mechanical_temperature_field_id(0),
        fea_thermo_mechanical_von_mises_field_id(0),
        fea_thermo_mechanical_coupling_residual_field_id(0),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Scalar);
        assert_eq!(descriptor.component_count, None);
    }
    let displacement_descriptor = descriptor(&fea_thermo_mechanical_displacement_field_id(0));
    assert_eq!(displacement_descriptor.kind, AnalysisFieldKind::Vector);
    assert_eq!(displacement_descriptor.component_count, Some(3));
    for field_id in [
        fea_thermo_mechanical_thermal_strain_field_id(0),
        fea_thermo_mechanical_thermal_stress_field_id(0),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Tensor);
        assert_eq!(descriptor.component_count, Some(6));
    }
}

#[test]
fn nonlinear_balanced_degrades_when_thermo_heterogeneity_is_high() {
    let _guard = analysis_test_guard();
    let mut model = sample_model_with_material_assignment_mismatch();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    set_model_thermo_coupling(
        &mut model,
        ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 90.0,
            thermal_expansion_coefficient: 1.2e-5,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: Vec::new(),
            time_profile: Vec::new(),
        },
    );

    let run = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            quality_policy: QualityPolicy::Balanced,
            ..AnalysisNonlinearRunOptions::production_recommended()
        },
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should return envelope");

    assert!(!run.data.publishable);
    assert_eq!(run.data.run_status, RunStatus::Degraded);
    assert!(run.data.quality_reasons.iter().any(|reason| {
        reason.code == QualityReasonCode::ThermoMechanicalConstitutiveSpreadHigh
            || reason.code == QualityReasonCode::ThermoMechanicalAssignmentHeterogeneityHigh
    }));
}

#[test]
fn analysis_results_query_can_exclude_nonlinear_payload() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let run = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_field_values: true,

            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: false,
            include_electromagnetic_results: false,
        },
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    assert!(results.data.nonlinear_results.is_none());
    assert!(results.data.summary.increment_count > 0);
    assert!(results.data.summary.failed_increment_count.is_some());
    assert!(results.data.summary.max_nonlinear_residual_norm.is_some());
    assert!(results.data.summary.max_nonlinear_increment_norm.is_some());
    assert!(results.data.summary.max_nonlinear_iteration_count.is_some());
    assert!(results.data.summary.final_increment_converged.is_some());
    assert!(results
        .data
        .summary
        .nonlinear_line_search_backtracks
        .is_some());
    assert!(results
        .data
        .summary
        .nonlinear_max_backtracks_per_increment
        .is_some());
    assert!(results
        .data
        .summary
        .nonlinear_tangent_rebuild_count
        .is_some());
    assert!(results
        .data
        .summary
        .nonlinear_iteration_spike_count
        .is_some());
    assert!(results
        .data
        .summary
        .nonlinear_convergence_stall_count
        .is_some());
    assert!(results
        .data
        .summary
        .nonlinear_backtrack_burst_count
        .is_some());
    let field_ids = results
        .data
        .field_descriptors
        .iter()
        .map(|descriptor| descriptor.field_id.as_str())
        .collect::<Vec<_>>();
    assert!(field_ids.contains(&fea_nonlinear_von_mises_field_id(0).as_str()));
    assert!(field_ids.contains(&fea_nonlinear_plastic_strain_field_id(0).as_str()));
    assert!(field_ids.contains(&fea_nonlinear_equivalent_plastic_strain_field_id(0).as_str()));
    assert!(field_ids.contains(&fea_nonlinear_contact_pressure_field_id(0).as_str()));
    assert!(field_ids.contains(&fea_nonlinear_contact_gap_field_id(0).as_str()));
}

#[test]
fn nonlinear_results_deserialize_with_missing_new_fields() {
    let payload = serde_json::json!({
        "nonlinear_payload_version": "nonlinear_results/v1",
        "load_factors": [0.5, 1.0],
        "displacement_snapshots": [],
        "residual_norms": [1.0e-6, 5.0e-7],
        "method": "incremental_newton_raphson"
    });
    let parsed: NonlinearResultsData =
        serde_json::from_value(payload).expect("legacy nonlinear payload should deserialize");

    assert_eq!(parsed.increment_norms.len(), 0);
    assert_eq!(parsed.iteration_counts.len(), 0);
    assert_eq!(parsed.failed_increments, 0);
    assert_eq!(parsed.line_search_backtracks, 0);
    assert_eq!(parsed.max_line_search_backtracks_per_increment, 0);
    assert_eq!(parsed.tangent_rebuild_count, 0);
    assert_eq!(parsed.iteration_spike_count, 0);
    assert_eq!(parsed.convergence_stall_count, 0);
    assert_eq!(parsed.backtrack_burst_count, 0);
    assert!(parsed.von_mises_snapshots.is_empty());
    assert!(parsed.plastic_strain_snapshots.is_empty());
    assert!(parsed.equivalent_plastic_strain_snapshots.is_empty());
    assert!(parsed.contact_pressure_snapshots.is_empty());
    assert!(parsed.contact_gap_snapshots.is_empty());
}

#[test]
fn analysis_run_transient_returns_native_transient_result() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];

    let envelope = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("transient run should return envelope");

    assert_eq!(envelope.operation, "fea.run_transient");
    assert_eq!(envelope.op_version, "fea.run_transient/v1");
    assert_eq!(envelope.data.run.solver_method, "implicit_euler_pcg");
    assert_eq!(envelope.data.provenance.solver_method, "implicit_euler_pcg");
    assert_eq!(envelope.data.run_status, RunStatus::Publishable);
    assert!(envelope.data.publishable);
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_TRANSIENT_CONVERGENCE"));
    let transient = envelope
        .data
        .transient_results
        .as_ref()
        .expect("transient payload should exist");
    assert_eq!(
        transient.integration_method,
        TransientIntegrationMethod::ImplicitEuler
    );
    assert!(!transient.time_points_s.is_empty());
    assert_eq!(
        transient.time_points_s.len(),
        transient.displacement_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.velocity_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.acceleration_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.von_mises_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.kinetic_energy_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.strain_energy_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.residual_norm_snapshots.len()
    );
    assert_eq!(
        transient.velocity_snapshots[1].field_id,
        fea_transient_velocity_field_id(1)
    );
    assert_eq!(
        transient.acceleration_snapshots[1].field_id,
        fea_transient_acceleration_field_id(1)
    );
    assert_eq!(
        transient.von_mises_snapshots[1].field_id,
        fea_transient_von_mises_field_id(1)
    );
    assert_eq!(
        transient.kinetic_energy_snapshots[1].field_id,
        fea_transient_kinetic_energy_field_id(1)
    );
    assert_eq!(
        transient.strain_energy_snapshots[1].field_id,
        fea_transient_strain_energy_field_id(1)
    );
    assert_eq!(
        transient.residual_norm_snapshots[1].field_id,
        fea_transient_residual_norm_field_id(1)
    );
    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("transient results should be queryable");
    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("transient descriptor should be present")
    };
    for field_id in [
        fea_transient_velocity_field_id(1),
        fea_transient_acceleration_field_id(1),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Vector);
        assert_eq!(descriptor.component_count, Some(3));
    }
    for field_id in [
        fea_transient_von_mises_field_id(1),
        fea_transient_kinetic_energy_field_id(1),
        fea_transient_strain_energy_field_id(1),
        fea_transient_residual_norm_field_id(1),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Scalar);
        assert_eq!(descriptor.component_count, None);
    }
}

#[test]
fn analysis_run_cfd_returns_typed_payload_and_flow_diagnostics() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps[0].kind = AnalysisStepKind::Cfd;
    model.boundary_conditions = sample_cfd_boundary_conditions(4.25);
    model
        .boundary_conditions
        .retain(|boundary| boundary.bc_id != "bc_cfd_wall_lower");
    if let Some(wall) = model
        .boundary_conditions
        .iter_mut()
        .find(|boundary| boundary.bc_id == "bc_cfd_wall_upper")
    {
        wall.kind = BoundaryConditionKind::CfdSlipWall;
    }
    model.cfd = Some(sample_cfd_domain(CfdSolveFamily::SteadyState, true));

    let envelope = analysis_run_cfd_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisCfdRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            time_step_s: 1.0e-3,
            step_count: 4,
            max_linear_iters: 64,
            tolerance: 1.0e-8,
            residual_warn_threshold: 1.0e-4,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(None, None),
    )
    .expect("cfd run should return envelope");

    assert_eq!(envelope.operation, "fea.run_cfd");
    assert_eq!(envelope.op_version, "fea.run_cfd/v1");
    assert_eq!(
        envelope.data.run.solver_method,
        "cfd_velocity_pressure_finite_volume"
    );
    assert_eq!(
        envelope.data.provenance.solver_method,
        "cfd_velocity_pressure_finite_volume"
    );
    assert!(envelope.data.transient_results.is_none());
    let velocity = envelope
        .data
        .run
        .field(FEA_FIELD_CFD_VELOCITY)
        .expect("cfd velocity field should be present");
    let pressure = envelope
        .data
        .run
        .field(FEA_FIELD_CFD_PRESSURE)
        .expect("cfd pressure field should be present");
    let vorticity = envelope
        .data
        .run
        .field(FEA_FIELD_CFD_VORTICITY)
        .expect("cfd vorticity field should be present");
    let wall_shear = envelope
        .data
        .run
        .field(FEA_FIELD_CFD_WALL_SHEAR_STRESS)
        .expect("cfd wall-shear field should be present");
    assert_eq!(velocity.shape.len(), 2);
    assert_eq!(velocity.shape[1], 3);
    assert_eq!(pressure.shape, vec![velocity.shape[0]]);
    assert_eq!(vorticity.shape, velocity.shape);
    assert_eq!(wall_shear.shape, vec![1, 3]);
    assert_ne!(wall_shear.shape[0], velocity.shape[0]);
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_CFD_FLOW"
            && diag.message.contains("inlet_velocity=4.25")
            && diag.message.contains("reynolds_number=")
            && diag.message.contains("solve_family=steady_state")
            && diag.message.contains("topology_basis=implicit_channel")));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_CFD_RESIDUAL"
            && diag.message.contains("max_momentum_residual=")
            && diag.message.contains("max_continuity_residual=")
    }));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_CFD_ASSEMBLY"
            && diag
                .message
                .contains("basis=finite_volume_velocity_pressure")
            && diag.message.contains("topology_basis=implicit_channel")
            && diag
                .message
                .contains("topology_geometry_source=implicit_channel")
            && diag.message.contains("domain_length_m=")
            && diag.message.contains("face_area_m2=")
            && diag.message.contains("control_volume_volume_m3=")
            && diag.message.contains("nominal_mass_flow_rate_kg_per_s=")
            && diag.message.contains("courant_number=")
            && diag.message.contains("mass_balance_residual=")
            && diag.message.contains("pressure_drop_pa=")
    }));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_CFD_BOUNDARY_CONDITIONS"
            && diag.message.contains("boundary_source=authored")
            && diag.message.contains("authored_boundary_count=3")
            && diag.message.contains("inlet_boundary_count=1")
            && diag.message.contains("outlet_boundary_count=1")
            && diag.message.contains("wall_boundary_count=1")
            && diag.message.contains("no_slip_wall_boundary_count=0")
            && diag.message.contains("slip_wall_boundary_count=1")
            && diag.message.contains("boundary_coverage_ratio=1")
            && diag.message.contains("nominal_inlet_velocity_m_per_s=4.25")
    }));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_CFD_PRESSURE_CORRECTION"
            && diag.message.contains("iteration_count=")
            && diag.message.contains("max_linear_iters=64")
            && diag.message.contains("tolerance=0.00000001")
            && diag.message.contains("pressure_correction_residual_ratio=")
            && diag.message.contains("velocity_correction_residual_ratio=")
    }));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_CFD_TRANSIENT_EVOLUTION"
            && diag.message.contains("solve_family=steady_state")
            && diag.message.contains("transient_scale_variation=")
    }));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_CFD_KNOWN_ANSWER"
            && diag.message.contains("pressure_drop_balance_ratio=")
            && diag.message.contains("mass_flux_uniformity_ratio=")
            && diag.message.contains("pressure_monotonic_cell_fraction=")
            && diag.message.contains("known_answer_coverage_ratio=")
    }));
    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("cfd results should be queryable");
    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("CFD descriptor should be present")
    };
    for field_id in [
        FEA_FIELD_CFD_VELOCITY,
        FEA_FIELD_CFD_VORTICITY,
        FEA_FIELD_CFD_WALL_SHEAR_STRESS,
    ] {
        let descriptor = descriptor(field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Vector);
        assert_eq!(descriptor.component_count, Some(3));
    }
    for field_id in [
        FEA_FIELD_CFD_PRESSURE,
        FEA_FIELD_CFD_RESIDUAL_MOMENTUM,
        FEA_FIELD_CFD_RESIDUAL_CONTINUITY,
        FEA_FIELD_CFD_REYNOLDS_NUMBER,
    ] {
        let descriptor = descriptor(field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Scalar);
        assert_eq!(descriptor.component_count, None);
    }
}

#[test]
fn analysis_run_cfd_uses_prep_control_volume_topology() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps[0].kind = AnalysisStepKind::Cfd;
    model.boundary_conditions = sample_cfd_boundary_conditions(3.0);
    model.cfd = Some(sample_cfd_domain(CfdSolveFamily::SteadyState, true));

    let run = solve_cfd_finite_volume_run(
        &model,
        model.cfd.as_ref().expect("cfd domain should exist"),
        ComputeBackend::Cpu,
        &AnalysisCfdRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            time_step_s: 1.0e-3,
            step_count: 2,
            max_linear_iters: 32,
            tolerance: 1.0e-8,
            residual_warn_threshold: 1.0e-4,
            prep_context: Some(sample_analysis_run_prep_context()),
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        Some(sample_analysis_run_prep_context()),
    );

    let velocity = run
        .field(FEA_FIELD_CFD_VELOCITY)
        .expect("cfd velocity field should be present");
    assert_eq!(velocity.shape, vec![15, 3]);
    assert!(run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_CFD_ASSEMBLY"
            && diag
                .message
                .contains("topology_basis=prep_control_volume_connectivity")
            && diag
                .message
                .contains("topology_geometry_source=prep_element_geometry")
            && diag.message.contains("control_volume_count=15")
            && diag.message.contains("control_volume_face_count=19")
            && diag
                .message
                .contains("control_volume_internal_face_count=11")
            && diag
                .message
                .contains("control_volume_boundary_face_count=8")
            && diag
                .message
                .contains("control_volume_connectivity_coverage_ratio=1")
            && diag.message.contains("domain_length_m=2.4")
            && diag.message.contains("face_area_m2=0.04")
            && diag.message.contains("face_area_m2=")
            && diag.message.contains("control_volume_volume_m3=")
            && diag.message.contains("courant_number=")
            && diag.message.contains("active_dimension_count=3")
            && diag.message.contains("element_geometry_node_count=4")
            && diag.message.contains("element_geometry_edge_count=5")
            && diag.message.contains("element_geometry_coverage_ratio=1")
    }));
}

#[test]
fn analysis_run_cfd_rejects_partial_authored_boundary_conditions() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps[0].kind = AnalysisStepKind::Cfd;
    model.cfd = Some(sample_cfd_domain(CfdSolveFamily::SteadyState, true));
    model.boundary_conditions = vec![BoundaryCondition {
        bc_id: "bc_cfd_inlet_only".to_string(),
        region_id: "fluid_inlet".to_string(),
        kind: BoundaryConditionKind::CfdInletVelocity {
            velocity_m_per_s: 4.25,
        },
    }];

    let err = analysis_run_cfd_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("partial authored cfd boundaries should fail validation");

    assert_eq!(err.operation, "fea.run_cfd");
    assert_eq!(err.op_version, "fea.run_cfd/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_CFD.INVALID_BOUNDARY_CONDITIONS");
}

#[test]
fn analysis_run_cht_returns_coupled_payload_and_diagnostics() {
    let _guard = analysis_test_guard();
    let model = sample_cht_model();

    let envelope = analysis_run_cht_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisChtRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            time_step_s: 1.0e-3,
            step_count: 4,
            max_linear_iters: 64,
            tolerance: 1.0e-8,
            residual_warn_threshold: 1.0e-4,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(None, None),
    )
    .expect("cht run should return envelope");

    assert_eq!(envelope.operation, "fea.run_cht");
    assert_eq!(envelope.op_version, "fea.run_cht/v1");
    assert_eq!(envelope.data.run.solver_method, "cht_conjugate_projection");
    assert!(envelope.data.transient_results.is_none());
    assert!(envelope.data.thermal_results.is_some());
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_CFD_FLOW"
            && diag.message.contains("reynolds_number=")
            && diag.message.contains("topology_basis=implicit_channel")));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_CFD_ASSEMBLY"
            && diag.message.contains("topology_basis=implicit_channel")
            && diag.message.contains("control_volume_count=")
            && diag.message.contains("domain_length_m=")
    }));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_CHT_COUPLING"
            && diag.message.contains("applied_temperature_delta_k=60")));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_CHT_INTERFACE_CLOSURE"
            && diag.message.contains("interface_face_count=")
            && diag.message.contains("max_temperature_jump_k=")
            && diag.message.contains("max_energy_residual=")
            && diag.message.contains("heat_flux_balance_ratio=")
            && diag.message.contains("thermal_transport_residual_ratio=")
            && diag
                .message
                .contains("interface_temperature_continuity_ratio=")
            && diag.message.contains("max_advection_temperature_shift_k=")
            && diag.message.contains("interface_conductance_w_per_m2k=")
            && diag
                .message
                .contains("flux_temperature_law_residual_ratio=")
            && diag
                .message
                .contains("heat_flux_realization_residual_ratio=")
            && diag.message.contains("coupled_interface_iteration_count=")
            && diag.message.contains("coupled_interface_residual_ratio=")
            && diag.message.contains("thermal_network_node_count=")
            && diag.message.contains("thermal_network_edge_count=")
            && diag.message.contains("thermal_network_residual_ratio=")
    }));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_CHT_KNOWN_ANSWER"
            && diag.message.contains("basis=heated_channel_conjugate_slab")
            && diag
                .message
                .contains("heated_channel_energy_residual_ratio=")
            && diag
                .message
                .contains("conjugate_slab_flux_law_residual_ratio=")
            && diag.message.contains("advection_shift_coverage_ratio=")
            && diag.message.contains("coupled_interface_residual_ratio=")
            && diag
                .message
                .contains("heat_flux_realization_residual_ratio=")
            && diag.message.contains("thermal_network_residual_ratio=")
            && diag.message.contains("known_answer_coverage_ratio=")
    }));
    let thermal = envelope
        .data
        .thermal_results
        .as_ref()
        .expect("thermal payload should exist");
    assert_eq!(thermal.time_points_s.len(), 4);
    let fluid_temperature = envelope
        .data
        .run
        .field(&fea_cht_fluid_temperature_field_id(0))
        .expect("cht fluid temperature field should be present");
    let solid_temperature = envelope
        .data
        .run
        .field(&fea_cht_solid_temperature_field_id(0))
        .expect("cht solid temperature field should be present");
    let interface_heat_flux = envelope
        .data
        .run
        .field(&fea_cht_interface_heat_flux_field_id(0))
        .expect("cht interface heat-flux field should be present");
    let interface_temperature_jump = envelope
        .data
        .run
        .field(&fea_cht_interface_temperature_jump_field_id(0))
        .expect("cht interface temperature jump field should be present");
    let thermal_flux_face_count = thermal
        .heat_flux_snapshots
        .first()
        .and_then(|field| field.shape.first().copied())
        .expect("thermal heat-flux snapshot should carry a recovery domain");
    let expected_interface_face_count =
        fluid_interface_face_count(CfdDomainTopology::from_model(&model, None));
    assert_eq!(solid_temperature.shape, fluid_temperature.shape);
    assert!(expected_interface_face_count >= thermal_flux_face_count);
    assert_eq!(
        interface_heat_flux.shape,
        vec![expected_interface_face_count]
    );
    assert_eq!(interface_temperature_jump.shape, interface_heat_flux.shape);
    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("cht results should be queryable");
    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("CHT descriptor should be present")
    };
    let velocity_descriptor = descriptor(FEA_FIELD_CHT_FLUID_VELOCITY);
    assert_eq!(velocity_descriptor.kind, AnalysisFieldKind::Vector);
    assert_eq!(velocity_descriptor.component_count, Some(3));
    for field_id in [
        FEA_FIELD_CHT_FLUID_PRESSURE.to_string(),
        fea_cht_fluid_temperature_field_id(0),
        fea_cht_solid_temperature_field_id(0),
        fea_cht_interface_temperature_jump_field_id(0),
        fea_cht_energy_residual_field_id(0),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Scalar);
        assert_eq!(descriptor.component_count, None);
    }
    let interface_heat_flux_descriptor = descriptor(&fea_cht_interface_heat_flux_field_id(0));
    assert_eq!(
        interface_heat_flux_descriptor.kind,
        AnalysisFieldKind::Scalar
    );
    assert_eq!(interface_heat_flux_descriptor.component_count, None);
}

#[test]
fn analysis_run_fsi_returns_coupled_payload_and_diagnostics() {
    let _guard = analysis_test_guard();
    let model = sample_fsi_model();

    let envelope = analysis_run_fsi_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisFsiRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            time_step_s: 1.0e-3,
            step_count: 4,
            max_linear_iters: 64,
            tolerance: 1.0e-8,
            residual_warn_threshold: 1.0e-4,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(None, None),
    )
    .expect("fsi run should return envelope");

    assert_eq!(envelope.operation, "fea.run_fsi");
    assert_eq!(envelope.op_version, "fea.run_fsi/v1");
    assert_eq!(
        envelope.data.run.solver_method,
        "fsi_partitioned_projection"
    );
    assert!(envelope.data.transient_results.is_none());
    assert!(envelope.data.thermal_results.is_none());
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_CFD_FLOW"
            && diag.message.contains("reynolds_number=")
            && diag.message.contains("topology_basis=implicit_channel")));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_CFD_RESIDUAL"
            && diag.message.contains("max_momentum_residual=")));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_CFD_ASSEMBLY"
            && diag.message.contains("topology_basis=implicit_channel")
            && diag.message.contains("control_volume_count=")
            && diag.message.contains("domain_length_m=")
    }));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_FSI_INTERFACE_RESIDUAL"
            && diag.message.contains("max_interface_residual=")));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| diag.code
        == "FEA_FSI_INTERFACE_CLOSURE"
        && diag.message.contains("interface_node_count=")
        && diag.message.contains("interface_face_count=")
        && diag.message.contains("force_balance_ratio=")
        && diag
            .message
            .contains("max_displacement_transfer_residual_m=")
        && diag.message.contains("max_coupling_iteration_count=")
        && diag.message.contains("pressure_feedback_residual_ratio=")
        && diag.message.contains("two_way_interface_residual_ratio=")
        && diag
            .message
            .contains("structural_traction_update_residual_ratio=")
        && diag
            .message
            .contains("pressure_displacement_law_residual_ratio=")
        && diag.message.contains("structural_solve_residual_ratio=")
        && diag.message.contains("interface_work_j_per_m2=")
        && diag.message.contains("structural_strain_energy_j_per_m2=")
        && diag
            .message
            .contains("interface_work_energy_residual_ratio=")
        && diag.message.contains("structural_coupling_edge_count=")
        && diag.message.contains("interface_stiffness_pa_per_m=")));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_FSI_KNOWN_ANSWER"
            && diag
                .message
                .contains("basis=pressure_loaded_wall_partitioned")
            && diag
                .message
                .contains("pressure_loaded_wall_displacement_law_residual_ratio=")
            && diag
                .message
                .contains("interface_traction_balance_residual_ratio=")
            && diag
                .message
                .contains("partitioned_pressure_feedback_residual_ratio=")
            && diag.message.contains("two_way_interface_residual_ratio=")
            && diag
                .message
                .contains("structural_traction_update_residual_ratio=")
            && diag.message.contains("structural_solve_residual_ratio=")
            && diag
                .message
                .contains("interface_work_energy_residual_ratio=")
            && diag.message.contains("known_answer_coverage_ratio=")));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_FSI_COUPLING"
            && diag.message.contains("cfd_profile_point_count=2")
            && diag.message.contains("authored_interface_count=")
            && diag.message.contains("interface_node_count=")
            && diag.message.contains("interface_face_count=")));
    let interface_pressure = envelope
        .data
        .run
        .field(&fea_fsi_interface_pressure_field_id(0))
        .expect("fsi interface pressure field should be present");
    let interface_traction = envelope
        .data
        .run
        .field(&fea_fsi_interface_traction_field_id(0))
        .expect("fsi interface traction field should be present");
    let interface_displacement = envelope
        .data
        .run
        .field(&fea_fsi_interface_displacement_field_id(0))
        .expect("fsi interface displacement field should be present");
    assert_eq!(interface_pressure.shape.len(), 1);
    assert_eq!(
        interface_traction.shape,
        vec![interface_pressure.shape[0], 3]
    );
    assert_eq!(interface_displacement.shape.len(), 2);
    assert_eq!(interface_displacement.shape[1], 3);
    assert_eq!(
        interface_displacement.shape[0],
        interface_pressure.shape[0] + 1
    );
    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("fsi results should be queryable");
    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("FSI descriptor should be present")
    };
    for field_id in [
        fea_fsi_fluid_velocity_field_id(0),
        fea_fsi_structural_displacement_field_id(0),
        fea_fsi_interface_traction_field_id(0),
        fea_fsi_interface_displacement_field_id(0),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Vector);
        assert_eq!(descriptor.component_count, Some(3));
    }
    for field_id in [
        fea_fsi_fluid_pressure_field_id(0),
        fea_fsi_interface_pressure_field_id(0),
        fea_fsi_interface_residual_field_id(0),
        fea_fsi_coupling_iteration_count_field_id(0),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Scalar);
        assert_eq!(descriptor.component_count, None);
    }
}

#[test]
fn cht_prepared_topology_uses_boundary_faces_for_interface_fields() {
    let _guard = analysis_test_guard();
    let model = sample_cht_model();
    let cfd_domain = model.cfd.as_ref().expect("cfd domain should exist");
    let prep_context = sample_analysis_run_prep_context();
    let topology = CfdDomainTopology::from_model(&model, Some(prep_context));
    let thermo_context = to_fea_thermo_mechanical_context(model_thermo_coupling_options(&model));
    let thermal_run = run_thermal_with_options(
        &model,
        ComputeBackend::Cpu,
        ThermalSolveOptions {
            step_count: 4,
            time_step_s: 1.0e-3,
            residual_target: 1.0e-4,
            prep_context: to_fea_prep_context(Some(prep_context), None),
            thermo_mechanical_context: thermo_context,
        },
    )
    .expect("thermal run should succeed");

    let (fields, closure) = build_cht_run_fields(
        cfd_domain,
        topology,
        &thermal_run,
        cht_interface_conductance_w_per_m2k(&model),
        64,
        1.0e-8,
    );
    let heat_flux = fields
        .iter()
        .find(|field| field.field_id == fea_cht_interface_heat_flux_field_id(0))
        .expect("CHT heat flux field should be present");
    let temperature_jump = fields
        .iter()
        .find(|field| field.field_id == fea_cht_interface_temperature_jump_field_id(0))
        .expect("CHT temperature jump field should be present");

    assert_eq!(
        topology.basis,
        CfdDomainTopologyBasis::PrepControlVolumeConnectivity
    );
    assert_eq!(fluid_interface_face_count(topology), 8);
    assert_eq!(closure.interface_face_count, 8);
    assert_eq!(closure.thermal_network_node_count, 8);
    assert_eq!(closure.thermal_network_edge_count, 11);
    assert_eq!(heat_flux.shape, vec![8]);
    assert_eq!(temperature_jump.shape, vec![8]);
}

#[test]
fn fsi_prepared_topology_uses_boundary_faces_for_interface_fields() {
    let _guard = analysis_test_guard();
    let model = sample_fsi_model();
    let cfd_domain = model.cfd.as_ref().expect("cfd domain should exist");
    let prep_context = sample_analysis_run_prep_context();
    let topology = CfdDomainTopology::from_model(&model, Some(prep_context));
    let (fluid_velocity, fluid_pressure) = recover_cfd_velocity_pressure(cfd_domain, topology, 0);
    let (residual_momentum, residual_continuity) =
        cfd_residual_norms(&fluid_velocity, &fluid_pressure, cfd_domain, topology, 4);
    let (fields, closure) = build_fsi_run_fields(
        cfd_domain,
        topology,
        4,
        fsi_structural_compliance_per_pa(&model),
        64,
        1.0e-8,
        &residual_momentum,
        &residual_continuity,
    );
    let pressure = fields
        .iter()
        .find(|field| field.field_id == fea_fsi_interface_pressure_field_id(0))
        .expect("FSI interface pressure field should be present");
    let traction = fields
        .iter()
        .find(|field| field.field_id == fea_fsi_interface_traction_field_id(0))
        .expect("FSI interface traction field should be present");

    assert_eq!(
        topology.basis,
        CfdDomainTopologyBasis::PrepControlVolumeConnectivity
    );
    assert_eq!(fluid_interface_face_count(topology), 8);
    assert_eq!(closure.interface_face_count, 8);
    assert_eq!(closure.structural_coupling_edge_count, 11);
    assert_eq!(pressure.shape, vec![8]);
    assert_eq!(traction.shape, vec![8, 3]);
}

#[test]
fn analysis_run_transient_with_options_controls_timeline() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];

    let envelope = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            time_step_s: 2.0e-3,
            min_time_step_s: 2.0e-3,
            max_time_step_s: 2.0e-3,
            step_count: 3,
            max_linear_iters: 64,
            tolerance: 1.0e-8,
            residual_target: 1.0e-6,
            adaptive_time_step: false,
            max_step_retries: 0,
            adapt_min_scale: 0.8,
            adapt_max_scale: 1.25,
            adapt_growth_exponent: 0.35,
            adapt_retry_growth_cap: 1.05,
            adapt_nonconverged_shrink: 0.75,
            dt_bucket_rel_tolerance: 0.0,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed with options");

    let transient = envelope
        .data
        .transient_results
        .as_ref()
        .expect("transient payload should exist");
    assert_eq!(transient.time_points_s.len(), 4);
    assert_eq!(transient.time_points_s[0], 0.0);
    assert!((transient.time_points_s[3] - 6.0e-3).abs() < 1.0e-12);
    assert!(envelope.data.provenance.deterministic_mode);
}

#[test]
fn analysis_run_transient_rejects_non_monotonic_thermo_time_profile() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    set_model_thermo_coupling(
        &mut model,
        ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 70.0,
            thermal_expansion_coefficient: 1.1e-5,
            field_artifact_id: None,
            field_source: Some(ThermoFieldSource {
                source_id: "field/transient-a".to_string(),
                revision: 1,
                interpolation_mode: Some(ThermoFieldInterpolationMode::Linear),
                expected_region_ids: vec!["tip".to_string()],
            }),
            region_temperature_deltas: vec![ThermoRegionTemperatureDelta {
                region_id: "tip".to_string(),
                temperature_delta_k: 70.0,
            }],
            time_profile: vec![
                ThermoTimeProfilePoint {
                    normalized_time: 0.8,
                    scale: 1.0,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 0.5,
                    scale: 0.9,
                },
            ],
        },
    );

    let err = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions::default(),
        OperationContext::new(None, None),
    )
    .expect_err("non-monotonic thermo time profile should be rejected");

    assert_eq!(err.error_code, "RM.FEA.RUN_TRANSIENT.INVALID_OPTIONS");
}

#[test]
fn analysis_run_nonlinear_rejects_unknown_thermo_expected_region_ids() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    set_model_thermo_coupling(
        &mut model,
        ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 80.0,
            thermal_expansion_coefficient: 1.2e-5,
            field_artifact_id: None,
            field_source: Some(ThermoFieldSource {
                source_id: "field/nonlinear-a".to_string(),
                revision: 2,
                interpolation_mode: Some(ThermoFieldInterpolationMode::Step),
                expected_region_ids: vec!["missing_region".to_string()],
            }),
            region_temperature_deltas: Vec::new(),
            time_profile: Vec::new(),
        },
    );

    let err = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions::production_recommended(),
        OperationContext::new(None, None),
    )
    .expect_err("unknown thermo expected region should be rejected");

    assert_eq!(err.error_code, "RM.FEA.RUN_NONLINEAR.INVALID_OPTIONS");
}

#[test]
fn analysis_run_nonlinear_rejects_invalid_plasticity_constitutive_options() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    set_model_plasticity(
        &mut model,
        PlasticityConstitutiveOptions {
            enabled: true,
            yield_strain: -1.0,
            hardening_modulus_ratio: 0.1,
            saturation_exponent: 1.0,
        },
    );

    let err = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions::default(),
        OperationContext::new(None, None),
    )
    .expect_err("nonlinear run should reject invalid plasticity options");

    assert_eq!(err.error_code, "RM.FEA.RUN_NONLINEAR.INVALID_OPTIONS");
}

#[test]
fn analysis_run_nonlinear_rejects_invalid_contact_interface_options() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    set_model_contact(
        &mut model,
        ContactInterfaceOptions {
            enabled: true,
            penalty_stiffness_scale: 0.0,
            max_penetration_ratio: 0.01,
            friction_coefficient: 0.0,
        },
    );

    let err = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions::default(),
        OperationContext::new(None, None),
    )
    .expect_err("nonlinear run should reject invalid contact options");

    assert_eq!(err.error_code, "RM.FEA.RUN_NONLINEAR.INVALID_OPTIONS");
}

#[test]
fn analysis_run_transient_can_resolve_thermo_field_artifact() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];

    let root = temp_artifact_root("transient-thermo-resolve").join("thermo-fields");
    let _root_guard = scoped_thermo_field_artifact_root(&root);
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("create thermo field artifact root");
    let mut field_artifact = serde_json::json!({
        "schema_version": "fea_thermo_field_artifact/v1",
        "source_geometry_id": model.geometry_id,
        "source_geometry_revision": model.geometry_revision,
        "artifact_status": "approved",
        "approved_by": "release-bot",
        "field_source": {
            "source_id": "artifact/transient-field",
            "revision": 1,
            "interpolation_mode": "linear",
            "expected_region_ids": [],
        },
        "region_temperature_deltas": [
            {"region_id": "tip", "temperature_delta_k": 72.0}
        ],
        "time_profile": [
            {"normalized_time": 0.0, "scale": 0.5},
            {"normalized_time": 1.0, "scale": 1.0}
        ]
    });
    let artifact_hash = thermo_field_payload_hash(
        &serde_json::from_value(field_artifact.clone()).expect("decode artifact struct"),
    );
    field_artifact["payload_hash"] = serde_json::Value::String(artifact_hash.clone());
    field_artifact["signature"] = serde_json::Value::String(thermo_field_signature(
        &artifact_hash,
        "release-bot",
        "runmat-dev-thermo-signing-key",
    ));
    fs::write(
        root.join("field_ok.json"),
        serde_json::to_vec_pretty(&field_artifact).expect("encode thermo field artifact"),
    )
    .expect("write thermo field artifact");
    set_model_thermo_coupling(
        &mut model,
        ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 70.0,
            thermal_expansion_coefficient: 1.1e-5,
            field_artifact_id: Some("field_ok".to_string()),
            field_source: None,
            region_temperature_deltas: Vec::new(),
            time_profile: Vec::new(),
        },
    );
    let run = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions::default(),
        OperationContext::new(None, None),
    )
    .expect("transient run should resolve thermo field artifact");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    let _ = fs::remove_dir_all(&root);

    assert!(results.data.summary.thermo_spatial_coverage_ratio.is_some());
    assert_eq!(results.data.summary.thermo_region_delta_count, Some(1.0));
}

#[test]
fn analysis_run_transient_rejects_missing_thermo_field_artifact() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];

    let root = temp_artifact_root("transient-thermo-missing").join("thermo-fields");
    let _root_guard = scoped_thermo_field_artifact_root(&root);
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("create empty thermo field artifact root");
    set_model_thermo_coupling(
        &mut model,
        ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 70.0,
            thermal_expansion_coefficient: 1.1e-5,
            field_artifact_id: Some("missing".to_string()),
            field_source: None,
            region_temperature_deltas: Vec::new(),
            time_profile: Vec::new(),
        },
    );

    let err = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions::default(),
        OperationContext::new(None, None),
    )
    .expect_err("missing thermo field artifact should be rejected");
    let _ = fs::remove_dir_all(&root);

    assert_eq!(err.error_code, "RM.FEA.RUN_THERMO_FIELD.NOT_FOUND");
}

#[test]
fn analysis_run_transient_artifact_backed_thermo_matches_inline_profile() {
    let _guard = analysis_test_guard();
    let mut model = sample_model_with_material_assignment_mismatch();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];

    let root = temp_artifact_root("transient-thermo-inline-parity").join("thermo-fields");
    let _root_guard = scoped_thermo_field_artifact_root(&root);
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("create thermo field artifact root");
    let mut inline_equivalent_artifact = serde_json::json!({
        "schema_version": "fea_thermo_field_artifact/v1",
        "source_geometry_id": model.geometry_id,
        "source_geometry_revision": model.geometry_revision,
        "artifact_status": "approved",
        "approved_by": "release-bot",
        "field_source": {
            "source_id": "artifact/inline-equivalent",
            "revision": 1,
            "interpolation_mode": "linear",
            "expected_region_ids": []
        },
        "region_temperature_deltas": [
            {"region_id": "tip", "temperature_delta_k": 90.0}
        ],
        "time_profile": [
            {"normalized_time": 0.0, "scale": 0.4},
            {"normalized_time": 1.0, "scale": 1.0}
        ]
    });
    let inline_hash = thermo_field_payload_hash(
        &serde_json::from_value(inline_equivalent_artifact.clone())
            .expect("decode inline artifact struct"),
    );
    inline_equivalent_artifact["payload_hash"] = serde_json::Value::String(inline_hash.clone());
    inline_equivalent_artifact["signature"] = serde_json::Value::String(thermo_field_signature(
        &inline_hash,
        "release-bot",
        "runmat-dev-thermo-signing-key",
    ));
    fs::write(
        root.join("inline_equivalent.json"),
        serde_json::to_vec_pretty(&inline_equivalent_artifact).expect("encode artifact"),
    )
    .expect("write artifact");

    let mut inline_model = model.clone();
    set_model_thermo_coupling(
        &mut inline_model,
        ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 90.0,
            thermal_expansion_coefficient: 1.2e-5,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: vec![ThermoRegionTemperatureDelta {
                region_id: "tip".to_string(),
                temperature_delta_k: 90.0,
            }],
            time_profile: vec![
                ThermoTimeProfilePoint {
                    normalized_time: 0.0,
                    scale: 0.4,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 1.0,
                    scale: 1.0,
                },
            ],
        },
    );
    let inline = analysis_run_transient_with_options_op(
        &inline_model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions::default(),
        OperationContext::new(None, None),
    )
    .expect("inline thermo run should succeed");

    let mut artifact_model = model;
    set_model_thermo_coupling(
        &mut artifact_model,
        ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 90.0,
            thermal_expansion_coefficient: 1.2e-5,
            field_artifact_id: Some("inline_equivalent".to_string()),
            field_source: None,
            region_temperature_deltas: Vec::new(),
            time_profile: Vec::new(),
        },
    );
    let artifact_backed = analysis_run_transient_with_options_op(
        &artifact_model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions::default(),
        OperationContext::new(None, None),
    )
    .expect("artifact-backed thermo run should succeed");
    let _ = fs::remove_dir_all(&root);

    let inline_transient = inline
        .data
        .transient_results
        .as_ref()
        .expect("inline transient payload");
    let artifact_transient = artifact_backed
        .data
        .transient_results
        .as_ref()
        .expect("artifact transient payload");
    let inline_final = inline_transient
        .displacement_snapshots
        .last()
        .and_then(|field| field.as_host_f64())
        .expect("inline host displacement");
    let artifact_final = artifact_transient
        .displacement_snapshots
        .last()
        .and_then(|field| field.as_host_f64())
        .expect("artifact host displacement");
    assert_eq!(inline_final.len(), artifact_final.len());
    let mut max_abs = 0.0_f64;
    for (lhs, rhs) in inline_final.iter().zip(artifact_final.iter()) {
        max_abs = max_abs.max((lhs - rhs).abs());
    }
    assert!(max_abs <= 1.0e-12);
}

#[test]
fn transient_balanced_degrades_when_thermo_mechanical_severity_is_high() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    for material in &mut model.materials {
        material.thermal.expansion_coefficient_per_k = 1.0e-3;
    }
    set_model_thermo_coupling(
        &mut model,
        ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 90.0,
            thermal_expansion_coefficient: 1.0e-3,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: Vec::new(),
            time_profile: Vec::new(),
        },
    );

    let run = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions {
            quality_policy: QualityPolicy::Balanced,
            adaptive_time_step: true,
            step_count: 8,
            ..AnalysisTransientRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect("transient run should return envelope");

    assert!(!run.data.publishable);
    assert_eq!(run.data.run_status, RunStatus::Degraded);
    assert!(run
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::ThermoMechanicalTransientStress));

    let transient = run
        .data
        .transient_results
        .as_ref()
        .expect("transient results should be present");
    assert_eq!(
        transient.thermo_mechanical_temperature_snapshots.len(),
        transient.time_points_s.len()
    );
    assert_eq!(
        transient.thermo_mechanical_thermal_strain_snapshots.len(),
        transient.time_points_s.len()
    );
    assert_eq!(
        transient.thermo_mechanical_thermal_stress_snapshots.len(),
        transient.time_points_s.len()
    );
    assert_eq!(
        transient.thermo_mechanical_displacement_snapshots.len(),
        transient.time_points_s.len()
    );
    assert_eq!(
        transient.thermo_mechanical_von_mises_snapshots.len(),
        transient.time_points_s.len()
    );
    assert_eq!(
        transient
            .thermo_mechanical_coupling_residual_snapshots
            .len(),
        transient.time_points_s.len()
    );
    assert_eq!(
        transient.thermo_mechanical_temperature_snapshots[0].field_id,
        fea_thermo_mechanical_temperature_field_id(0)
    );
    assert_eq!(
        transient.thermo_mechanical_thermal_strain_snapshots[0].field_id,
        fea_thermo_mechanical_thermal_strain_field_id(0)
    );
    assert_eq!(
        transient.thermo_mechanical_thermal_stress_snapshots[0].field_id,
        fea_thermo_mechanical_thermal_stress_field_id(0)
    );
    assert_eq!(
        transient.thermo_mechanical_displacement_snapshots[0].field_id,
        fea_thermo_mechanical_displacement_field_id(0)
    );
    assert_eq!(
        transient.thermo_mechanical_von_mises_snapshots[0].field_id,
        fea_thermo_mechanical_von_mises_field_id(0)
    );
    assert_eq!(
        transient.thermo_mechanical_coupling_residual_snapshots[0].field_id,
        fea_thermo_mechanical_coupling_residual_field_id(0)
    );

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("thermo-mechanical transient results should be queryable");
    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("transient thermo-mechanical descriptor should be present")
    };
    for field_id in [
        fea_thermo_mechanical_temperature_field_id(0),
        fea_thermo_mechanical_von_mises_field_id(0),
        fea_thermo_mechanical_coupling_residual_field_id(0),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Scalar);
        assert_eq!(descriptor.component_count, None);
    }
    let displacement_descriptor = descriptor(&fea_thermo_mechanical_displacement_field_id(0));
    assert_eq!(displacement_descriptor.kind, AnalysisFieldKind::Vector);
    assert_eq!(displacement_descriptor.component_count, Some(3));
    for field_id in [
        fea_thermo_mechanical_thermal_strain_field_id(0),
        fea_thermo_mechanical_thermal_stress_field_id(0),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Tensor);
        assert_eq!(descriptor.component_count, Some(6));
    }
}

#[test]
fn transient_balanced_degrades_when_thermo_heterogeneity_is_high() {
    let _guard = analysis_test_guard();
    let mut model = sample_model_with_material_assignment_mismatch();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    set_model_thermo_coupling(
        &mut model,
        ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 90.0,
            thermal_expansion_coefficient: 1.2e-5,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: Vec::new(),
            time_profile: Vec::new(),
        },
    );

    let run = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions {
            quality_policy: QualityPolicy::Balanced,
            adaptive_time_step: true,
            step_count: 8,
            ..AnalysisTransientRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect("transient run should return envelope");

    assert!(!run.data.publishable);
    assert_eq!(run.data.run_status, RunStatus::Degraded);
    assert!(run.data.quality_reasons.iter().any(|reason| {
        reason.code == QualityReasonCode::ThermoMechanicalConstitutiveSpreadHigh
            || reason.code == QualityReasonCode::ThermoMechanicalAssignmentHeterogeneityHigh
    }));
}

#[test]
fn nonlinear_balanced_degrades_when_plasticity_severity_is_high() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    set_model_plasticity(
        &mut model,
        PlasticityConstitutiveOptions {
            enabled: true,
            yield_strain: 2.0e-4,
            hardening_modulus_ratio: 0.2,
            saturation_exponent: 4.0,
        },
    );

    let run = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            quality_policy: QualityPolicy::Balanced,
            ..AnalysisNonlinearRunOptions::balanced()
        },
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should return envelope");

    assert!(!run.data.publishable);
    assert_eq!(run.data.run_status, RunStatus::Degraded);
    assert!(run
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::PlasticityNonlinearStress));
    assert!(run
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_PLASTIC_NONLINEAR"));
}

#[test]
fn nonlinear_balanced_degrades_when_contact_severity_is_high() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    set_model_contact(
        &mut model,
        ContactInterfaceOptions {
            enabled: true,
            penalty_stiffness_scale: 0.15,
            max_penetration_ratio: 0.035,
            friction_coefficient: 0.9,
        },
    );

    let run = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            quality_policy: QualityPolicy::Balanced,
            ..AnalysisNonlinearRunOptions::balanced()
        },
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should return envelope");

    assert!(!run.data.publishable);
    assert_eq!(run.data.run_status, RunStatus::Degraded);
    assert!(run
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::ContactNonlinearStress));
    assert!(run
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_CONTACT_NONLINEAR"));
}

#[test]
fn analysis_run_modal_returns_native_modal_result() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model_run".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");

    let envelope = analysis_run_modal_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("modal run should produce modal result");

    assert_eq!(envelope.operation, "fea.run_modal");
    assert_eq!(envelope.op_version, "fea.run_modal/v1");
    assert_eq!(
        envelope.data.run.solver_method,
        "matrix_free_subspace_iteration"
    );
    assert_eq!(
        envelope.data.provenance.solver_method,
        "matrix_free_subspace_iteration"
    );
    assert_eq!(envelope.data.run_status, RunStatus::Degraded);
    assert!(!envelope.data.publishable);
    let modal = envelope
        .data
        .modal_results
        .as_ref()
        .expect("modal payload should exist");
    assert!(!modal.eigenvalues_hz.is_empty());
    assert_eq!(modal.eigenvalues_hz.len(), modal.mode_shapes.len());
    assert_eq!(
        modal.mode_shapes[0].field_id,
        fea_modal_mode_shape_field_id(1)
    );
    assert_eq!(modal.mode_shapes[0].shape.len(), 2);
    assert_eq!(modal.mode_shapes[0].shape[1], 3);
    assert_eq!(modal.eigenvalues_hz.len(), modal.residual_norms.len());
    assert!(modal.residual_norms.iter().all(|value| value.is_finite()));
    assert_eq!(modal.modal_payload_version, "modal_results/v1");
    assert_eq!(modal.mode_units, ModalFrequencyUnits::Hz);
    assert_eq!(modal.frequency_basis, ModalFrequencyBasis::NativeEigenSolve);
    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("modal results should be queryable");
    let field_ids = results
        .data
        .field_descriptors
        .iter()
        .map(|descriptor| descriptor.field_id.as_str())
        .collect::<Vec<_>>();
    assert!(field_ids.contains(&FEA_FIELD_MODAL_FREQUENCY_HZ));
    assert!(field_ids.contains(&FEA_FIELD_MODAL_EIGENVALUE));
    assert!(field_ids.contains(&FEA_FIELD_MODAL_MODAL_MASS));
    assert!(field_ids.contains(&FEA_FIELD_MODAL_MODAL_STIFFNESS));
    assert!(field_ids.contains(&FEA_FIELD_MODAL_PARTICIPATION_FACTOR));
    assert!(field_ids.contains(&FEA_FIELD_MODAL_RESIDUAL_NORM));
    assert!(field_ids.contains(&FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION));
    let orthogonality_descriptor = results
        .data
        .field_descriptors
        .iter()
        .find(|descriptor| descriptor.field_id == FEA_FIELD_MODAL_M_ORTHOGONALITY)
        .expect("modal orthogonality descriptor should be present");
    assert_eq!(orthogonality_descriptor.kind, AnalysisFieldKind::Tensor);
    assert_eq!(
        orthogonality_descriptor.shape,
        vec![modal.eigenvalues_hz.len(), modal.eigenvalues_hz.len()]
    );
    assert_eq!(orthogonality_descriptor.component_count, None);
    assert!(envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::ModalResidualExceeded));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_MODAL_CONVERGENCE"));
}

#[test]
fn analysis_run_acoustic_returns_acoustic_fields_and_diagnostics() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let acoustic_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "acoustic_model_run".to_string(),
            profile: AnalysisCreateModelProfile::AcousticHarmonic,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("acoustic model should be created");

    let envelope = analysis_run_acoustic_op(
        &acoustic_model.data,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("acoustic run should produce acoustic fields");

    assert_eq!(envelope.operation, "fea.run_acoustic");
    assert_eq!(envelope.op_version, "fea.run_acoustic/v1");
    assert_eq!(
        envelope.data.run.solver_method,
        "acoustic_domain_graph_helmholtz_harmonic"
    );
    assert!(envelope.data.modal_results.is_none());
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_ACOUSTIC_HARMONIC_RESPONSE"));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_ACOUSTIC_DOMAIN_ASSEMBLY"));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_ACOUSTIC_DOMAIN_ASSEMBLY"
            && diag.message.contains("domain_edge_count=2")
            && diag.message.contains("domain_active_dimension_count=2")
    }));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_ACOUSTIC_HELMHOLTZ_RESIDUAL"));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_ACOUSTIC_FREQUENCY_RESPONSE"));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_ACOUSTIC_KNOWN_ANSWER"
            && diag.message.contains("tube_mode_alignment_error_ratio=")
            && diag.message.contains("tube_pressure_variation_ratio=")
            && diag.message.contains("cavity_mode_spacing_ratio=")
            && diag.message.contains("known_answer_coverage_ratio=1")
    }));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_ACOUSTIC_HARMONIC_RESPONSE"
            && diag.message.contains("acoustic_material_coverage_ratio=1")
    }));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_ACOUSTIC_BOUNDARY_MODEL"
            && diag.message.contains("acoustic_boundary_coverage_ratio=1")
            && diag.message.contains("rigid_wall_count=1")
    }));
    assert!(envelope
        .data
        .run
        .field(FEA_FIELD_ACOUSTIC_PRESSURE_REAL)
        .is_some());
    assert!(envelope
        .data
        .run
        .field(FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE)
        .is_some());
    assert!(envelope
        .data
        .run
        .field(FEA_FIELD_ACOUSTIC_PARTICLE_VELOCITY)
        .is_some());
    let response_field_count = envelope
        .data
        .run
        .fields
        .iter()
        .filter(|field| field.field_id.starts_with("acoustic.frequency_response."))
        .count();
    assert_eq!(response_field_count, 3);
    let expected_first_response_hz = 125.0_f64 * 3.0 * 3.0_f64.sqrt() * 0.75;
    assert!(envelope
        .data
        .run
        .field(&fea_acoustic_frequency_response_field_id(
            expected_first_response_hz
        ))
        .is_some());
    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("acoustic results should be queryable");
    let field_ids = results
        .data
        .field_descriptors
        .iter()
        .map(|descriptor| descriptor.field_id.as_str())
        .collect::<Vec<_>>();
    assert!(field_ids.contains(&FEA_FIELD_ACOUSTIC_PRESSURE_REAL));
    assert!(field_ids.contains(&FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE));
    assert!(field_ids.contains(&FEA_FIELD_ACOUSTIC_PARTICLE_VELOCITY));
    assert!(field_ids
        .iter()
        .any(|field_id| field_id.starts_with("acoustic.frequency_response.")));
}

#[test]
fn analysis_run_modal_with_options_controls_requested_mode_count() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model_run_opts".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");

    let envelope = analysis_run_modal_with_options_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        AnalysisModalRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            mode_count: 2,
            residual_warn_threshold: 1.0e-2,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(None, None),
    )
    .expect("modal run should succeed with options");

    let modal = envelope
        .data
        .modal_results
        .as_ref()
        .expect("modal payload should exist");
    assert!(!modal.eigenvalues_hz.is_empty());
    assert!(modal.eigenvalues_hz.len() <= 2);
    assert!(envelope.data.provenance.deterministic_mode);
}

#[test]
fn analysis_results_include_modal_payload_for_modal_runs() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model_results".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");

    let run = analysis_run_modal_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("modal run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    let modal = results
        .data
        .modal_results
        .as_ref()
        .expect("modal payload should propagate to results");
    assert!(!modal.eigenvalues_hz.is_empty());
    assert_eq!(modal.eigenvalues_hz.len(), modal.mode_shapes.len());
    assert_eq!(modal.mode_shapes[0].shape.len(), 2);
    assert_eq!(modal.mode_shapes[0].shape[1], 3);
    assert_eq!(modal.eigenvalues_hz.len(), modal.residual_norms.len());
    assert_eq!(modal.modal_payload_version, "modal_results/v1");
    assert_eq!(modal.mode_units, ModalFrequencyUnits::Hz);
    assert_eq!(modal.frequency_basis, ModalFrequencyBasis::NativeEigenSolve);
    assert!(results.data.summary.mode_count > 0);
    assert_eq!(
        results.data.summary.mode_count,
        results.data.summary.available_mode_indices.len()
    );
    assert!(results.data.summary.min_frequency_hz.is_some());
    assert!(results.data.summary.max_frequency_hz.is_some());
    assert!(results.data.summary.max_modal_residual_norm.is_some());
    assert!(results.data.summary.first_mode_converged.is_some());
}

#[test]
fn analysis_results_query_can_exclude_modal_payload() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model_results_filter".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");
    let run = analysis_run_modal_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("modal run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_field_values: true,

            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: false,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
            include_electromagnetic_results: true,
        },
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    assert!(results.data.modal_results.is_none());
}

#[test]
fn analysis_results_query_rejects_unknown_modal_mode_index() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model_results_index".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");
    let run = analysis_run_modal_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("modal run should succeed");

    let err = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_field_values: true,

            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: vec![10],
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
            include_electromagnetic_results: true,
        },
        OperationContext::new(None, None),
    )
    .expect_err("results should fail for unknown mode index");

    assert_eq!(err.error_code, "RM.FEA.RESULTS.MODE_NOT_FOUND");
    assert_eq!(err.operation, "fea.results");
    assert_eq!(err.op_version, "fea.results/v1");
}

#[test]
fn analysis_results_include_transient_payload_for_transient_runs() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    let run = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    let transient = results
        .data
        .transient_results
        .as_ref()
        .expect("transient payload should propagate");
    assert_eq!(
        transient.integration_method,
        TransientIntegrationMethod::ImplicitEuler
    );
    assert!(!transient.time_points_s.is_empty());
    assert_eq!(
        transient.time_points_s.len(),
        transient.displacement_snapshots.len()
    );
}

#[test]
fn analysis_results_include_electromagnetic_fields_for_em_runs() {
    let _guard = analysis_test_guard();
    let spec = sample_electromagnetic_study_spec();
    let run = analysis_run_study_op(&spec, OperationContext::new(None, None))
        .expect("electromagnetic study should run");

    let results = analysis_results_by_run_id_op(
        &run.data.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("electromagnetic results should load");
    let field_ids = results
        .data
        .field_descriptors
        .iter()
        .map(|field| field.field_id.as_str())
        .collect::<Vec<_>>();

    assert!(field_ids.contains(&FEA_FIELD_EM_VECTOR_POTENTIAL_REAL));
    assert!(field_ids.contains(&FEA_FIELD_EM_MAGNETIC_FLUX_DENSITY_MAGNITUDE));
}

#[cfg(feature = "plot-core")]
#[test]
fn analysis_generate_study_run_figures_returns_mesh_figures() {
    let _guard = analysis_test_guard();
    let spec = sample_linear_static_study_spec();
    let run = analysis_run_study_op(&spec, OperationContext::new(None, None))
        .expect("linear static study should run");

    let figures = analysis_generate_study_run_figures(
        &spec,
        &run.data.run_id,
        AnalysisFigureGenerationOptions {
            include_comparison: false,
            include_trends: false,
            ..AnalysisFigureGenerationOptions::default()
        },
    )
    .expect("study figures should be generated");

    assert!(figures
        .iter()
        .any(|figure| matches!(figure.kind, AnalysisGeneratedFigureKind::MeshResult)));
    assert!(figures.iter().any(|figure| !figure.figure.is_empty()));
}

#[test]
fn analysis_results_query_can_exclude_transient_payload() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    let run = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_field_values: true,

            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: false,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
            include_electromagnetic_results: true,
        },
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    assert!(results.data.transient_results.is_none());
    assert!(results.data.summary.snapshot_count > 0);
    assert_eq!(results.data.summary.time_start_s, Some(0.0));
    assert!(results.data.summary.time_end_s.unwrap_or(0.0) > 0.0);
    assert!(results.data.summary.max_transient_residual_norm.is_some());
    assert!(results.data.summary.final_step_converged.is_some());
}

#[test]
fn analysis_results_query_rejects_unknown_transient_snapshot_index() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    let run = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed");

    let err = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_field_values: true,

            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: vec![999],
            include_nonlinear_results: true,
            include_electromagnetic_results: true,
        },
        OperationContext::new(None, None),
    )
    .expect_err("results should fail for unknown transient snapshot index");

    assert_eq!(
        err.error_code,
        "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND"
    );
    assert_eq!(err.operation, "fea.results");
    assert_eq!(err.op_version, "fea.results/v1");
}
