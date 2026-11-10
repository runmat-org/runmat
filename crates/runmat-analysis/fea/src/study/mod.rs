//! Placement-neutral compilation of realized studies onto exact solver topology.

mod acoustic_planning;
mod acoustic_product;
mod acoustic_response_planning;
mod acoustic_response_product;
mod compilation;
mod electrical_planning;
mod electrical_product;
mod fluid_material;
mod fluid_planning;
mod fluid_product;
mod identity;
mod magnetic_planning;
mod magnetic_product;
mod maxwell_planning;
mod maxwell_product;
mod nonlinear_product;
mod planning;
mod product;
mod thermal_planning;
mod thermal_product;

pub use acoustic_planning::{
    plan_acoustic_study, AcousticStudyPlan, AcousticStudyPlanningError,
    AcousticStudyPlanningPolicy, PlannedAcousticFrequency,
};
pub use acoustic_product::{
    build_acoustic_study_product, plan_acoustic_study_product, AcousticStudyProduct,
    AcousticStudyProductError, AcousticStudyProductInput,
};
pub use acoustic_response_planning::{
    is_acoustic_response_study, plan_acoustic_response_study, AcousticResponseStudyPlanningError,
    AcousticResponseStudyPlanningPolicy,
};
pub use acoustic_response_product::{
    build_acoustic_response_study_product, plan_acoustic_response_study_product,
    AcousticResponseStudyProduct, AcousticResponseStudyProductError,
    AcousticResponseStudyProductInput,
};
pub use compilation::{
    compile_analysis_study, compile_cross_section_study, AnalysisStudyCompilationError,
    AnalysisStudyCompilationInput, AnalysisStudyDefinition, CompiledAnalysisStudy,
};
pub use electrical_planning::{
    plan_electrical_study, ElectricalStudyPlanningError, ElectricalStudyPlanningPolicy,
};
pub use electrical_product::{
    build_electrical_study_product, plan_electrical_study_product, ElectricalStudyProduct,
    ElectricalStudyProductError, ElectricalStudyProductInput,
};
pub use fluid_material::{
    derive_canonical_fluid_material, derive_fluid_cells, is_flow_step, FluidCells,
    FluidMaterialCompilationError,
};
pub use fluid_planning::{plan_fluid_study, FluidStudyPlanningError, FluidStudyPlanningPolicy};
pub use fluid_product::{
    build_fluid_study_product, plan_fluid_study_product, FluidStudyProduct, FluidStudyProductError,
    FluidStudyProductInput,
};
pub use identity::{
    structural_study_source_digest, StructuralStudyMaterialBinding,
    StructuralStudySourceIdentityError,
};
pub use magnetic_planning::{
    plan_magnetic_study, MagneticStudyPlanningError, MagneticStudyPlanningPolicy,
};
pub use magnetic_product::{
    build_magnetic_study_product, plan_magnetic_study_product, MagneticStudyProduct,
    MagneticStudyProductError, MagneticStudyProductInput,
};
pub use maxwell_planning::{
    plan_maxwell_study, MaxwellStudyPlanningError, MaxwellStudyPlanningPolicy,
};
pub use maxwell_product::{
    build_maxwell_study_product, plan_maxwell_study_product, MaxwellStudyProduct,
    MaxwellStudyProductError, MaxwellStudyProductInput,
};
pub use nonlinear_product::{
    build_nonlinear_solid_study_product, plan_nonlinear_solid_study_product,
    NonlinearSolidStudyPlanningError, NonlinearSolidStudyProduct, NonlinearSolidStudyProductInput,
};
pub use planning::{
    plan_structural_study, plan_structural_study_with_fields, StructuralNumericalPlanningPolicy,
    StructuralStudyPlanningError, StructuralStudyPlanningPolicy,
};
pub use product::{
    build_structural_study_product, build_structural_study_product_with_fields,
    StructuralStudyProduct, StructuralStudyProductError, StructuralStudyProductInput,
};
pub use thermal_planning::{
    plan_thermal_step, plan_thermal_step_with_fields, plan_thermal_study,
    plan_thermal_study_with_fields, thermal_step_initial_selection_supported, ElementHeatSource,
    ThermalStudyPlanningError, ThermalStudyPlanningPolicy,
};
pub use thermal_product::{
    build_thermal_study_product, plan_thermal_study_product,
    plan_thermal_study_product_with_fields, ThermalStudyProduct, ThermalStudyProductError,
    ThermalStudyProductInput,
};
