//! Class identities of the typed FEA model objects. The catalog declares each
//! public spelling once; the runtime derives its class names from these.

use runmat_types::StaticClassIdentity;

pub const FEA_ACOUSTIC_BOUNDARY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.AcousticBoundary");
pub const FEA_ACOUSTIC_CONSTRAINT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.AcousticConstraint");
pub const FEA_ACOUSTIC_EXCITATION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.AcousticExcitation");
pub const FEA_ACOUSTIC_INITIAL_CONDITION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.AcousticInitialCondition");
pub const FEA_ACOUSTIC_STEP_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.AcousticStep");
pub const FEA_ACTIVATION_LIFECYCLE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ActivationLifecycle");
pub const FEA_BOUNDARY_NORMAL_TANGENT_FRAME_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.BoundaryNormalTangentFrame");
pub const FEA_CARTESIAN_COORDINATE_FRAME_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.CartesianCoordinateFrame");
pub const FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ConjugateHeatTransferInterface");
pub const FEA_CONJUGATE_HEAT_TRANSFER_STEP_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ConjugateHeatTransferStep");
pub const FEA_CONTACT_INTERFACE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ContactInterface");
pub const FEA_COORDINATE_FRAME_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.CoordinateFrame");
pub const FEA_CYLINDRICAL_COORDINATE_FRAME_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.CylindricalCoordinateFrame");
pub const FEA_DIRECTION_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Direction");
pub const FEA_ELECTRICAL_CONSTRAINT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ElectricalConstraint");
pub const FEA_ELECTRICAL_EXCITATION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ElectricalExcitation");
pub const FEA_ELECTRICAL_INITIAL_CONDITION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ElectricalInitialCondition");
pub const FEA_ELECTRO_THERMAL_DOMAIN_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ElectroThermalDomain");
pub const FEA_ELECTRO_THERMAL_STEP_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ElectroThermalStep");
pub const FEA_ELECTROMAGNETIC_DOMAIN_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ElectromagneticDomain");
pub const FEA_ELECTROMAGNETIC_STEP_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ElectromagneticStep");
pub const FEA_ELEMENT_COORDINATE_FRAME_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ElementCoordinateFrame");
pub const FEA_EXCITATION_CASE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ExcitationCase");
pub const FEA_FLUID_BOUNDARY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.FluidBoundary");
pub const FEA_FLUID_CONSTRAINT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.FluidConstraint");
pub const FEA_FLUID_DOMAIN_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.FluidDomain");
pub const FEA_FLUID_EXCITATION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.FluidExcitation");
pub const FEA_FLUID_INITIAL_CONDITION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.FluidInitialCondition");
pub const FEA_FLUID_STEP_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.FluidStep");
pub const FEA_FLUID_STRUCTURE_INTERFACE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.FluidStructureInterface");
pub const FEA_FLUID_STRUCTURE_STEP_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.FluidStructureStep");
pub const FEA_GEOMETRIC_PERIODIC_PAIRING_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.GeometricPeriodicPairing");
pub const FEA_GLOBAL_COORDINATE_FRAME_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.GlobalCoordinateFrame");
pub const FEA_IDENTITY_PERIODIC_TRANSFORM_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.IdentityPeriodicTransform");
pub const FEA_INITIAL_CONDITION_TARGET_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.InitialConditionTarget");
pub const FEA_MAGNETIC_BOUNDARY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.MagneticBoundary");
pub const FEA_MAGNETIC_CONSTRAINT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.MagneticConstraint");
pub const FEA_MAGNETIC_EXCITATION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.MagneticExcitation");
pub const FEA_MAGNETIC_INITIAL_CONDITION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.MagneticInitialCondition");
pub const FEA_MATCHING_PERIODIC_PAIRING_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.MatchingPeriodicPairing");
pub const FEA_MATERIAL_COORDINATE_FRAME_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.MaterialCoordinateFrame");
pub const FEA_MECHANICAL_DAMPING_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.MechanicalDamping");
pub const FEA_PERIODIC_PAIRING_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.PeriodicPairing");
pub const FEA_PERIODIC_TRANSFORM_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.PeriodicTransform");
pub const FEA_REFERENCE_POINT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ReferencePoint");
pub const FEA_ROTATION_PERIODIC_TRANSFORM_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.RotationPeriodicTransform");
pub const FEA_SPHERICAL_COORDINATE_FRAME_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.SphericalCoordinateFrame");
pub const FEA_STEP_OBJECT_ACTIVATION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StepObjectActivation");
pub const FEA_STEP_OBJECT_REF_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StepObjectRef");
pub const FEA_STRUCTURAL_BOUNDARY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StructuralBoundary");
pub const FEA_STRUCTURAL_CONSTRAINT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StructuralConstraint");
pub const FEA_STRUCTURAL_DOF_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StructuralDof");
pub const FEA_STRUCTURAL_DOF_VALUE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StructuralDofValue");
pub const FEA_STRUCTURAL_EXCITATION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StructuralExcitation");
pub const FEA_STRUCTURAL_INITIAL_CONDITION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StructuralInitialCondition");
pub const FEA_STRUCTURAL_MOTION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StructuralMotion");
pub const FEA_STRUCTURAL_ROTATION_DOF_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StructuralRotationDof");
pub const FEA_STRUCTURAL_ROTATION_DOF_VALUE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StructuralRotationDofValue");
pub const FEA_STRUCTURAL_STEP_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StructuralStep");
pub const FEA_STRUCTURAL_TRANSLATION_DOF_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StructuralTranslationDof");
pub const FEA_STRUCTURAL_TRANSLATION_DOF_VALUE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StructuralTranslationDofValue");
pub const FEA_STRUCTURAL_TRANSLATION_MOTION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StructuralTranslationMotion");
pub const FEA_THERMAL_BOUNDARY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ThermalBoundary");
pub const FEA_THERMAL_CONSTRAINT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ThermalConstraint");
pub const FEA_THERMAL_DOMAIN_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ThermalDomain");
pub const FEA_THERMAL_EXCITATION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ThermalExcitation");
pub const FEA_THERMAL_INITIAL_CONDITION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ThermalInitialCondition");
pub const FEA_THERMAL_STEP_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.ThermalStep");
pub const FEA_THERMO_MECHANICAL_DOMAIN_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ThermoMechanicalDomain");
pub const FEA_THERMO_MECHANICAL_STEP_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ThermoMechanicalStep");
pub const FEA_TRANSLATION_PERIODIC_TRANSFORM_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.TranslationPeriodicTransform");

/// Every concrete coordinate-frame class a frame-valued argument may hold.
pub const FEA_COORDINATE_FRAME_CLASSES: &[StaticClassIdentity] = &[
    FEA_GLOBAL_COORDINATE_FRAME_CLASS,
    FEA_MATERIAL_COORDINATE_FRAME_CLASS,
    FEA_ELEMENT_COORDINATE_FRAME_CLASS,
    FEA_BOUNDARY_NORMAL_TANGENT_FRAME_CLASS,
    FEA_CARTESIAN_COORDINATE_FRAME_CLASS,
    FEA_CYLINDRICAL_COORDINATE_FRAME_CLASS,
    FEA_SPHERICAL_COORDINATE_FRAME_CLASS,
];
