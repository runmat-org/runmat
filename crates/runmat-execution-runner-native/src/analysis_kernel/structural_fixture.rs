//! One original structural fixture for native frequency and modal tests.
use runmat_analysis_core::{
    AnalysisConstraint, ExcitationTarget, PrescribedStructuralDofs, RegionTarget,
    StructuralConstraint, StructuralConstraintKind,
};
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_meshing_core::ElementOrder;

pub(super) fn model_and_mesh(
    order: ElementOrder,
) -> (
    runmat_analysis_core::AnalysisModel,
    runmat_meshing_core::SolverMeshArtifact,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    *model.material_assignments[0]
        .target
        .selected_region_id_mut()
        .expect("fixture uses selected material region") =
        runmat_meshing_core::fixtures::fixture_region("region");
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "fixed-face".into(),
        frame: runmat_analysis_core::CoordinateFrame::Global,
        kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:0")),
            translations: [
                runmat_analysis_core::StructuralTranslationDof::Ux,
                runmat_analysis_core::StructuralTranslationDof::Uy,
                runmat_analysis_core::StructuralTranslationDof::Uz,
            ]
            .into_iter()
            .map(|dof| runmat_analysis_core::TranslationDofValue {
                dof,
                value: runmat_analysis_core::PrescribedValue::Constant(
                    runmat_analysis_core::Displacement::new(0.0),
                ),
            })
            .collect(),
            rotations: Vec::new(),
        }),
    })];
    model.excitations[0].excitation_mut().target = ExcitationTarget::Region(
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:3")),
    );
    (
        model,
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order),
    )
}
