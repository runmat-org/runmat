use crate::FeaRunError;
use runmat_analysis_core::{
    AnalysisModel, AnalysisProcedure, ExcitationAction, ExcitationKinematics, PhysicsFamily,
    RestartPolicy, SignalDomainKind,
};

pub(crate) fn require_fixed_structural_nonlinear(model: &AnalysisModel) -> Result<(), FeaRunError> {
    if model.sequence.steps.len() != 1
        || model.sequence.steps[0].family != PhysicsFamily::Structural
        || model.sequence.steps[0].procedure != AnalysisProcedure::Nonlinear
    {
        return Err(FeaRunError::InvalidModel(
            "J2 requires one uncoupled structural nonlinear authored step".into(),
        ));
    }
    if model.sequence.steps[0].restart != RestartPolicy::Fresh {
        return Err(FeaRunError::InvalidModel(
            "J2 restart requires an admitted equilibrated constitutive-history transition; fresh execution cannot consume a restart request".into(),
        ));
    }
    if !model.initial_conditions.is_empty() {
        return Err(FeaRunError::InvalidModel(
            "J2 initial conditions require an equilibrated constitutive-history transition".into(),
        ));
    }
    if !model.boundaries.is_empty() {
        return Err(FeaRunError::InvalidModel(
            "J2 boundary operators are not realized in this envelope; use supported typed mechanical pressure/traction excitations".into(),
        ));
    }
    for action in &model.excitations {
        if !matches!(
            action,
            ExcitationAction::Force(_)
                | ExcitationAction::Traction(_)
                | ExcitationAction::Pressure(_)
                | ExcitationAction::BodyForce(_)
        ) || action.excitation().kinematics != ExcitationKinematics::FixedFrame
        {
            return Err(FeaRunError::InvalidModel("J2 supports fixed-frame mechanical dead loads only; no follower or coupled actions".into()));
        }
        let signal = model
            .signals
            .iter()
            .find(|signal| signal.signal_id() == &action.excitation().signal_id)
            .ok_or_else(|| {
                FeaRunError::InvalidModel("J2 dead load has no physical signal".into())
            })?;
        if signal.domain() != SignalDomainKind::Constant || signal.generator().is_some() {
            return Err(FeaRunError::InvalidModel(
                "J2 dead loads require constant retained signals, not evolving or executable laws"
                    .into(),
            ));
        }
    }
    Ok(())
}
