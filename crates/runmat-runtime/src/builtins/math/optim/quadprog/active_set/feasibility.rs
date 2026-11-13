//! Genuine original-row phase one, under the same iteration/time owner.
use super::*;
use num_rational::BigRational;
struct Original<'a, V> {
    roles: &'a [conic::LinearRowRole],
    validation: &'a mut V,
}
impl<V: compiled::Validation> conic::solve::TerminalValidation for Original<'_, V> {
    async fn farkas(
        &mut self,
        weights: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.validation
            .certificate(
                compiled::CertificateCandidate::Farkas {
                    rows: self.roles,
                    multipliers: weights,
                },
                c,
            )
            .await
    }
    async fn ray(&mut self, _: &[BigRational], _: &mut ScalarCooperation) -> BuiltinResult<bool> {
        Ok(false)
    }
    async fn feasible(
        &mut self,
        _: &[BigRational],
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(false)
    }
}
pub(super) async fn solve<V: compiled::Validation>(
    model: &Model,
    seed: &[f64],
    controls: &mut options::Controls,
    validation: &mut V,
    c: &mut ScalarCooperation,
) -> BuiltinResult<conic::solve::Candidate> {
    let zero = zeros(model.f.len(), c).await?;
    let phase = conic::Model {
        p: None,
        f: &zero,
        a: &model.a,
        b: &model.b,
        cones: &model.cones,
        identity: OWNER,
        objective_scale: 1.,
    };
    conic::solve::candidate_validated(
        &phase,
        seed,
        controls,
        c,
        &mut Original {
            roles: &model.roles,
            validation,
        },
    )
    .await
}
