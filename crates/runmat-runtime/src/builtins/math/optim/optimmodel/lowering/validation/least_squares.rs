//! Same original point/cache and certificate authority for residual lifting.
use super::*;
use crate::builtins::math::optim::lsqlin::compiled as lsq;
async fn roles(
    rows: lsq::Rows<'_>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Vec<RowRole>> {
    let mut out = Vec::new();
    for role in rows.iter() {
        out.try_reserve(1).map_err(|e| policy.reserve(e))?;
        out.push(role);
        c.completed(1).await?;
    }
    Ok(out)
}
impl lsq::Validation for Validator<'_> {
    async fn stopped_candidate(
        &mut self,
        point: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.stopped_original(point, c).await
    }
    async fn objective_limit(
        &mut self,
        point: &[f64],
        _cost: f64,
        limit: f64,
        feasibility: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<bool>> {
        self.objective_limit_original(point, limit, feasibility, c)
            .await
    }
    async fn solution(
        &mut self,
        point: &[f64],
        rows: lsq::Rows<'_>,
        duals: &[f64],
        feasibility: f64,
        optimality: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if !self.prepare(point, c).await?
            || !self.primal_at_tolerance(point, feasibility, c).await?
        {
            return Ok(false);
        }
        let Some(duals) = rows.external(duals) else {
            return Ok(false);
        };
        let roles = roles(rows, self.policy, c).await?;
        self.authority
            .kkt_at_tolerance(point, &roles, duals, optimality, c)
            .await
    }
    async fn incumbent(
        &mut self,
        point: &[f64],
        feasibility: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if !self.prepare(point, c).await? {
            return Ok(false);
        }
        self.primal_at_tolerance(point, feasibility, c).await
    }
    async fn certificate(
        &mut self,
        candidate: lsq::Certificate<'_>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        match candidate {
            lsq::Certificate::InconsistentBounds {
                column,
                lower,
                upper,
            } => {
                self.authority
                    .certificate(
                        CertificateCandidate::InconsistentBounds {
                            column,
                            lower,
                            upper,
                        },
                        c,
                    )
                    .await
            }
            lsq::Certificate::Farkas { rows, multipliers } => {
                let Some(multipliers) = rows.external(multipliers) else {
                    return Ok(false);
                };
                let roles = roles(rows, self.policy, c).await?;
                self.authority
                    .certificate(
                        CertificateCandidate::Farkas {
                            rows: &roles,
                            multipliers,
                        },
                        c,
                    )
                    .await
            }
        }
    }
}
