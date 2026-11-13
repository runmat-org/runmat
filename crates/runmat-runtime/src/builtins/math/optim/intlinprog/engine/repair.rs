//! Same existing integer-tree repair owner, with original source validation retained.
use super::*;
pub(super) async fn repair<V: compiled::Validation>(
    workspace: &mut Workspace,
    proposal: &[f64],
    controls: &mut controls::Controls,
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<Option<Incumbent>> {
    if proposal.len() != workspace.n() || controls.time_left() == 0.0 {
        return Ok(None);
    }
    if !workspace.is_mixed() {
        return proofs::incumbent(&workspace.view(), &[], &[], proposal, cooperation).await;
    }
    for i in 0..workspace.semis.len() {
        let domain = workspace.semis[i];
        let value = proposal[domain.column];
        if !value.is_finite() {
            return Ok(None);
        }
        if value < domain.lower / 2.0 {
            workspace.tighten(Bound::Lower {
                column: domain.column,
                value: 0.0,
            });
            workspace.tighten(Bound::Upper {
                column: domain.column,
                value: 0.0,
            });
        } else {
            workspace.tighten(Bound::Lower {
                column: domain.column,
                value: domain.lower,
            });
        }
        cooperation.completed(1).await?;
    }
    for i in 0..workspace.integers.len() {
        let column = workspace.integers[i];
        let value = match workspace
            .semis
            .binary_search_by_key(&column, |domain| domain.column)
        {
            Ok(index) => {
                let domain = workspace.semis[index];
                if proposal[column] < domain.lower / 2.0 {
                    0.0
                } else if domain.lower > domain.upper.floor() {
                    return Ok(None);
                } else {
                    proposal[column]
                        .round()
                        .max(domain.lower)
                        .min(domain.upper.floor())
                }
            }
            Err(_) => proposal[column].round(),
        };
        if !value.is_finite() {
            return Ok(None);
        }
        workspace.tighten(Bound::Lower { column, value });
        workspace.tighten(Bound::Upper { column, value });
        cooperation.completed(1).await?;
    }
    if !workspace.consistent() {
        return Ok(None);
    }
    workspace.publish_bounds(cooperation).await?;
    let seed = workspace.seed(cooperation).await?;
    let model = workspace.view();
    let result = {
        let node = workspace.node();
        let mut original = compiled::Terminal {
            node: &node,
            validation: &mut *validation,
        };
        solve::candidate_validated(&model, &seed, controls, cooperation, &mut original).await?
    };
    match result {
        Candidate::Converged { x, .. } | Candidate::Stopped { x: Some(x), .. } => {
            proofs::incumbent(
                &model,
                &workspace.integers,
                &workspace.semis,
                &x,
                cooperation,
            )
            .await
        }
        Candidate::CertifiedUnbounded {
            direction,
            feasible,
        } => {
            drop(direction);
            proofs::from_verified(
                &model,
                &workspace.integers,
                &workspace.semis,
                feasible,
                cooperation,
            )
            .await
        }
        _ => Ok(None),
    }
}
