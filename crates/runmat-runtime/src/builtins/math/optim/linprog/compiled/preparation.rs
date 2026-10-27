//! Original geometry, options and bound checks precede numerical admission.
use super::*;
use futures::future::Either;
pub(super) async fn run<'a, V: Validation>(
    problem: Problem,
    options: Option<&Value>,
    products: Products,
    cooperation: &'a mut ScalarCooperation,
    validation: &'a mut V,
) -> BuiltinResult<Either<Outcome, Ready<'a, V>>> {
    crate::context::host_allocation::require_unbounded_workspace(
        cooperation.runtime(),NAME,
        "finite host-memory execution is unsupported for sparse LP factorization and exact certificates",
    )?;
    validate(&problem, cooperation).await?;
    // The existing controls parser consumes only its actual solver profile and
    // retains the original policy. No ambient context is acquired by this port.
    let runtime = cooperation.runtime().cloned();
    let controls =
        super::super::options::Controls::parse(options, runtime.as_ref(), cooperation).await?;
    for (column, (&lower, &upper)) in problem.lb.iter().zip(&problem.ub).enumerate() {
        if lower > upper || lower == f64::INFINITY || upper == f64::NEG_INFINITY {
            if validation
                .certificate(
                    CertificateCandidate::InconsistentBounds {
                        column,
                        lower,
                        upper,
                    },
                    cooperation,
                )
                .await?
            {
                return Ok(Either::Left(Outcome::infeasible(
                    "No finite feasible point exists within the supplied bounds.",
                )));
            }
            return Ok(Either::Left(Outcome::failure(
                -4,
                0,
                "The assembled bound conflict was not verified against the original model domain.",
            )));
        }
        cooperation.completed(1).await?;
    }

    Ok(Either::Right(Ready {
        problem,
        controls,
        products,
        cooperation,
        validation,
    }))
}
