//! Restarted native Krylov/Rayleigh-Ritz execution for borrowed sparse and
//! selected callable operators. No dense coefficient fallback enters here.
mod basis;
mod class;
mod error;
mod geometry;
mod operator;
mod projected;
mod probe;
mod publication;
mod request;
mod residual_norm;
mod ritz;
mod scalar;
mod selection;
mod storage;
mod work;
#[cfg(test)]
mod tests;
use crate::{
    call::BuiltinArguments,
    context::{host_allocation::HostAllocationReceipt, RuntimeContext},
    execution::RootedValue,
    BuiltinResult,
};
use error::{admission, factor_error, invalid, matrix_error, operator_error, unsupported};
use runmat_value::{ComplexElement, NumericDType, Value};
use scalar::Scalar;

pub(super) async fn evaluate(
    mut args: BuiltinArguments,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    let requested = crate::output_count::current_output_count();
    if requested.is_some_and(|n| n > 3) {
        return Err(invalid("eigs has at most three outputs"));
    }
    let a = RootedValue::in_optional_context(
        args.pop_front()
            .ok_or_else(|| invalid("missing coefficient"))?,
        runtime,
    )?;
    let request = request::Request::parse(a.value(), &args, runtime).await?;
    let preliminary = class::Class::request(a.value(), &request);
    let probe =
        if crate::builtins::math::linalg::factor::iterative::controls::is_callable(a.value()) {
            Some(
                probe::select(
                    a.value(),
                    &request,
                    runtime
                        .ok_or_else(|| unsupported("callable eigs requires execution context"))?,
                    preliminary,
                )
                .await?,
            )
        } else {
            None
        };
    let class = probe.as_ref().map_or(preliminary, |probe| probe.class);
    match class {
        class::Class::RealF32 => run::<f32>(a.value(), request, runtime, requested, probe).await,
        class::Class::RealF64 => run::<f64>(a.value(), request, runtime, requested, probe).await,
        class::Class::ComplexF32 => {
            run::<ComplexElement<f32>>(a.value(), request, runtime, requested, probe).await
        }
        class::Class::ComplexF64 => {
            run::<ComplexElement<f64>>(a.value(), request, runtime, requested, probe).await
        }
    }
}

async fn run<'a, T: Scalar>(
    a: &'a Value,
    mut request: request::Request<'a>,
    runtime: Option<&RuntimeContext>,
    requested: Option<usize>,
    probe: Option<probe::Probe>,
) -> BuiltinResult<Value> {
    let coefficient_complex = if super::super::iterative::controls::is_callable(a) {
        probe.as_ref().is_some_and(|probe| probe.class.complex())
    } else {
        [Some(a), request.b]
            .into_iter()
            .flatten()
            .filter_map(class::Class::of)
            .any(class::Class::complex)
    };
    let sigma = match request
        .imaginary_alias
        .resolve(request.sigma, coefficient_complex)
    {
        super::Sigma::Near(value) => super::Sigma::Near(
            T::native(value)
                .ok_or_else(|| invalid("numeric sigma exceeds selected native range"))?
                .complex(),
        ),
        other => other,
    };
    let (callable, probe_input, first) = match probe {
        Some(probe) => match probe.product {
            probe::Product::ReusableFirstProduct => {
                (Some(probe.selected), Some(probe.input), Some(probe.output))
            }
            probe::Product::DiscoveryOnly => {
                drop(probe.input);
                drop(probe.output);
                (Some(probe.selected), None, None)
            }
        },
        None => (None, None, None),
    };
    let geometry = geometry::Geometry::checked(request.n, request.subspace, request.k)
        .ok_or_else(|| invalid("iterative geometry overflow"))?;
    let real_spectrum =
        request.symmetric && (request.b.is_none() || request.cholesky || request.definite);
    let kind =
        if real_spectrum && !matches!(request.sigma,super::Sigma::Near(value) if value.im!=0.0) {
            projected::Kind::SelfAdjoint
        } else {
            projected::Kind::General
        };
    let receipt = HostAllocationReceipt::admit(
        runtime,
        geometry
            .quote::<T>(requested, kind)
            .map_err(|e| admission(runtime, e))?,
    )
    .map_err(|e| admission(runtime, e))?;
    let mut work = work::Work::new(runtime, receipt.as_ref())?;
    work.check()?;
    if geometry.n == 0 {
        return publication::publish(
            &[],
            &[],
            &[],
            0,
            T::DTYPE,
            request.failure,
            requested,
            &mut work,
        )
        .await;
    }
    let mut native = storage::Native::<T>::new(geometry.native, runtime, receipt.as_ref()).await?;
    let mut complex = storage::Buffer::new(geometry.complex, runtime, receipt.as_ref()).await?;
    let mut indices = storage::Buffer::new(geometry.indices, runtime, receipt.as_ref()).await?;
    let mut projected =
        projected::Projected::new(geometry.p, kind, runtime, receipt.as_ref()).await?;
    let (q, tail) = native.get_mut().split_at_mut(geometry.n * geometry.p);
    let (products, temporary) = tail.split_at_mut(geometry.n * geometry.p);
    let (vectors, tail) = complex.get_mut().split_at_mut(geometry.n * geometry.k);
    let (values, residual) = tail.split_at_mut(geometry.k);
    let (order, tail) = indices.get_mut().split_at_mut(geometry.p);
    let (selected, tail) = tail.split_at_mut(geometry.k);
    let flags = tail;
    let mut operator =
        operator::Operator::<T>::prepare(a, &mut request, callable.as_ref(), first, &mut work)
            .await?;
    work.check()?;
    if let Some(input) = probe_input.as_ref() {
        for (index, destination) in q[..geometry.n].iter_mut().enumerate() {
            *destination = T::native(basis::element(input.value(), index)?)
                .ok_or_else(|| invalid("initial probe vector exceeds selected class"))?;
            work.tick().await?;
        }
    } else {
        basis::initial::<T>(&mut q[..geometry.n], request.start, &mut work).await?;
    }
    basis::range_seed(&mut q[..geometry.n], &mut operator, temporary, &mut work).await?;
    if operator.uses_metric()
        && !basis::metric::orthogonalize(&[], &mut q[..geometry.n], &operator, temporary, &mut work)
            .await?
    {
        return Err(invalid("StartVector has zero B-norm"));
    }
    let mut available = 1;
    let tolerance = request
        .tolerance
        .unwrap_or(if T::DTYPE == NumericDType::F32 {
            1e-6
        } else {
            1e-14
        });
    let mut all_converged = false;
    for iteration in 0..request.iterations {
        let active = basis::expand(
            q,
            products,
            temporary,
            geometry.n,
            available,
            &mut operator,
            &mut work,
        )
        .await?;
        if active < geometry.k {
            return Err(super::error_with_detail(
                "eigs: B-metric invariant subspace contains fewer finite eigenpairs than requested",
                &runmat_builtins::EIGS_ERROR_BREAKDOWN,
            ));
        }
        ritz::project(
            &q[..geometry.n * active],
            &products[..geometry.n * active],
            projected.matrix.get_mut(),
            geometry.n,
            &operator,
            temporary,
            &mut work,
        )
        .await?;
        projected = projected.compute(active, runtime, receipt.as_ref()).await?;
        work.check()?;
        for index in 0..active {
            projected.matrix.get_mut()[index] =
                ritz::transformed(projected.values.get()[index], sigma);
            if real_spectrum {
                // Exact source symmetry (or the declared callable contract)
                // proves a real spectrum, even for a complex inverse shift.
                projected.matrix.get_mut()[index].im = 0.0;
            }
            work.tick().await?;
        }
        selection::select(
            &projected.matrix.get()[..active],
            &mut order[..active],
            selected,
            sigma,
            &mut work,
        )
        .await?;
        ritz::extract(
            &q[..geometry.n * active],
            &products[..geometry.n * active],
            &projected.vectors.get()[..active * active],
            &projected.values.get()[..active],
            selected,
            vectors,
            flags,
            geometry.n,
            tolerance,
            &mut work,
        )
        .await?;
        for (destination, &index) in values.iter_mut().zip(selected.iter()) {
            *destination = projected.matrix.get()[index];
            work.tick().await?;
        }
        operator
            .normalize_vectors(vectors, temporary, residual, &mut work)
            .await?;
        operator
            .verify_matrix(
                values, vectors, flags, temporary, residual, tolerance, &mut work,
            )
            .await?;
        let mut count = 0usize;
        for &flag in flags.iter() {
            count += usize::from(flag != 0);
            work.tick().await?;
        }
        if request.display {
            work.flush().await?;
            crate::console::record_console_line(
                crate::console::ConsoleStream::Stdout,
                format!("eigs: restart {}; {count}/{} selected eigenpairs satisfy the residual criterion", iteration + 1, request.k),
            )?;
            work.check()?;
        }
        let converged = count == request.k;
        all_converged = converged;
        if converged || iteration + 1 == request.iterations {
            break;
        }
        available = basis::restart::<T>(
            q,
            vectors,
            geometry.n,
            geometry.k,
            &mut operator,
            temporary,
            &mut work,
        )
        .await?;
    }
    if requested != Some(3) && !all_converged {
        work.flush().await?;
        crate::warnings::emit(crate::warnings::WarningRequest {
            builtin: "eigs",
            identifier: "RunMat:eigs:NotConverged",
            message: "Not all requested eigenpairs converged; use the third output to inspect the convergence flag.",
            show_identifier: true,
        })?;
        work.check()?;
    }
    publication::publish(
        values,
        vectors,
        flags,
        geometry.n,
        T::DTYPE,
        request.failure,
        requested,
        &mut work,
    )
    .await
}
