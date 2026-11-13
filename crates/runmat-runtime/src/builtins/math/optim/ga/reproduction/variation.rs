//! Variation proposes coordinates; only original-domain admission can install
//! them. Numerical overflow never becomes a clipped or falsely feasible gene.
use super::super::super::intlinprog::feasibility::Repair;
use super::super::{
    invalid, linear, operators::crossover, population::Generation, random, ScalarCooperation,
};
use super::{
    destination::Destination,
    mutation::{Mutation, Proposal},
    Current,
};
use crate::BuiltinResult;
use runmat_builtins::OptimGeneticPopulation;

#[derive(Default)]
pub(super) struct Status {
    pub retained: bool,
    pub timed_out: bool,
    pub unverified: bool,
}
pub(super) struct Request<'a, 'b> {
    pub crossover: bool,
    pub first: usize,
    pub second: Option<usize>,
    pub destination: usize,
    pub current: &'a Current<'b>,
}
pub(super) async fn builtin<G: linear::Geometry>(
    request: Request<'_, '_>,
    next: &mut Destination,
    mut domain: Option<&mut G>,
    mutation: &mut Mutation,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Status> {
    let current = request.current;
    let population = current
        .generation
        .numeric()
        .ok_or_else(|| invalid("built-in variation requires a numeric population"))?;
    let parent =
        super::super::population::coordinates(population, request.first, cooperation).await?;
    let proposal = if request.crossover {
        let second = request
            .second
            .ok_or_else(|| invalid("crossover needs two actual parents"))?;
        let second_point =
            super::super::population::coordinates(population, second, cooperation).await?;
        let parameters = crossover::Parameters {
            kind: current
                .options
                .crossover
                .kind()
                .ok_or_else(|| invalid("custom crossover uses semantic dispatch"))?,
            ratio: &current.options.crossover_ratio,
            first_is_better: !current.comparison[request.first]
                .compare(current.comparison[second])
                .is_gt(),
        };
        let draws = random::uniform(
            crossover::draw_count(&parameters, parent.len())?,
            cooperation,
        )
        .await?;
        match crossover::child(parameters, &parent, &second_point, &draws, cooperation).await? {
            Some(point) => Proposal::Point(point),
            None => Proposal::Retain,
        }
    } else {
        mutation
            .propose(
                &parent,
                domain.as_deref_mut(),
                current.options,
                current.timing,
                cooperation,
            )
            .await?
    };
    install(
        next,
        request.destination,
        proposal,
        &parent,
        domain,
        current,
        cooperation,
    )
    .await
}

async fn install<G: linear::Geometry>(
    next: &mut Destination,
    row: usize,
    proposal: Proposal,
    parent: &[f64],
    mut domain: Option<&mut G>,
    current: &Current<'_>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Status> {
    let proposal = match proposal {
        Proposal::Point(point) => {
            if let Some(domain) = domain.as_mut() {
                match domain.repair(&point, cooperation).await? {
                    Repair::Unchanged => Proposal::Point(point),
                    Repair::Point(point) => Proposal::Point(point),
                    Repair::TimeLimit => Proposal::TimeLimit,
                    Repair::Infeasible | Repair::Unverified => Proposal::Unverified,
                }
            } else {
                Proposal::Point(point)
            }
        }
        other => other,
    };
    let mut status = Status::default();
    let point = match &proposal {
        Proposal::Point(point) => point.as_slice(),
        Proposal::TimeLimit => {
            status.timed_out = true;
            return Ok(status);
        }
        Proposal::Retain | Proposal::Unverified => {
            status.retained = true;
            status.unverified = matches!(proposal, Proposal::Unverified);
            parent
        }
    };
    if point.len() != parent.len() {
        return Err(invalid("variation changed genome length"));
    }
    let mut identical = true;
    for (&value, &old) in point.iter().zip(parent) {
        identical &= value.to_bits() == old.to_bits();
        cooperation.completed(1).await?;
    }
    status.retained |= identical;
    next.point(
        row,
        point,
        current.options.population == OptimGeneticPopulation::BitString,
        cooperation,
    )
    .await?;
    Ok(status)
}

pub(super) struct Custom<'a, 'b> {
    pub rows: &'a [usize],
    pub parents: &'a [usize],
    pub crossover: bool,
    pub children: Generation,
    pub current: &'a Current<'b>,
}
pub(super) async fn install_custom<G: linear::Geometry, I: super::super::callbacks::Invocation>(
    request: Custom<'_, '_>,
    next: &mut Destination,
    mut domain: Option<&mut G>,
    copied_from: &mut [Option<usize>],
    invocation: &mut I,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<std::ops::ControlFlow<I::Stop, Status>> {
    if matches!(request.children, Generation::Custom { .. }) {
        if let std::ops::ControlFlow::Break(stop) = next
            .children(request.rows, request.children, invocation, cooperation)
            .await?
        {
            return Ok(std::ops::ControlFlow::Break(stop));
        }
        return Ok(std::ops::ControlFlow::Continue(Status::default()));
    }
    let children = request
        .children
        .numeric()
        .expect("numeric custom offspring");
    let current = request
        .current
        .generation
        .numeric()
        .ok_or_else(|| invalid("numeric custom offspring changed population representation"))?;
    let mut combined = Status::default();
    for (child, &destination) in request.rows.iter().enumerate() {
        if request
            .current
            .options
            .maximum
            .reached(request.current.timing.elapsed())
        {
            combined.timed_out = true;
            break;
        }
        let index = request.parents[if request.crossover { child * 2 } else { child }];
        let parent = super::super::population::coordinates(current, index, cooperation).await?;
        let proposal = super::super::population::coordinates(children, child, cooperation).await?;
        let status = install(
            next,
            destination,
            Proposal::Point(proposal),
            &parent,
            domain.as_deref_mut(),
            request.current,
            cooperation,
        )
        .await?;
        if status.retained {
            copied_from[destination] = Some(index);
        }
        combined.unverified |= status.unverified;
        combined.timed_out |= status.timed_out;
        if combined.timed_out {
            break;
        }
    }
    Ok(std::ops::ControlFlow::Continue(combined))
}
