//! Converted sketch entities in an assembly (Convert entities, Offset of a component edge): an
//! entity written `from = published(occurrence = upper, name = upper_e1)` takes its geometry from
//! the edges component `upper` publishes as `upper_e1`, carried into assembly coordinates by the
//! occurrence's placement and projected onto the sketch's plane, on every regeneration. The
//! solver holds it fixed, so the rest of the sketch is dimensioned and related to it, and it
//! follows the component when the component moves or its part changes.
//!
//! The projected polyline is read as the simplest curve through it ([`fit`]).

use crate::assembly::model::{PathSegment, RelPath};
use crate::assembly::resolve::Resolver;
use crate::assembly::source::PublishedGeometry;
use crate::ids::SketchEntityId;
use crate::model::datum::Frame;
use crate::model::sketch::{
    Sketch, SketchConstraint, SketchConstraintKind, SketchProjection, SketchTarget,
};
use crate::regen::kernel::Transform;
use crate::sketch_conversion::{fit, in_plane};

impl<'a, 'b> Resolver<'a, 'b> {
    /// `sketch` with each converted entity re-read from what it was converted from and fixed,
    /// in the sketch's `frame` (assembly coordinates at node `owner`).
    pub(in crate::assembly) fn with_projections(
        &mut self,
        owner: u32,
        frame: &Frame,
        sketch: &Sketch,
    ) -> Result<Sketch, String> {
        let mut out = sketch.clone();
        for entity in out.entities.iter_mut() {
            let (occurrence, published) = match entity.projected.clone() {
                None => continue,
                Some(SketchProjection::Published {
                    occurrence,
                    published,
                }) => (occurrence, published),
                Some(_) => {
                    return Err(format!(
                        "`{}` is converted from a part's own edges or faces; an assembly sketch \
                         converts what a component publishes: `from = published(occurrence = \
                         <component>, name = <published edges>)`",
                        entity.id
                    ))
                }
            };
            let chains = self.projection_chains(owner, &occurrence, &published)?;
            let [chain] = chains.as_slice() else {
                return Err(format!(
                    "`{}`'s `{}` is {} separate edge chains; convert one connected chain per entity",
                    occurrence,
                    published,
                    chains.len()
                ));
            };
            let local: Vec<[f64; 2]> = chain.iter().map(|p| in_plane(frame, *p)).collect();
            entity.geometry = fit(&local).ok_or_else(|| {
                format!(
                    "`{}`'s `{}` projects to a point on this sketch's plane",
                    occurrence, published
                )
            })?;
        }
        for entity in &sketch.entities {
            if entity.projected.is_some() {
                out.constraints.push(SketchConstraint {
                    id: None,
                    kind: SketchConstraintKind::Fix {
                        target: SketchTarget::Entity {
                            entity: entity.id.clone(),
                        },
                    },
                });
            }
        }
        Ok(out)
    }

    /// The edge chains component `occurrence` (a dotted path below `owner`) publishes as
    /// `name`, in `owner`'s coordinates (metres).
    fn projection_chains(
        &mut self,
        owner: u32,
        occurrence: &str,
        name: &str,
    ) -> Result<Vec<Vec<[f64; 3]>>, String> {
        let path = RelPath(
            occurrence
                .split('.')
                .map(|n| PathSegment {
                    name: n.to_owned(),
                    instance: None,
                })
                .collect(),
        );
        let host = self
            .walk(owner, &path)
            .map_err(|_| format!("the assembly has no component `{occurrence}`"))?;
        let node = self.nodes[host as usize].clone();
        let summary = self
            .configured_summary(
                &node.definition,
                node.revision.as_ref(),
                &node.configuration,
            )
            .map_err(|e| e.to_string())?;
        let chains = match summary.publishes.get(&crate::ids::PublishName::new(name)) {
            Some(PublishedGeometry::Path { chains }) => chains.clone(),
            Some(_) => {
                return Err(format!(
                    "`{occurrence}`'s `{name}` isn't an edge set to convert"
                ))
            }
            None => {
                return Err(format!(
                    "`{occurrence}` ({}) publishes no `{name}`",
                    node.definition
                ))
            }
        };
        let to_owner: Transform = self.chain(owner, host);
        Ok(chains
            .into_iter()
            .map(|c| c.points.into_iter().map(|p| to_owner.apply(p)).collect())
            .collect())
    }
}

/// The entity IDs of a sketch that are converted geometry.
pub fn projected_ids(sketch: &Sketch) -> Vec<SketchEntityId> {
    sketch
        .entities
        .iter()
        .filter(|e| e.projected.is_some())
        .map(|e| e.id.clone())
        .collect()
}
