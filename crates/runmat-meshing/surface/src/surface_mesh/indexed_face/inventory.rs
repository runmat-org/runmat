//! Allocation-free inventory of the validated indexed wire layout. ExactVec owns
//! fixed collection capacities; external enum tags avoid a buffered value tree.
use super::decoding::{BoundaryRow, EdgeRow, NodeRow, TriangleRow, UseRow};
use super::{decoding::DecodedFace, value::DigestWords};
use crate::{
    ExactFaceMesh, ExactFaceMeshBoundarySegment, ExactFaceMeshEdgeParameter,
    ExactFaceMeshJoinedCut, ExactFaceMeshNode, ExactFaceMeshNodeUse, ExactFaceMeshTriangle,
};
use runmat_canonical_codec::{CanonicalCodecControl, CanonicalCodecError, CanonicalCodecErrorKind};
use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_curve::SharedCurveSegmentSplit;

pub(super) fn requested_bytes(
    value: &[u8],
    control: &dyn CanonicalCodecControl,
) -> Result<u64, CanonicalCodecError> {
    let mut inventory = Inventory {
        decoder: minicbor::Decoder::new(value),
        control,
        bytes: runmat_canonical_codec::CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT,
        expansion: 0,
    };
    inventory.fields(|p, key| match key {
        "format_version" | "schema_version" => p.skip(),
        "partition" => p.partition(),
        "outcome" => p.fields(|p, kind| match kind {
            "converged" => p.fields(|p, key| match key {
                "faces" => {
                    let count = p.vector::<DecodedFace>(Inventory::face)?;
                    p.expand_slots::<ExactFaceMesh>(count)
                }
                _ => Err(invalid("unknown converged field")),
            }),
            "requires_curve_splits" => p.fields(|p, key| match key {
                "splits" => p
                    .vector::<SharedCurveSegmentSplit>(Inventory::split)
                    .map(|_| ()),
                _ => Err(invalid("unknown split outcome field")),
            }),
            _ => Err(invalid("unknown indexed outcome")),
        }),
        _ => Err(invalid("unknown indexed envelope field")),
    })?;
    if inventory.decoder.position() != value.len() {
        return Err(invalid("unread indexed inventory"));
    }
    inventory
        .bytes
        .checked_add(inventory.expansion)
        .ok_or_else(overflow)
}
pub(super) fn requested_surface_bytes(
    value: &[u8],
    control: &dyn CanonicalCodecControl,
) -> Result<u64, CanonicalCodecError> {
    let mut inventory = Inventory {
        decoder: minicbor::Decoder::new(value),
        control,
        bytes: runmat_canonical_codec::CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT,
        expansion: 0,
    };
    inventory.fields(|profile, key| match key {
        "format_version" | "schema_version" => profile.skip(),
        "face_ids" => profile
            .vector::<PersistentEntityId>(Inventory::entity)
            .map(|_| ()),
        "shells" => profile
            .vector::<crate::ExactSurfaceShellEvidence>(|profile| {
                profile.fields(|profile, key| match key {
                    "source_shell_id" => profile.entity(),
                    "face_count"
                    | "shared_edge_count"
                    | "open_edge_count"
                    | "nonmanifold_edge_count"
                    | "is_sheet_shell"
                    | "is_watertight" => profile.skip(),
                    _ => Err(invalid("unknown surface shell evidence field")),
                })
            })
            .map(|_| ()),
        "rows" => profile.face(),
        _ => Err(invalid("unknown global surface field")),
    })?;
    if inventory.decoder.position() != value.len() {
        return Err(invalid("unread global surface inventory"));
    }
    inventory
        .bytes
        .checked_add(inventory.expansion)
        .ok_or_else(overflow)
}

const SHARED_IDENTITY_ALLOCATION_BYTES: u64 = (std::mem::size_of::<
    runmat_geometry_core::PersistentEntityIdData,
>() + 2 * std::mem::size_of::<usize>()) as u64;

struct Inventory<'a, 'c> {
    decoder: minicbor::Decoder<'a>,
    control: &'c dyn CanonicalCodecControl,
    bytes: u64,
    expansion: u64,
}
impl Inventory<'_, '_> {
    fn checkpoint(&self) -> Result<(), CanonicalCodecError> {
        if self.control.is_cancelled() {
            Err(invalid("surface typed inventory interrupted"))
        } else {
            Ok(())
        }
    }
    fn add(&mut self, bytes: u64) -> Result<(), CanonicalCodecError> {
        self.bytes = self.bytes.checked_add(bytes).ok_or_else(overflow)?;
        Ok(())
    }
    fn expand(&mut self, bytes: u64) -> Result<(), CanonicalCodecError> {
        self.expansion = self.expansion.checked_add(bytes).ok_or_else(overflow)?;
        Ok(())
    }
    fn expand_slots<T>(&mut self, count: u64) -> Result<(), CanonicalCodecError> {
        self.expand(
            count
                .checked_mul(std::mem::size_of::<T>() as u64)
                .ok_or_else(overflow)?,
        )
    }
    fn skip(&mut self) -> Result<(), CanonicalCodecError> {
        self.checkpoint()?;
        self.decoder.skip().map_err(invalid)
    }
    fn fields(
        &mut self,
        mut field: impl FnMut(&mut Self, &str) -> Result<(), CanonicalCodecError>,
    ) -> Result<(), CanonicalCodecError> {
        let count = self
            .decoder
            .map()
            .map_err(invalid)?
            .ok_or_else(|| invalid("definite map required"))?;
        for _ in 0..count {
            self.checkpoint()?;
            let key = self.decoder.str().map_err(invalid)?;
            field(self, key)?;
        }
        Ok(())
    }
    fn vector<T>(
        &mut self,
        mut item: impl FnMut(&mut Self) -> Result<(), CanonicalCodecError>,
    ) -> Result<u64, CanonicalCodecError> {
        let count = self
            .decoder
            .array()
            .map_err(invalid)?
            .ok_or_else(|| invalid("definite vector required"))?;
        self.add(
            count
                .checked_mul(std::mem::size_of::<T>() as u64)
                .ok_or_else(overflow)?,
        )?;
        for _ in 0..count {
            self.checkpoint()?;
            item(self)?;
        }
        Ok(count)
    }
    fn tuple(&mut self, count: u64) -> Result<(), CanonicalCodecError> {
        if self.decoder.array().map_err(invalid)? == Some(count) {
            Ok(())
        } else {
            Err(invalid("indexed row width differs"))
        }
    }
    /// One shared identity allocation per decoded entity; handles live in their slots.
    fn entity(&mut self) -> Result<(), CanonicalCodecError> {
        self.add(SHARED_IDENTITY_ALLOCATION_BYTES)?;
        self.fields(|p, key| match key {
            "kind" => p.skip(),
            "source_topology_id" => {
                let length = p.decoder.str().map_err(invalid)?.len();
                p.add(length as u64)
            }
            "assembly_path" => p
                .vector::<String>(|p| {
                    let length = p.decoder.str().map_err(invalid)?.len();
                    p.add(length as u64)
                })
                .map(|_| ()),
            _ => Err(invalid("unknown entity field")),
        })
    }
    fn partition(&mut self) -> Result<(), CanonicalCodecError> {
        self.fields(|p, key| match key {
            "kind" | "partition_index" | "partition_count" => p.skip(),
            "entity_range" => {
                if p.decoder.datatype().map_err(invalid)? == minicbor::data::Type::Null {
                    return p.skip();
                }
                p.fields(|p, key| match key {
                    "first" | "last" => p.entity(),
                    "entity_count" => p.skip(),
                    _ => Err(invalid("unknown entity range field")),
                })
            }
            _ => Err(invalid("unknown partition field")),
        })
    }
    fn split(&mut self) -> Result<(), CanonicalCodecError> {
        self.fields(|p, key| match key {
            "source_edge_id" => p.entity(),
            "endpoint_node_ids" | "edge_parameters" | "split_parameter" => p.skip(),
            _ => Err(invalid("unknown split field")),
        })
    }
    fn column<T>(&mut self) -> Result<(), CanonicalCodecError> {
        self.fields(|p, key| match key {
            "Constant" => p.skip(),
            "Dense" => p.vector::<T>(Self::skip).map(|_| ()),
            _ => Err(invalid("unknown quality column")),
        })
    }
    fn quality(&mut self) -> Result<(), CanonicalCodecError> {
        self.fields(|p, key| match key {
            "unit_normal" | "metric_edge_lengths" => p.column::<[f64; 3]>(),
            "acceptance_sample_count" => p.column::<u64>(),
            "physical_area_m2"
            | "minimum_metric_angle_rad"
            | "physical_aspect_ratio"
            | "chordal_deviation_m"
            | "normal_deviation_rad"
            | "accepted_chordal_deviation_m"
            | "accepted_normal_deviation_rad" => p.column::<f64>(),
            _ => Err(invalid("unknown quality field")),
        })
    }
    fn array_length(&mut self, expected: u64) -> Result<(), CanonicalCodecError> {
        self.checkpoint()?;
        if self.decoder.array().map_err(invalid)? != Some(expected) {
            return Err(invalid("indexed row has an invalid fixed arity"));
        }
        Ok(())
    }
    fn node_reference(&mut self) -> Result<(), CanonicalCodecError> {
        // A foreign digest remains available for independent rejection. It owns
        // one fixed allocation; accepted local references remain small indices.
        if self.decoder.datatype().map_err(invalid)? == minicbor::data::Type::Array {
            self.add(std::mem::size_of::<DigestWords>() as u64)?;
        }
        self.skip()
    }
    fn triangle_row(&mut self) -> Result<(), CanonicalCodecError> {
        self.array_length(4)?;
        for _ in 0..3 {
            self.skip()?;
        }
        self.array_length(3)?;
        for _ in 0..3 {
            self.node_reference()?;
        }
        Ok(())
    }
    fn boundary_row(&mut self) -> Result<(), CanonicalCodecError> {
        self.array_length(4)?;
        self.skip()?;
        self.skip()?;
        self.array_length(2)?;
        for _ in 0..2 {
            self.node_reference()?;
        }
        self.skip()
    }
    fn face(&mut self) -> Result<(), CanonicalCodecError> {
        self.fields(|p, key| match key {
            "schema_version"
            | "source_face"
            | "maximum_chordal_deviation_m"
            | "maximum_normal_deviation_rad" => p.skip(),
            "entities" => p.vector::<PersistentEntityId>(Self::entity).map(|_| ()),
            "charts" => p.vector::<DigestWords>(Self::skip).map(|_| ()),
            "quality" => p.quality(),
            "nodes" => {
                let nodes = p.vector::<NodeRow>(|p| {
                    p.tuple(3)?;
                    p.skip()?;
                    p.skip()?;
                    let uses = p.vector::<UseRow>(|p| {
                        p.tuple(5)?;
                        for _ in 0..4 {
                            p.skip()?;
                        }
                        let edges = p.vector::<EdgeRow>(Self::skip)?;
                        p.expand_slots::<ExactFaceMeshEdgeParameter>(edges)
                    })?;
                    p.expand_slots::<ExactFaceMeshNodeUse>(uses)
                })?;
                p.expand_slots::<ExactFaceMeshNode>(nodes)
            }
            "triangles" => {
                let count = p.vector::<TriangleRow>(Self::triangle_row)?;
                p.expand_slots::<ExactFaceMeshTriangle>(count)
            }
            "boundary_segments" => {
                let count = p.vector::<BoundaryRow>(Self::boundary_row)?;
                p.expand_slots::<ExactFaceMeshBoundarySegment>(count)
            }
            "joined_chart_cuts" => p.vector::<ExactFaceMeshJoinedCut>(Self::skip).map(|_| ()),
            _ => Err(invalid("unknown indexed face field")),
        })
    }
}
fn invalid(reason: impl std::fmt::Display) -> CanonicalCodecError {
    CanonicalCodecError {
        kind: CanonicalCodecErrorKind::Invalid,
        field: "surface typed inventory",
        reason: reason.to_string(),
    }
}
fn overflow() -> CanonicalCodecError {
    CanonicalCodecError {
        kind: CanonicalCodecErrorKind::ResourceLimit,
        field: "surface typed inventory",
        reason: "typed storage inventory overflow".into(),
    }
}
