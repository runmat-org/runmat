//! V4 owned wire layout. Count definite collection slots and owned strings;
//! tagged-enum Serde scratch is bounded for one row, never every numeric cell.
use crate::cdt::volume_mesh::representation::quality::QualityValues;
use crate::cdt::volume_mesh::representation::{FacetRow, NodeRow, SegmentRow};
use crate::cdt::{DelaunayFacetSteinerInsertion, DelaunayVolumeNode};
use runmat_canonical_codec::{
    CanonicalCodecControl, CanonicalCodecError, CanonicalCodecErrorKind, ExactVec,
};
use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::StableDigest;
use runmat_meshing_size::metric::{MetricSourceKind, MetricTensor3};

pub(super) fn requested_bytes(
    value: &[u8],
    control: &dyn CanonicalCodecControl,
) -> Result<u64, CanonicalCodecError> {
    let mut profile = Inventory {
        decoder: minicbor::Decoder::new(value),
        control,
        heap: 0,
        // One short enum name/key and bounded Serde visitor metadata.
        scratch: runmat_canonical_codec::CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT,
    };
    profile.fields(|profile, key| match key {
        "schema_version" => profile.skip(),
        "mesh" => profile.mesh(),
        _ => Err(invalid("unknown indexed envelope field")),
    })?;
    if profile.decoder.position() != value.len() {
        return Err(invalid("unread typed inventory"));
    }
    profile
        .heap
        .checked_add(profile.scratch)
        .ok_or_else(overflow)
}

struct Inventory<'a, 'c> {
    decoder: minicbor::Decoder<'a>,
    control: &'c dyn CanonicalCodecControl,
    heap: u64,
    scratch: u64,
}
impl Inventory<'_, '_> {
    fn checkpoint(&self) -> Result<(), CanonicalCodecError> {
        if self.control.is_cancelled() {
            Err(invalid("typed inventory interrupted"))
        } else {
            Ok(())
        }
    }
    fn add(&mut self, bytes: u64) -> Result<(), CanonicalCodecError> {
        self.heap = self.heap.checked_add(bytes).ok_or_else(overflow)?;
        Ok(())
    }
    fn slots<T>(&mut self, count: u64) -> Result<(), CanonicalCodecError> {
        self.add(
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
            .ok_or_else(|| invalid("definite typed map required"))?;
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
    ) -> Result<(), CanonicalCodecError> {
        let count = self
            .decoder
            .array()
            .map_err(invalid)?
            .ok_or_else(|| invalid("definite typed vector required"))?;
        self.slots::<T>(count)?;
        for _ in 0..count {
            self.checkpoint()?;
            item(self)?;
        }
        Ok(())
    }
    fn tuple(&mut self, count: u64) -> Result<(), CanonicalCodecError> {
        if self.decoder.array().map_err(invalid)? != Some(count) {
            return Err(invalid("invalid indexed row width"));
        }
        Ok(())
    }
    fn temporary_since(&mut self, start: usize) -> Result<(), CanonicalCodecError> {
        // The shared structural bound covers a single internally tagged Serde
        // value. Prefix and envelope framing are stack literals, not copied data.
        let raw = &self.decoder.input()[start..self.decoder.position()];
        let bytes = runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate(
            [b"typed-decode-scratch\0\x82\x60".as_slice(), raw], self.control)?;
        self.scratch = self.scratch.max(bytes);
        Ok(())
    }
    fn node(&mut self) -> Result<(), CanonicalCodecError> {
        self.fields(|profile, key| match key {
            "construction" => {
                let start = profile.decoder.position();
                profile.skip()?;
                profile.temporary_since(start)
            }
            "identity" | "coordinates_m" => profile.skip(),
            _ => Err(invalid("unknown volume node field")),
        })
    }
    fn string(&mut self) -> Result<(), CanonicalCodecError> {
        let length = self.decoder.str().map_err(invalid)?.len();
        self.add(length as u64)
    }
    /// One shared identity allocation per dictionary entity; handles live in their slots.
    fn entity(&mut self) -> Result<(), CanonicalCodecError> {
        self.add(
            (std::mem::size_of::<runmat_geometry_core::PersistentEntityIdData>()
                + 2 * std::mem::size_of::<usize>()) as u64,
        )?;
        self.fields(|profile, key| match key {
            "kind" => profile.skip(),
            "source_topology_id" => profile.string(),
            "assembly_path" => {
                let count = profile
                    .decoder
                    .array()
                    .map_err(invalid)?
                    .ok_or_else(|| invalid("definite assembly path required"))?;
                // Persistent entity admission already limits this collection to
                // 256; reject before Serde's cautious preallocation could grow.
                if count > 256 {
                    return Err(invalid("assembly path exceeds its entity limit"));
                }
                profile.slots::<String>(count)?;
                for _ in 0..count {
                    profile.checkpoint()?;
                    profile.string()?;
                }
                Ok(())
            }
            _ => Err(invalid("unknown persistent entity field")),
        })
    }
    fn volume(&mut self) -> Result<(), CanonicalCodecError> {
        self.fields(|profile, key| match key {
            "nodes" => profile.vector::<DelaunayVolumeNode>(Self::node),
            "entities" => profile.vector::<PersistentEntityId>(Self::entity),
            "charts" => profile.vector::<StableDigest>(Self::skip),
            "cells" => profile.vector::<([u32; 4], Option<u32>)>(Self::skip),
            "node_provenance" => profile.vector::<NodeRow>(|profile| {
                profile.tuple(2)?;
                profile.skip()?;
                profile.vector::<u32>(Self::skip)
            }),
            "segment_provenance" => profile.vector::<SegmentRow>(|profile| {
                profile.tuple(3)?;
                profile.skip()?;
                profile.vector::<u32>(Self::skip)?;
                profile.skip()
            }),
            "facet_provenance" => profile.vector::<FacetRow>(|profile| {
                profile.tuple(4)?;
                profile.skip()?;
                profile.skip()?;
                profile.vector::<u32>(Self::skip)?;
                profile.vector::<u32>(Self::skip)
            }),
            _ => Err(invalid("unknown indexed volume field")),
        })
    }
    fn quality(&mut self) -> Result<(), CanonicalCodecError> {
        self.fields(|profile, key| match key {
            "metrics" => profile.vector::<MetricTensor3>(Self::skip),
            "contexts" => {
                profile.vector::<ExactVec<u32>>(|profile| profile.vector::<u32>(Self::skip))
            }
            "active_sources" => profile.vector::<ExactVec<MetricSourceKind>>(|profile| {
                profile.vector::<MetricSourceKind>(Self::skip)
            }),
            "values" => profile.vector::<QualityValues>(Self::skip),
            "worsts" | "bounds" => profile.skip(),
            _ => Err(invalid("unknown indexed quality field")),
        })
    }
    fn mutation(&mut self) -> Result<(), CanonicalCodecError> {
        let start = self.decoder.position();
        self.fields(|profile, key| match key {
            "kind" => profile.skip(),
            "mutation" => profile.fields(|profile, key| match key {
                "removed_tetrahedron_node_identities" | "created_tetrahedron_node_identities" => {
                    profile.vector::<[StableDigest; 4]>(Self::skip)
                }
                // All other admitted mutation fields contain only inline values.
                "kind"
                | "source_node_identity"
                | "replacement_node"
                | "source_tetrahedron_node_identities"
                | "inserted_node"
                | "source_edge_node_identities"
                | "construction_point_index" => profile.skip(),
                _ => Err(invalid("unknown volume mutation field")),
            }),
            _ => Err(invalid("unknown tagged mutation field")),
        })?;
        self.temporary_since(start)
    }
    /// Canonical maps order "kind" before "mutation", so the step kind selects its profile.
    fn quality_mutation(&mut self) -> Result<(), CanonicalCodecError> {
        let mut refined = None;
        self.fields(|profile, key| match key {
            "kind" => {
                refined = Some(match profile.decoder.str().map_err(invalid)? {
                    "refined" => true,
                    "optimized" => false,
                    _ => return Err(invalid("unknown volume quality mutation kind")),
                });
                Ok(())
            }
            "mutation" => match refined {
                Some(true) => profile.node(),
                Some(false) => profile.mutation(),
                None => Err(invalid(
                    "volume quality mutation kind must precede its payload",
                )),
            },
            _ => Err(invalid("unknown volume quality mutation field")),
        })
    }
    fn mesh(&mut self) -> Result<(), CanonicalCodecError> {
        self.fields(|profile, key| match key {
            "volume" => profile.volume(),
            "quality" => profile.quality(),
            "facet_recovery_insertions" => {
                profile.vector::<DelaunayFacetSteinerInsertion>(Self::skip)
            }
            "quality_mutations" => {
                profile.vector::<crate::cdt::DelaunayVolumeQualityMutation>(Self::quality_mutation)
            }
            _ => Err(invalid("unknown indexed mesh field")),
        })
    }
}
fn invalid(reason: impl std::fmt::Display) -> CanonicalCodecError {
    CanonicalCodecError {
        kind: CanonicalCodecErrorKind::Invalid,
        field: "volume typed inventory",
        reason: reason.to_string(),
    }
}
fn overflow() -> CanonicalCodecError {
    CanonicalCodecError {
        kind: CanonicalCodecErrorKind::ResourceLimit,
        field: "volume typed inventory",
        reason: "typed storage inventory overflow".into(),
    }
}
