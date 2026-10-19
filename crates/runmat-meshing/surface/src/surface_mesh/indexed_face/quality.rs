//! One declaration owns the lossless quality columns and their reconstruction.
use super::*;
use serde::{Deserialize, Serialize};
pub(super) struct TriangleIdentity {
    pub triangle_id: StableDigest,
    pub chart_id: StableDigest,
    pub source_face_id: PersistentEntityId,
    pub node_ids: [StableDigest; 3],
}
macro_rules! columns {
    ($($field:ident:$kind:ty),* $(,)?)=>{
        #[derive(Serialize)]
        pub(super) struct QualityEncoding<'a>{ $( $field:Column<'a,$kind>, )* }
        impl<'a> QualityEncoding<'a> {
            pub fn new(rows:&'a [crate::ExactFaceMeshTriangle],control:&dyn MeshingCancellationSignal)->Result<Self,ExactSurfaceMeshError>{
                Ok(Self {$( $field:Column::new(rows,|row|row.$field,control)?, )*})
            }
        }
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        pub(super) struct QualityDecoded {$( $field:DecodedColumn<$kind>, )*}
        impl QualityDecoded {
            pub fn validate(&self,count:usize)->Result<(),ExactSurfaceMeshError>{ $(self.$field.validate_length(count)?;)* Ok(()) }
            pub fn triangle(&self,index:usize,identity:TriangleIdentity)->crate::ExactFaceMeshTriangle {
                crate::ExactFaceMeshTriangle {
                    triangle_id:identity.triangle_id,chart_id:identity.chart_id,source_face_id:identity.source_face_id,node_ids:identity.node_ids,
                    $($field:self.$field.at(index),)*
                }
            }
        }
    }
}
columns! {
    unit_normal:[f64;3], physical_area_m2:f64, metric_edge_lengths:[f64;3],
    minimum_metric_angle_rad:f64,physical_aspect_ratio:f64,chordal_deviation_m:f64,
    normal_deviation_rad:f64,acceptance_sample_count:u64,
    accepted_chordal_deviation_m:f64,accepted_normal_deviation_rad:f64,
}
