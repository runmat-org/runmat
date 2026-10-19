//! Checked mappings between semantic deep-learning axes and column-major data.
//!
//! A permutation is a view/coordinate transform. It does not mutate an input
//! tensor, change its labels, or imply an allocation. Consumers choose whether
//! a provider can use the view or whether a physical transpose is required.

use runmat_builtins::{
    DeepLearningAxisError, DeepLearningAxisLabel, DeepLearningAxisLayout,
    DeepLearningAxisPermutation, DeepLearningInputKind,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) enum AxisGeometryError {
    Labels(DeepLearningAxisError),
    ShapeOverflow,
    RankMismatch {
        expected: usize,
        actual: usize,
    },
    CoordinateOutOfBounds {
        axis: usize,
        index: usize,
        size: usize,
    },
    MissingAxis(DeepLearningAxisLabel),
    NoPoolingAxis,
}

impl From<DeepLearningAxisError> for AxisGeometryError {
    fn from(value: DeepLearningAxisError) -> Self {
        Self::Labels(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) struct AxisPlan {
    layout: DeepLearningAxisLayout,
    shape: Vec<usize>,
    column_major_strides: Vec<usize>,
    element_count: usize,
}

impl AxisPlan {
    pub(in crate::builtins::deep_learning) fn for_input(
        format: &str,
        kind: DeepLearningInputKind,
        shape: &[usize],
    ) -> Result<Self, AxisGeometryError> {
        let layout = DeepLearningAxisLayout::for_input(format, kind, shape.len())?;
        Self::new(layout, shape)
    }

    pub(in crate::builtins::deep_learning) fn new(
        layout: DeepLearningAxisLayout,
        shape: &[usize],
    ) -> Result<Self, AxisGeometryError> {
        if layout.axes().len() != shape.len() {
            return Err(AxisGeometryError::RankMismatch {
                expected: layout.axes().len(),
                actual: shape.len(),
            });
        }
        let mut stride = 1usize;
        let mut column_major_strides = Vec::with_capacity(shape.len());
        for size in shape {
            column_major_strides.push(stride);
            stride = stride
                .checked_mul(*size)
                .ok_or(AxisGeometryError::ShapeOverflow)?;
        }
        Ok(Self {
            layout,
            shape: shape.to_vec(),
            column_major_strides,
            element_count: stride,
        })
    }

    pub(in crate::builtins::deep_learning) fn layout(&self) -> &DeepLearningAxisLayout {
        &self.layout
    }

    pub(in crate::builtins::deep_learning) fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(in crate::builtins::deep_learning) fn element_count(&self) -> usize {
        self.element_count
    }

    pub(in crate::builtins::deep_learning) fn axis_size(
        &self,
        label: DeepLearningAxisLabel,
    ) -> Option<usize> {
        self.layout.index(label).map(|axis| self.shape[axis])
    }

    /// A channel transform retains the physical order of every axis.
    pub(in crate::builtins::deep_learning) fn with_channel_extent(
        &self,
        output_channels: usize,
    ) -> Result<Self, AxisGeometryError> {
        let channel_axis = self.layout.index(DeepLearningAxisLabel::Channel).ok_or(
            AxisGeometryError::MissingAxis(DeepLearningAxisLabel::Channel),
        )?;
        let mut shape = self.shape.clone();
        shape[channel_axis] = output_channels;
        Self::new(self.layout.clone(), &shape)
    }

    /// Temporal pooling removes T for CBT. The public CT and BT forms retain
    /// a singleton U axis. Spatial pooling keeps S singleton, including SCBT,
    /// where T is not the reduced axis.
    pub(in crate::builtins::deep_learning) fn after_global_average_pooling_1d(
        &self,
    ) -> Result<Self, AxisGeometryError> {
        if let Some(spatial_axis) = self.layout.index(DeepLearningAxisLabel::Spatial) {
            let mut shape = self.shape.clone();
            shape[spatial_axis] = 1;
            return Self::new(self.layout.clone(), &shape);
        }
        let time_axis = self
            .layout
            .index(DeepLearningAxisLabel::Time)
            .ok_or(AxisGeometryError::NoPoolingAxis)?;
        let mut shape = self.shape.clone();
        let mut labels = self.layout.axes().to_vec();
        if self.layout.index(DeepLearningAxisLabel::Batch).is_some()
            && self.layout.index(DeepLearningAxisLabel::Channel).is_some()
        {
            shape.remove(time_axis);
            labels.remove(time_axis);
        } else {
            shape[time_axis] = 1;
            labels[time_axis] = DeepLearningAxisLabel::Unspecified;
        }
        let layout = DeepLearningAxisLayout::from_axes(&labels)?;
        Self::new(layout, &shape)
    }

    /// Recurrent output replaces flattened source features with the hidden
    /// channel width. Last-output mode removes T; an unbatched last result
    /// retains the documented singleton unspecified axis.
    pub(in crate::builtins::deep_learning) fn after_recurrent(
        &self,
        output_channels: usize,
        sequence: bool,
    ) -> Result<Self, AxisGeometryError> {
        let batch = self.axis_size(DeepLearningAxisLabel::Batch);
        let time = self.axis_size(DeepLearningAxisLabel::Time);
        use DeepLearningAxisLabel::{Batch, Channel, Time, Unspecified};
        let (shape, labels) = match (batch, time, sequence) {
            (Some(batch), Some(time), true) => (
                vec![output_channels, batch, time],
                vec![Channel, Batch, Time],
            ),
            (Some(batch), _, _) => (vec![output_channels, batch], vec![Channel, Batch]),
            (None, Some(time), true) => (vec![output_channels, time], vec![Channel, Time]),
            (None, _, _) => (vec![output_channels, 1], vec![Channel, Unspecified]),
        };
        Self::new(DeepLearningAxisLayout::from_axes(&labels)?, &shape)
    }

    pub(in crate::builtins::deep_learning) fn offset(
        &self,
        indices: &[usize],
    ) -> Result<usize, AxisGeometryError> {
        if indices.len() != self.shape.len() {
            return Err(AxisGeometryError::RankMismatch {
                expected: self.shape.len(),
                actual: indices.len(),
            });
        }
        let mut offset = 0usize;
        for (axis, (&index, &size)) in indices.iter().zip(&self.shape).enumerate() {
            if index >= size {
                return Err(AxisGeometryError::CoordinateOutOfBounds { axis, index, size });
            }
            offset += index * self.column_major_strides[axis];
        }
        Ok(offset)
    }

    pub(in crate::builtins::deep_learning) fn lower_to(
        &self,
        target: &DeepLearningAxisLayout,
    ) -> Result<AxisLowering, AxisGeometryError> {
        let permutation = self.layout.permutation_to(target)?;
        let shape = permutation
            .target_to_source()
            .iter()
            .map(|&source_axis| self.shape[source_axis])
            .collect();
        Ok(AxisLowering {
            source: self.clone(),
            target: target.clone(),
            shape,
            permutation,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) struct AxisLowering {
    source: AxisPlan,
    target: DeepLearningAxisLayout,
    shape: Vec<usize>,
    permutation: DeepLearningAxisPermutation,
}

impl AxisLowering {
    pub(in crate::builtins::deep_learning) fn target_to_source(&self) -> &[usize] {
        self.permutation.target_to_source()
    }

    pub(in crate::builtins::deep_learning) fn target_layout(&self) -> &DeepLearningAxisLayout {
        &self.target
    }

    pub(in crate::builtins::deep_learning) fn target_shape(&self) -> &[usize] {
        &self.shape
    }

    pub(in crate::builtins::deep_learning) fn source_offset(
        &self,
        target_indices: &[usize],
    ) -> Result<usize, AxisGeometryError> {
        if target_indices.len() != self.shape.len() {
            return Err(AxisGeometryError::RankMismatch {
                expected: self.shape.len(),
                actual: target_indices.len(),
            });
        }
        let mut source_indices = vec![0; target_indices.len()];
        for (target_axis, &source_axis) in self.permutation.target_to_source().iter().enumerate() {
            source_indices[source_axis] = target_indices[target_axis];
        }
        self.source.offset(&source_indices)
    }

    /// Maps a column-major target linear index to the input's physical index.
    /// Useful for a materialization when a provider cannot consume a view.
    pub(in crate::builtins::deep_learning) fn source_offset_linear(
        &self,
        mut target_linear: usize,
    ) -> Result<usize, AxisGeometryError> {
        if target_linear >= self.source.element_count {
            return Err(AxisGeometryError::CoordinateOutOfBounds {
                axis: 0,
                index: target_linear,
                size: self.source.element_count,
            });
        }
        let mut target_indices = Vec::with_capacity(self.shape.len());
        for &size in &self.shape {
            // A valid target linear offset cannot encounter a zero extent.
            target_indices.push(target_linear % size);
            target_linear /= size;
        }
        self.source_offset(&target_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(format: &str) -> DeepLearningAxisLayout {
        DeepLearningAxisLayout::parse(format, format.len()).unwrap()
    }

    #[test]
    fn nonsquare_feature_bc_and_cb_have_independent_numerical_oracles() {
        // Three observations, two channels; each physical tensor spells out
        // the same logical values but has a different column-major order.
        let bc = AxisPlan::for_input("", DeepLearningInputKind::Feature, &[3, 2]).unwrap();
        let cb = AxisPlan::for_input("CB", DeepLearningInputKind::Feature, &[2, 3]).unwrap();
        let bc_values = [1., 2., 3., 10., 20., 30.];
        let cb_values = [1., 10., 2., 20., 3., 30.];
        let bc_view = bc.lower_to(&layout("BC")).unwrap();
        let cb_view = cb.lower_to(&layout("BC")).unwrap();
        assert_eq!(bc_view.target_shape(), &[3, 2]);
        assert_eq!(cb_view.target_shape(), &[3, 2]);
        let expected = [21., 42., 63.]; // FC weights [1,2], zero bias.
        for batch in 0..3 {
            let bc_result = bc_values[bc_view.source_offset(&[batch, 0]).unwrap()]
                + 2. * bc_values[bc_view.source_offset(&[batch, 1]).unwrap()];
            let cb_result = cb_values[cb_view.source_offset(&[batch, 0]).unwrap()]
                + 2. * cb_values[cb_view.source_offset(&[batch, 1]).unwrap()];
            assert_eq!(bc_result, expected[batch]);
            assert_eq!(cb_result, expected[batch]);
        }
        assert_eq!(cb_view.permutation.inverse().target_to_source(), &[1, 0]);
        assert_eq!(cb_view.source_offset_linear(3).unwrap(), 1);
    }

    #[test]
    fn cbt_time_and_batch_indices_do_not_collapse() {
        let cbt =
            AxisPlan::for_input("CBT", DeepLearningInputKind::VectorSequence, &[2, 3, 4]).unwrap();
        let btc = cbt.lower_to(&layout("BTC")).unwrap();
        assert_eq!(btc.target_shape(), &[3, 4, 2]);
        assert_eq!(btc.source_offset(&[2, 3, 1]).unwrap(), 23);
        assert_eq!(btc.permutation.inverse().target_to_source(), &[2, 0, 1]);
    }

    #[test]
    fn spatial_and_unknown_axes_retain_occurrence_identity() {
        let plan = AxisPlan::new(layout("SSCBUU"), &[2, 3, 4, 5, 6, 7]).unwrap();
        let lowered = plan.lower_to(&layout("UCSUBS")).unwrap();
        assert_eq!(lowered.target_shape(), &[6, 4, 2, 7, 5, 3]);
        assert_eq!(
            lowered.source_offset(&[1, 2, 1, 3, 4, 2]).unwrap(),
            plan.offset(&[1, 2, 2, 4, 1, 3]).unwrap()
        );
    }

    #[test]
    fn channel_and_pool_transitions_preserve_or_update_labels() {
        let bc = AxisPlan::for_input("", DeepLearningInputKind::Feature, &[3, 2]).unwrap();
        assert_eq!(bc.with_channel_extent(5).unwrap().shape(), &[3, 5]);
        let cb = AxisPlan::for_input("CB", DeepLearningInputKind::Feature, &[2, 3]).unwrap();
        assert_eq!(cb.with_channel_extent(5).unwrap().shape(), &[5, 3]);

        let cbt =
            AxisPlan::for_input("CBT", DeepLearningInputKind::VectorSequence, &[2, 3, 4]).unwrap();
        let pooled = cbt.after_global_average_pooling_1d().unwrap();
        assert_eq!(pooled.layout().labels(), "CB");
        assert_eq!(pooled.shape(), &[2, 3]);

        let ct = AxisPlan::for_input("CT", DeepLearningInputKind::VectorSequence, &[2, 4]).unwrap();
        let pooled = ct.after_global_average_pooling_1d().unwrap();
        assert_eq!(pooled.layout().labels(), "CU");
        assert_eq!(pooled.shape(), &[2, 1]);

        let bt = AxisPlan::new(layout("BT"), &[2, 4]).unwrap();
        let pooled = bt.after_global_average_pooling_1d().unwrap();
        assert_eq!(pooled.layout().labels(), "BU");
        assert_eq!(pooled.shape(), &[2, 1]);

        let sbt = AxisPlan::new(layout("SBT"), &[4, 2, 3]).unwrap();
        let pooled = sbt.after_global_average_pooling_1d().unwrap();
        assert_eq!(pooled.layout().labels(), "SBT");
        assert_eq!(pooled.shape(), &[1, 2, 3]);

        let scb = AxisPlan::for_input("SCB", DeepLearningInputKind::Spatial1d, &[4, 2, 3]).unwrap();
        let pooled = scb.after_global_average_pooling_1d().unwrap();
        assert_eq!(pooled.layout().labels(), "SCB");
        assert_eq!(pooled.shape(), &[1, 2, 3]);

        let scbt = AxisPlan::for_input(
            "SCBT",
            DeepLearningInputKind::SpatialSequence1d,
            &[4, 2, 3, 5],
        )
        .unwrap();
        let pooled = scbt.after_global_average_pooling_1d().unwrap();
        assert_eq!(pooled.layout().labels(), "SCBT");
        assert_eq!(pooled.shape(), &[1, 2, 3, 5]);
    }

    #[test]
    fn overflow_and_invalid_coordinates_reject_without_allocation() {
        assert_eq!(
            AxisPlan::new(layout("CB"), &[usize::MAX, 2]),
            Err(AxisGeometryError::ShapeOverflow)
        );
        let plan = AxisPlan::new(layout("CB"), &[2, 3]).unwrap();
        assert!(matches!(
            plan.offset(&[2, 0]),
            Err(AxisGeometryError::CoordinateOutOfBounds { .. })
        ));
        assert!(matches!(
            plan.lower_to(&layout("BCU")),
            Err(AxisGeometryError::Labels(
                DeepLearningAxisError::IncompatiblePermutation
            ))
        ));
    }
}
