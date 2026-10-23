use crate::*;

pub const IMAGESC_FOUR_CHANNEL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "imagesc-four-channel-cdata",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "imagesc with M-by-N-by-4 CData is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ImagescFourChannelCDataExtension"),
};

pub const IMAGESC_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [IMAGESC_FOUR_CHANNEL_EXTENSION];
