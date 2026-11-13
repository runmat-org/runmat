use crate::*;

pub const IMAGE_FOUR_CHANNEL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "image-four-channel-cdata",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "image with M-by-N-by-4 CData is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ImageFourChannelCDataExtension"),
};

pub const IMAGE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [IMAGE_FOUR_CHANNEL_EXTENSION];
