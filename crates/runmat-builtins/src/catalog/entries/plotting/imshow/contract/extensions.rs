use crate::*;

pub const IMSHOW_FOUR_CHANNEL_IMAGE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "imshow-four-channel-image",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "imshow with numeric M-by-N-by-4 image data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ImshowFourChannelImageExtension"),
    };

pub const IMSHOW_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [IMSHOW_FOUR_CHANNEL_IMAGE_EXTENSION];
