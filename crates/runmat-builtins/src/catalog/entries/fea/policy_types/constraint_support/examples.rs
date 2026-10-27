//! Executable authoring examples: no solver, CAD import, or invented exemption.
use crate::*;

pub(super) const DIRECTION: &[BuiltinExample] = &[BuiltinExample {
    id: "normalize-direction",
    title: "Normalize a direction in the XY plane",
    program: "direction = fea.Direction.vector([3, 4, 0]);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(direction), 'fea.Direction')); assert(max(abs(direction.components - [0.6, 0.8, 0])) < 1e-12);",
    },
}];

pub(super) const MATCHING: &[BuiltinExample] = &[BuiltinExample {
    id: "matching-topology",
    title: "Require an already matching periodic topology",
    program: "pairing = fea.PeriodicPairing.matching();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(pairing), 'fea.MatchingPeriodicPairing'));",
    },
}];

pub(super) const GEOMETRIC: &[BuiltinExample] = &[BuiltinExample {
    id: "geometric-tolerance",
    title: "Set a one-micrometer geometric pairing tolerance",
    program: "tolerance = fea.Length.meters(1e-6); pairing = fea.PeriodicPairing.geometric(tolerance);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(pairing), 'fea.GeometricPeriodicPairing')); assert(pairing.tolerance.meters == 1e-6);",
    },
}];

pub(super) const IDENTITY: &[BuiltinExample] = &[BuiltinExample {
    id: "identity-transform",
    title: "Leave periodic coordinates unchanged",
    program: "transform = fea.PeriodicTransform.identity();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(transform), 'fea.IdentityPeriodicTransform'));",
    },
}];

pub(super) const TRANSLATION: &[BuiltinExample] = &[BuiltinExample {
    id: "translate-one-period",
    title: "Translate the secondary side by one axial period",
    program: "offset = fea.Translation.meters([0.01, 0, 0]); transform = fea.PeriodicTransform.translation(offset);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(transform), 'fea.TranslationPeriodicTransform')); assert(isequal(transform.translation.meters, [0.01, 0, 0]));",
    },
}];

pub(super) const ROTATION: &[BuiltinExample] = &[BuiltinExample {
    id: "rotate-quarter-turn",
    title: "Author a quarter-turn around the Z axis",
    program: "origin = fea.Position.meters([0, 0, 0]); axis = fea.Direction.vector([0, 0, 1]); angle = fea.Rotation.radians(pi / 2); transform = fea.PeriodicTransform.rotation(origin, axis, angle);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(transform), 'fea.RotationPeriodicTransform')); assert(isequal(transform.origin.meters, [0, 0, 0])); assert(isequal(transform.axis.components, [0, 0, 1])); assert(transform.angle.radians == pi / 2);",
    },
}];
