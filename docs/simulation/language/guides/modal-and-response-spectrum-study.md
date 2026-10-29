# Build a modal response-spectrum study

Response-spectrum analysis is a two-analysis graph. The modal analysis publishes a basis; the
response analysis consumes that exact state and a model-owned spectrum.

Declare the spectrum in the model with its axis, ordinate quantity, damping, response convention,
direction, modal and directional combination, cumulative-mass requirement, truncation correction,
and rigid-response policy. None of those choices is inferred from a template.

Then declare state flow:

```fea
analysis modes {
    structural modal { modes = 30 }
    retain basis
}

analysis response_spectrum after analysis.modes {
    structural modal_response_spectrum {
        basis = analysis.modes.state.basis
        spectra = [model.response_spectrum.qualification_x]
    }
}
```

`after` provides ordering; `basis` supplies the typed state. Both are required because another
ordered predecessor might not publish a compatible basis.

Check and run the complete example:

```sh
runmat check docs/simulation/examples/modal-response-spectrum.fea
runmat run docs/simulation/examples/modal-response-spectrum.fea
```

Before using the envelope, inspect modal residuals, mass orthogonality, cumulative effective mass,
truncation and missing-mass treatment, directional combination, recovered-field provenance, and
formulation identity. A requested mode count by itself is not a completeness criterion.
