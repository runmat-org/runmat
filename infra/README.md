## RunMat Infra

This directory manages the telemetry ingestion stack (Cloudflare zone, Spectrum, Worker, DNS records) via Terraform.

### Remote state

Terraform state is stored in a Cloudflare R2 bucket using the S3 backend. The GitHub workflow passes the required settings via `terraform init -backend-config=...`. For local runs, supply the same values manually:

```bash
terraform init \
  -backend-config="bucket=$CF_TF_STATE_BUCKET" \
  -backend-config="key=infra/terraform.tfstate" \
  -backend-config="region=auto" \
  -backend-config="endpoints.s3=$CF_R2_ENDPOINT" \
  -backend-config="skip_credentials_validation=true" \
  -backend-config="skip_region_validation=true"
```

Set the following environment variables before running `terraform` locally:

- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`: R2 access key pair.
- `CLOUDFLARE_API_TOKEN`: token with permissions for the delegated zone, Spectrum, Workers, and R2 bucket.
- `TF_VAR_cloudflare_account_id`: Cloudflare account id that owns `runmat.org`.
- `TF_VAR_posthog_api_key`
- `TF_VAR_posthog_host`
- `TF_VAR_telemetry_ingestion_key`
- `TF_VAR_ga_measurement_id`
- `TF_VAR_ga_api_secret`

### GitHub Actions deployment

`.github/workflows/terraform.yml` runs automatically when files under `infra/` change:

- **Plan job** (pull requests from this repo) formats, validates, and uploads the plan artifact.
- **Apply job** (pushes to `main`) re-initializes the backend and applies changes.

Add the following secrets in the repository settings for the workflow:

| Secret | Purpose |
| --- | --- |
| `CF_TF_STATE_BUCKET` | R2 bucket that stores Terraform state |
| `CF_R2_ENDPOINT` | R2 S3-compatible endpoint (e.g. `https://<account>.r2.cloudflarestorage.com`) |
| `CF_R2_ACCESS_KEY_ID` / `CF_R2_SECRET_ACCESS_KEY` | Credentials for the bucket |
| `CLOUDFLARE_API_TOKEN` | Terraform provider authentication |
| `CLOUDFLARE_ACCOUNT_ID` | Passed through to `TF_VAR_cloudflare_account_id` |
| `POSTHOG_API_KEY` / `POSTHOG_HOST` | Worker bindings for PostHog |
| `TELEMETRY_INGESTION_KEY` | Optional shared secret (`x-telemetry-key`) |
| `GA_MEASUREMENT_ID` / `GA_API_SECRET` | Optional GA4 forwarding |

Whenever the workflow applies successfully, both HTTPS and UDP ingestion endpoints are up to date without manual steps.

