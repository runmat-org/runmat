#!/usr/bin/env bash
# The commands that created RunMat's GCP runners (project runmat, 2026-11-20).
# Kept as the record and for rebuilding; each step fails if the resource exists.
# Run from infra/ci-runners with gcloud and gh logged in as a project owner and
# repository admin.
set -euo pipefail
P=runmat Z=us-central1-b R=us-central1
here="$(cd "$(dirname "$0")/.." && pwd)"; cd "$here"

# Identities: the VMs may only write logs and metrics. The autostart service may
# only get/start/resume the two VMs and read the webhook secret.
gcloud iam service-accounts create ci-runner-vm --project $P --display-name "Design CI: GitHub runner VMs (logs and metrics only)"
gcloud iam service-accounts create ci-runner-autostart --project $P --display-name "Design CI: starts runner VMs on queued jobs"
gcloud iam service-accounts create ci-runner-build --project $P --display-name "Design CI: builds the autostart service image"
for role in roles/logging.logWriter roles/monitoring.metricWriter; do
  gcloud projects add-iam-policy-binding $P --member serviceAccount:ci-runner-vm@$P.iam.gserviceaccount.com --role $role --condition=None
done
for role in roles/run.builder roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding $P --member serviceAccount:ci-runner-build@$P.iam.gserviceaccount.com --role $role --condition=None
done
gcloud iam roles create ciRunnerStarter --project $P --title "Design CI runner starter" \
  --permissions compute.instances.get,compute.instances.start,compute.instances.resume --stage GA

# Network: IAP SSH/RDP only.
gcloud compute firewall-rules create ci-runner-allow-iap --project $P --network default --direction INGRESS --priority 900 \
  --source-ranges 35.235.240.0/20 --allow tcp:22,tcp:3389 --target-tags ci-runner
gcloud compute firewall-rules create ci-runner-deny-ingress --project $P --network default --direction INGRESS --priority 1000 \
  --source-ranges 0.0.0.0/0 --action DENY --rules all --target-tags ci-runner

# Disks and VMs. Slots: "<slot>=<owner/repo>" pairs; each is one runner registration.
create_vm() { # name machine image-family image-project boot-gb work-gb cache-gb label slots startup-key startup-file
  gcloud compute disks create "$1-work" --project $P --zone $Z --type pd-ssd --size "$6GB" --labels purpose=ci-runner,owner=design-ci
  gcloud compute disks create "$1-cache" --project $P --zone $Z --type pd-ssd --size "$7GB" --labels purpose=ci-runner,owner=design-ci
  gcloud compute instances create "$1" --project $P --zone $Z --machine-type "$2" \
    --image-family "$3" --image-project "$4" --boot-disk-size "$5GB" --boot-disk-type pd-balanced \
    --disk "name=$1-work,device-name=ci-work,auto-delete=no" --disk "name=$1-cache,device-name=ci-cache,auto-delete=no" \
    --service-account ci-runner-vm@$P.iam.gserviceaccount.com --scopes cloud-platform --tags ci-runner \
    --labels purpose=ci-runner,owner=design-ci --shielded-secure-boot \
    --metadata "ci-label=$8,ci-runner-version=2.338.2,ci-idle-minutes=30,ci-slots=$9" \
    --metadata-from-file "${10}=${11}"
  gcloud compute instances add-iam-policy-binding "$1" --project $P --zone $Z \
    --member serviceAccount:ci-runner-autostart@$P.iam.gserviceaccount.com --role projects/$P/roles/ciRunnerStarter
}
pub() { for i in $(seq 1 "$1"); do printf 'pub%s=runmat-org/runmat ' "$i"; done; }
prv() { for i in $(seq 1 "$1"); do printf 'prv%s=runmat-org/runmat-private ' "$i"; done; }
create_vm ci-linux-x64 n2-highmem-32 ubuntu-2404-lts-amd64 ubuntu-os-cloud 100 500 1500 runmat-gcp-linux-x64 \
  "$(pub 5)$(prv 2)" startup-script linux/startup.sh
create_vm ci-windows-x64 n2-highmem-64 windows-2022 windows-cloud 200 600 2000 runmat-gcp-windows-x64 \
  "$(pub 8)$(prv 1)" windows-startup-script-ps1 windows/startup.ps1
gcp/push-bundle.sh linux --register
gcp/push-bundle.sh windows --register   # remove ci-registration metadata once the slots are online

# Autostart: webhook secret, Cloud Run service, repository webhooks.
python3 -c "import secrets; print(secrets.token_hex(32), end='')" |
  gcloud secrets create ci-runner-webhook-secret --project $P --replication-policy automatic --data-file=-
gcloud secrets add-iam-policy-binding ci-runner-webhook-secret --project $P \
  --member serviceAccount:ci-runner-autostart@$P.iam.gserviceaccount.com --role roles/secretmanager.secretAccessor
# One instance stays up: a watcher thread retries a failed start (quota, VM still
# stopping) for hours, and a scaled-in instance would end it.
gcloud run deploy ci-runner-autostart --project $P --region $R --source autostart \
  --build-service-account projects/$P/serviceAccounts/ci-runner-build@$P.iam.gserviceaccount.com \
  --service-account ci-runner-autostart@$P.iam.gserviceaccount.com --allow-unauthenticated \
  --no-cpu-throttling --min-instances 1 --max-instances 2 --memory 512Mi --cpu 1 --timeout 900 \
  --set-secrets WEBHOOK_SECRET=ci-runner-webhook-secret:latest --labels purpose=ci-runner,owner=design-ci
url="$(gcloud run services describe ci-runner-autostart --project $P --region $R --format 'value(status.url)')/github"
secret="$(gcloud secrets versions access latest --secret ci-runner-webhook-secret --project $P)"
for repo in runmat-org/runmat runmat-org/runmat-private; do
  python3 -c 'import json, sys; print(json.dumps({"name": "web", "active": True, "events": ["workflow_job"],
    "config": {"url": sys.argv[1], "content_type": "json", "secret": sys.argv[2], "insecure_ssl": "0"}}))' "$url" "$secret" |
    gh api -X POST "repos/$repo/hooks" --input - >/dev/null
done

# Disk alert: email ALERT_EMAIL (the owner) when a runner disk passes 85%.
: "${ALERT_EMAIL:?set ALERT_EMAIL to the address that receives disk alerts}"
token="$(gcloud auth print-access-token)"
channel="$(curl -fsS -X POST -H "Authorization: Bearer $token" -H 'Content-Type: application/json' \
  "https://monitoring.googleapis.com/v3/projects/$P/notificationChannels" \
  -d '{"type":"email","displayName":"Design CI runner alerts (owner email)","labels":{"email_address":"'"$ALERT_EMAIL"'"}}' |
  python3 -c 'import json, sys; print(json.load(sys.stdin)["name"])')"
python3 -c 'import json, sys; p = json.load(open("gcp/disk-alert-policy.json")); p["notificationChannels"] = [sys.argv[1]]; print(json.dumps(p))' "$channel" |
  curl -fsS -X POST -H "Authorization: Bearer $token" -H 'Content-Type: application/json' \
    "https://monitoring.googleapis.com/v3/projects/$P/alertPolicies" -d @- >/dev/null
