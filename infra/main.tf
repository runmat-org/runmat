terraform {
  backend "gcs" {}

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.32"
    }
  }
}

locals {
  zone_dns_name = "${var.telemetry_domain}."
  worker_image  = var.worker_image != "" ? var.worker_image : "us-docker.pkg.dev/${var.project_id}/telemetry/worker:latest"
  udp_image     = var.udp_forwarder_image != "" ? var.udp_forwarder_image : "us-docker.pkg.dev/${var.project_id}/telemetry/udp-forwarder:latest"
  udp_enabled   = var.enable_udp_forwarder && var.udp_forwarder_image != ""
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "services" {
  for_each = toset([
    "run.googleapis.com",
    "dns.googleapis.com",
    "compute.googleapis.com",
    "iam.googleapis.com",
    "artifactregistry.googleapis.com",
  ])
  service = each.value
}

resource "google_dns_managed_zone" "telemetry" {
  name        = "telemetry-${replace(var.telemetry_domain, ".", "-")}"
  dns_name    = local.zone_dns_name
  description = "Delegated zone for RunMat telemetry"
}

resource "google_dns_record_set" "ns_records" {
  managed_zone = google_dns_managed_zone.telemetry.name
  name         = local.zone_dns_name
  type         = "NS"
  ttl          = 300
  rrdatas      = google_dns_managed_zone.telemetry.name_servers
}

resource "google_cloud_run_service" "telemetry" {
  name     = "runmat-telemetry"
  location = var.region
  template {
    spec {
      containers {
        image = local.worker_image
        env {
          name  = "POSTHOG_API_KEY"
          value = var.posthog_api_key
        }
        env {
          name  = "POSTHOG_HOST"
          value = var.posthog_host
        }
        env {
          name  = "INGESTION_KEY"
          value = var.telemetry_ingestion_key
        }
        env {
          name  = "GA_MEASUREMENT_ID"
          value = var.ga_measurement_id
        }
        env {
          name  = "GA_API_SECRET"
          value = var.ga_api_secret
        }
      }
    }
  }
  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [google_project_service.services]
}

resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_service.telemetry.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_domain_mapping" "telemetry" {
  location = var.region
  name     = var.telemetry_domain

  metadata {
    namespace = var.project_id
  }

  spec {
    route_name = google_cloud_run_service.telemetry.name
  }
}

locals {
  domain_records = try(google_cloud_run_domain_mapping.telemetry.status[0].resource_records, [])
}

resource "google_dns_record_set" "domain_mapping" {
  for_each     = { for record in local.domain_records : record.name => record }
  managed_zone = google_dns_managed_zone.telemetry.name
  name         = each.value.name
  type         = each.value.type
  ttl          = 300
  rrdatas      = [each.value.rrdata]
  depends_on   = [google_cloud_run_domain_mapping.telemetry]
}

resource "google_compute_firewall" "udp_forwarder" {
  count   = local.udp_enabled ? 1 : 0
  name    = "telemetry-udp-allow"
  network = "default"
  allows {
    protocol = "udp"
    ports    = ["7846"]
  }
  direction     = "INGRESS"
  target_tags   = ["telemetry-udp"]
  source_ranges = ["0.0.0.0/0"]
}

resource "google_compute_address" "udp" {
  count  = local.udp_enabled ? 1 : 0
  name   = "telemetry-udp-ip"
  region = var.region
}

resource "google_compute_instance_template" "udp" {
  count        = local.udp_enabled ? 1 : 0
  name_prefix  = "telemetry-udp-"
  machine_type = var.udp_machine_type
  tags         = ["telemetry-udp"]

  disk {
    auto_delete  = true
    boot         = true
    source_image = "projects/cos-cloud/global/images/family/cos-stable"
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata = {
    "gce-container-declaration" = <<-EOT
spec:
  containers:
  - name: udp-forwarder
    image: ${local.udp_image}
    env:
    - name: TELEMETRY_HTTP_ENDPOINT
      value: "${google_cloud_run_service.telemetry.status[0].url}/ingest"
    - name: TELEMETRY_INGESTION_KEY
      value: "${var.telemetry_ingestion_key}"
    - name: FORWARDER_CONCURRENCY
      value: "8"
  restartPolicy: Always
EOT
  }
}

resource "google_compute_region_health_check" "udp" {
  count  = local.udp_enabled ? 1 : 0
  name   = "telemetry-udp-health"
  region = var.region
  udp_health_check {
    port = 7846
  }
}

resource "google_compute_region_instance_group_manager" "udp" {
  count              = local.udp_enabled ? 1 : 0
  name               = "telemetry-udp-mig"
  region             = var.region
  base_instance_name = "telemetry-udp"
  target_size        = var.udp_min_instances
  version {
    instance_template = google_compute_instance_template.udp[0].self_link
  }
  auto_healing_policies {
    health_check      = google_compute_region_health_check.udp[0].self_link
    initial_delay_sec = 30
  }
}

resource "google_compute_target_pool" "udp" {
  count     = local.udp_enabled ? 1 : 0
  name      = "telemetry-udp-pool"
  region    = var.region
  instances = local.udp_enabled ? [google_compute_region_instance_group_manager.udp[0].instance_group] : []
}

resource "google_compute_forwarding_rule" "udp" {
  count                 = local.udp_enabled ? 1 : 0
  name                  = "telemetry-udp-forwarding"
  region                = var.region
  load_balancing_scheme = "EXTERNAL"
  ip_protocol           = "UDP"
  port_range            = "7846"
  target                = google_compute_target_pool.udp[0].self_link
  ip_address            = google_compute_address.udp[0].self_link
}

resource "google_dns_record_set" "udp_dns" {
  count        = local.udp_enabled ? 1 : 0
  managed_zone = google_dns_managed_zone.telemetry.name
  name         = "${var.telemetry_udp_subdomain}."
  type         = "A"
  ttl          = 120
  rrdatas      = [google_compute_address.udp[0].address]
}

output "telemetry_https_endpoint" {
  description = "HTTPS endpoint the CLI can post telemetry to"
  value       = "https://${var.telemetry_domain}/ingest"
}

output "telemetry_name_servers" {
  description = "Name servers to add as NS records for the delegated subdomain"
  value       = google_dns_managed_zone.telemetry.name_servers
}

output "telemetry_udp_endpoint" {
  description = "UDP hostname:port (null when disabled)"
  value       = local.udp_enabled ? "${var.telemetry_udp_subdomain}:7846" : null
}