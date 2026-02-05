terraform {
  backend "gcs" {}

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.32"
    }
    dns = {
      source  = "hashicorp/dns"
      version = "~> 3.4"
    }
  }
}

locals {
  telemetry_domains = length(var.telemetry_domains) > 0 ? var.telemetry_domains : [
    var.telemetry_domain,
    "telemetry.runmat.com",
  ]
  telemetry_domain_map = {
    for domain in toset(local.telemetry_domains) : domain => {
      zone_name     = "telemetry-${replace(domain, ".", "-")}"
      zone_dns_name = "${domain}."
      udp_subdomain = "udp.${domain}"
    }
  }
  worker_image = var.worker_image != "" ? var.worker_image : "us-docker.pkg.dev/${var.project_id}/telemetry/worker:latest"
  udp_image    = var.udp_forwarder_image != "" ? var.udp_forwarder_image : "us-docker.pkg.dev/${var.project_id}/telemetry/udp-forwarder:latest"
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
  for_each    = local.telemetry_domain_map
  name        = each.value.zone_name
  dns_name    = each.value.zone_dns_name
  description = "Delegated zone for RunMat telemetry"
}

resource "google_dns_record_set" "ns_records" {
  for_each     = local.telemetry_domain_map
  managed_zone = google_dns_managed_zone.telemetry[each.key].name
  name         = each.value.zone_dns_name
  type         = "NS"
  ttl          = 300
  rrdatas      = google_dns_managed_zone.telemetry[each.key].name_servers
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
  for_each = local.telemetry_domain_map
  location = var.region
  name     = each.key

  metadata {
    namespace = var.project_id
  }

  spec {
    route_name = google_cloud_run_service.telemetry.name
  }
}

data "dns_a_record_set" "ghs" {
  host = "ghs.googlehosted.com"
}

data "dns_aaaa_record_set" "ghs" {
  host = "ghs.googlehosted.com"
}

resource "google_dns_record_set" "domain_mapping_a" {
  for_each     = local.telemetry_domain_map
  managed_zone = google_dns_managed_zone.telemetry[each.key].name
  name         = each.value.zone_dns_name
  type         = "A"
  ttl          = 300
  rrdatas      = sort(data.dns_a_record_set.ghs.addrs)
}

resource "google_dns_record_set" "domain_mapping_aaaa" {
  for_each     = local.telemetry_domain_map
  managed_zone = google_dns_managed_zone.telemetry[each.key].name
  name         = each.value.zone_dns_name
  type         = "AAAA"
  ttl          = 300
  rrdatas      = sort(data.dns_aaaa_record_set.ghs.addrs)
}

resource "google_compute_firewall" "udp_forwarder" {
  name    = "telemetry-udp-allow"
  network = "default"
  allow {
    protocol = "udp"
    ports    = ["7846"]
  }
  direction     = "INGRESS"
  target_tags   = ["telemetry-udp"]
  source_ranges = ["0.0.0.0/0"]
}

resource "google_compute_firewall" "udp_health" {
  name    = "telemetry-udp-health-fw"
  network = "default"
  allow {
    protocol = "tcp"
    ports    = ["9000"]
  }
  direction     = "INGRESS"
  target_tags   = ["telemetry-udp"]
  source_ranges = ["35.191.0.0/16", "130.211.0.0/22"]
}

resource "google_compute_address" "udp" {
  name   = "telemetry-udp-ip"
  region = var.region
}

resource "google_compute_health_check" "udp" {
  name = "telemetry-udp-health"
  tcp_health_check {
    port = 9000
  }
}

resource "google_compute_instance_template" "udp" {
  name_prefix  = "telemetry-udp-"
  machine_type = var.udp_machine_type
  tags         = ["telemetry-udp"]

  lifecycle {
    create_before_destroy = true
  }

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
    - name: HEALTH_PORT
      value: "9000"
  restartPolicy: Always
EOT
  }
}

resource "google_compute_instance_group_manager" "udp" {
  name               = "telemetry-udp-mig"
  zone               = "${var.region}-a"
  base_instance_name = "telemetry-udp"
  target_size        = var.udp_min_instances
  version {
    instance_template = google_compute_instance_template.udp.self_link
  }
  auto_healing_policies {
    health_check      = google_compute_health_check.udp.self_link
    initial_delay_sec = 30
  }
  target_pools = [google_compute_target_pool.udp.self_link]
}

resource "google_compute_target_pool" "udp" {
  name   = "telemetry-udp-pool"
  region = var.region
}

resource "google_compute_forwarding_rule" "udp" {
  name                  = "telemetry-udp-forwarding"
  region                = var.region
  load_balancing_scheme = "EXTERNAL"
  ip_protocol           = "UDP"
  port_range            = "7846"
  target                = google_compute_target_pool.udp.self_link
  ip_address            = google_compute_address.udp.self_link
}

resource "google_dns_record_set" "udp_dns" {
  for_each     = local.telemetry_domain_map
  managed_zone = google_dns_managed_zone.telemetry[each.key].name
  name         = "${each.value.udp_subdomain}."
  type         = "A"
  ttl          = 120
  rrdatas      = [google_compute_address.udp.address]
}

output "telemetry_https_endpoints" {
  description = "HTTPS endpoints the CLI can post telemetry to"
  value       = [for domain in local.telemetry_domains : "https://${domain}/ingest"]
}

output "telemetry_name_servers" {
  description = "Name servers to add as NS records for delegated subdomains"
  value       = { for domain, zone in google_dns_managed_zone.telemetry : domain => zone.name_servers }
}

output "telemetry_udp_endpoints" {
  description = "UDP hostname:port for CLI datagrams"
  value       = [for domain, meta in local.telemetry_domain_map : "${meta.udp_subdomain}:7846"]
}
