terraform {
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
  }
}

provider "cloudflare" {
  account_id = var.cloudflare_account_id
}

resource "cloudflare_zone" "telemetry" {
  account_id = var.cloudflare_account_id
  name       = var.telemetry_domain
  plan       = "free"
}

resource "cloudflare_record" "telemetry_root" {
  zone_id = cloudflare_zone.telemetry.id
  name    = "@"
  type    = "AAAA"
  value   = "100::"
  proxied = true
  ttl     = 1
  comment = "Placeholder record proxied through Cloudflare so the Worker route can attach."
}

resource "cloudflare_workers_script" "telemetry" {
  name    = "runmat-telemetry"
  content = file("${path.module}/worker.js")

  plain_text_binding {
    name = "POSTHOG_API_KEY"
    text = var.posthog_api_key
  }

  plain_text_binding {
    name = "POSTHOG_HOST"
    text = var.posthog_host
  }

  plain_text_binding {
    name = "INGESTION_KEY"
    text = var.telemetry_ingestion_key
  }

  plain_text_binding {
    name = "GA_MEASUREMENT_ID"
    text = var.ga_measurement_id
  }

  plain_text_binding {
    name = "GA_API_SECRET"
    text = var.ga_api_secret
  }
}

resource "cloudflare_workers_route" "telemetry" {
  zone_id     = cloudflare_zone.telemetry.id
  script_name = cloudflare_workers_script.telemetry.name
  pattern     = "${cloudflare_zone.telemetry.name}/*"
}

resource "cloudflare_record" "udp_endpoint" {
  zone_id = cloudflare_zone.telemetry.id
  name    = trimsuffix(var.telemetry_udp_subdomain, ".${cloudflare_zone.telemetry.name}")
  type    = "AAAA"
  value   = "100::"
  proxied = true
  ttl     = 1
  comment = "UDP intake host (Cloudflare Spectrum origin)"
}

resource "cloudflare_spectrum_application" "telemetry_udp" {
  zone_id      = cloudflare_zone.telemetry.id
  protocol     = "udp"
  traffic_type = "direct"
  name         = "runmat-telemetry-udp"
  edge_port    = 7846
  origin_port  = 443
  dns {
    type = "CNAME"
    name = cloudflare_record.udp_endpoint.name
  }
  origin_dns {
    name = cloudflare_zone.telemetry.name
  }
  tls = "flexible"
}

output "telemetry_https_endpoint" {
  description = "HTTPS endpoint the CLI can post telemetry to"
  value       = "https://${cloudflare_zone.telemetry.name}/ingest"
}

output "telemetry_name_servers" {
  description = "Name servers to add as NS records for the delegated subdomain"
  value       = cloudflare_zone.telemetry.name_servers
}

output "telemetry_udp_endpoint" {
  description = "Spectrum UDP hostname:port"
  value       = "${var.telemetry_udp_subdomain}:7846"
}