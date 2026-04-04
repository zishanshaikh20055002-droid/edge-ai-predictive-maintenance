from slowapi import Limiter
from slowapi.util import get_remote_address

# ── Limiter instance ──────────────────────────────────────────
# get_remote_address extracts the client IP from the request.
# In production behind a proxy (nginx, Cloudflare), use:
#   get_remote_address_from_forwarded  (reads X-Forwarded-For)
# For now, direct IP is fine for local/dev use.

limiter = Limiter(key_func=get_remote_address)

# ── Limit strings reference ───────────────────────────────────
# Format: "N/period"  where period = second | minute | hour | day
#
#   "5/minute"    → 5 requests per minute per IP
#   "100/hour"    → 100 requests per hour per IP
#   "1/second"    → 1 request per second (hard throttle)
#
# Multiple limits can be stacked with a list:
#   @limiter.limit("5/minute;100/hour")