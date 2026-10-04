"""Cloudflare Access check for the remote voice page.

Every request that arrives through the tunnel must carry the
Cf-Access-Jwt-Assertion token that Access adds after its login. It is
verified here, signature and all, against the team's published keys:
the audience must be this application's AUD tag, and the email must be
the one allowed in backtalk.json. Checking a plain email header instead
would trust whatever a request claims if Access were ever switched off
in front of the hostname; the signature can't be faked.

Fails closed: no config, no token, an unreachable key server, or a bad
signature all mean "refused".
"""
import asyncio
import time

from backtalk.config import CFG

_KEYS = {"at": 0.0, "client": None}
_KEY_TTL = 3600


def _client():
    import jwt
    team = (CFG.get("remote_access_team") or "").strip().rstrip("/")
    if not team:
        return None
    if _KEYS["client"] is None or time.time() - _KEYS["at"] > _KEY_TTL:
        _KEYS["client"] = jwt.PyJWKClient(
            f"https://{team}/cdn-cgi/access/certs", cache_keys=True)
        _KEYS["at"] = time.time()
    return _KEYS["client"]


def _verify_sync(token: str) -> tuple[bool, str]:
    import jwt
    aud = (CFG.get("remote_access_aud") or "").strip()
    allowed = (CFG.get("remote_allowed_email") or "").strip().lower()
    team = (CFG.get("remote_access_team") or "").strip().rstrip("/")
    if not (aud and allowed and team):
        return False, "remote access is not configured (team/aud/email)"
    if not token:
        return False, "no Access token (is Access in front of the hostname?)"
    try:
        key = _client().get_signing_key_from_jwt(token).key
        claims = jwt.decode(token, key, algorithms=["RS256"], audience=aud,
                            issuer=f"https://{team}")
    except Exception as e:
        return False, f"Access token rejected ({type(e).__name__})"
    email = str(claims.get("email", "")).lower()
    if email != allowed:
        return False, f"Access token for an unexpected account"
    return True, email


async def verify(token: str | None) -> tuple[bool, str]:
    return await asyncio.get_running_loop().run_in_executor(
        None, _verify_sync, token or "")
