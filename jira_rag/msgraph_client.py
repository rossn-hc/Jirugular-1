# jira_rag/msgraph_client.py
#!/usr/bin/env python
"""
MSGraphClient – minimal wrapper around Microsoft Graph using MSAL + requests.
Adapted from your previous project with:
- added ProtocolError import
- optional iutils helpers (graceful fallback if absent)
"""

import time
import jwt
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import ProtocolError

try:
    import iutils  # optional
except Exception:
    iutils = None

import msal


class MSGraphClient:
    """
    Generic Microsoft Graph client with token caching + auto-refresh and paged GETs.
    """

    def __init__(self, tenant_id, client_id, client_secret, baseline="jira-rag"):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.scopes = ["https://graph.microsoft.com/.default"]
        self.baseline = baseline
        self.token = None
        self.token_expiration = 0  # unix seconds

        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        # Prime token
        self.access_token = self.get_access_token()

    def get_access_token(self):
        now = time.time()
        if self.token and now < self.token_expiration:
            return self.token

        app = msal.ConfidentialClientApplication(
            client_id=self.client_id,
            authority=f"https://login.microsoftonline.com/{self.tenant_id}",
            client_credential=self.client_secret,
        )
        token_response = app.acquire_token_for_client(scopes=self.scopes)
        if "access_token" not in token_response:
            raise RuntimeError(
                f"Could not obtain access token: {token_response.get('error_description')}"
            )

        self.token = token_response["access_token"]
        decoded = jwt.decode(self.token, options={"verify_signature": False})
        self.token_expiration = decoded.get("exp", int(now) + 3000)
        return self.token

    def query_msgraph(self, base_url=None, params=None, max_retries=5, headers=None):
        headers = headers or {
            "Authorization": f"Bearer {self.get_access_token()}",
            "Content-Type": "application/json",
        }

        results = []
        retries = 0
        next_link = None
        old_link = None

        while True:
            url = next_link or base_url
            try:
                resp = self.session.get(url, headers=headers, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                new_records = data.get("value", []) if isinstance(data, dict) else []
                results.extend(new_records)
                next_link = data.get("@odata.nextLink") if isinstance(data, dict) else None
                if not next_link or old_link == next_link:
                    break
                old_link = next_link

            except requests.exceptions.ChunkedEncodingError as e:
                retries += 1
                if retries > max_retries:
                    raise
                time.sleep(5)

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 429:
                    retries += 1
                    if retries > max_retries:
                        raise
                    retry_after = int(e.response.headers.get("Retry-After", 10))
                    time.sleep(retry_after)
                elif status == 401:
                    headers["Authorization"] = f"Bearer {self.get_access_token()}"
                    continue
                elif status in (400,):
                    # bad request; stop paging
                    break
                else:
                    # return what we have
                    return results

            except (requests.exceptions.RequestException, ProtocolError):
                retries += 1
                if retries > max_retries:
                    raise
                time.sleep(5)

        return results

    # ---------------- Example endpoints you used ----------------

    def query_sign_ins(self, email=None, start_date=None, end_date=None,
                       app_display_name="Windows Sign In", ip_range=None, top=None, max_retries=5):
        params = {}
        clauses = []
        if email:
            clauses.append(f"userPrincipalName eq '{email}'")
        if app_display_name:
            clauses.append(f"(appDisplayName eq '{app_display_name}' or appDisplayName eq 'Microsoft Teams')")
        if start_date and end_date:
            clauses.append(
                "status/errorCode eq 0 and "
                "(appDisplayName eq 'Windows Sign In' or appDisplayName eq 'Microsoft Teams') and "
                f"(createdDateTime ge {start_date} and createdDateTime le {end_date})"
            )
        if clauses:
            params["$filter"] = " and ".join(clauses)
        params["$select"] = (
            "id,createdDateTime,userPrincipalName,userDisplayName,appDisplayName,userId,ipAddress,"
            "deviceDetail,location,clientAppUsed"
        )
        params["$orderby"] = "createdDateTime desc"
        if top:
            params["$top"] = top

        base_url = "https://graph.microsoft.com/v1.0/auditLogs/signIns"
        data = self.query_msgraph(base_url, params=params, max_retries=max_retries)

        # Optional post-processing if iutils is available
        if ip_range and iutils:
            out = []
            for r in data:
                ip = r.get("ipAddress")
                r["goc_building"] = iutils.is_ip_in_range(ip, ip_range)
                r["baseline"] = self.baseline
                out.append(r)
            return out

        return data

    def query_logs(self, start_date=None, end_date=None, category=None,
                   app_displayName=None, ip_range=None, top=None, max_retries=5):
        params = {}
        clauses = []
        if category:
            clauses.append(f"category eq '{category}'")
        if app_displayName:
            clauses.append(f"initiatedBy/app/displayName eq '{app_displayName}'")
        if clauses:
            params["$filter"] = " and ".join(clauses)
        if top:
            params["$top"] = top

        base_url = "https://graph.microsoft.com/v1.0/auditLogs/directoryAudits"
        data = self.query_msgraph(base_url, params=params, max_retries=max_retries)
        return data

    def get_user_presence(self, email=None, top=None, max_retries=5):
        base_url = f"https://graph.microsoft.com/v1.0/users/{email}/presence"
        headers = {
            "Authorization": f"Bearer {self.get_access_token()}",
            "Content-Type": "application/json",
        }
        try:
            # query_msgraph returns list; presence endpoint returns dict – call requests directly instead
            resp = self.session.get(base_url, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Failed to fetch presence for {email}: {e}")
            return None

    def query_users(self, email=None, start_date=None, end_date=None,
                    top=None, max_retries=5, orbit=False, presence=False, accountType=None):
        params = {}
        clauses = []
        if email:
            clauses.append(f"userPrincipalName eq '{email}'")
        if accountType:
            clauses.append(f"userType eq '{accountType}'")
        if clauses:
            params["$filter"] = " and ".join(clauses)
        if top:
            params["$top"] = top

        base_url = "https://graph.microsoft.com/beta/users"
        data = self.query_msgraph(base_url, params=params, max_retries=max_retries)

        results = []
        for r in data:
            # attach a few computed fields
            sign = r.get("signInActivity", {})
            r["lastSignInDateTime"] = sign.get("lastSignInDateTime")
            r["baseline"] = self.baseline
            if orbit and r.get("userPrincipalName"):
                r["orbit"] = self.query_user_people(r["userPrincipalName"], top=5, max_retries=max_retries)
            if presence and r.get("userPrincipalName"):
                r["presence"] = self.get_user_presence(r["userPrincipalName"], top=top, max_retries=max_retries)
            results.append(r)
        return results

    def query_devices(self, start_date=None, end_date=None, email=None, device_name=None,
                      top=None, max_retries=5):
        params = {}
        clauses = []
        if email:
            clauses.append(f"userPrincipalName eq '{email}'")
        if device_name:
            clauses.append(f"deviceName eq '{device_name}'")
        if start_date and end_date:
            clauses.append(f"lastSyncDateTime ge {start_date} and lastSyncDateTime le {end_date}")
        if clauses:
            params["$filter"] = " and ".join(clauses)
        params["$orderby"] = "lastSyncDateTime desc"
        if top:
            params["$top"] = top

        base_url = "https://graph.microsoft.com/v1.0/deviceManagement/managedDevices"
        return self.query_msgraph(base_url, params=params, max_retries=max_retries)

    def query_user_people(self, email, top=5, max_retries=5):
        params = {"$select": "scoredEmailAddresses"}
        if top:
            params["$top"] = top
        base_url = f"https://graph.microsoft.com/v1.0/users/{email}/people"

        # first page
        first = self.query_msgraph(base_url, params=params, max_retries=max_retries)
        # query_msgraph returns a flat list already (paged loop inside). Just crop to top.
        return first[:top] if isinstance(first, list) else []
