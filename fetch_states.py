import requests, json, os, time
import pandas as pd

# API endpoint
url = "https://ingres.iith.ac.in/api/gec/getBusinessDataForUserOpen"

# Common base payload
base_payload = {
    "approvalLevel": 1,
    "category": "all",
    "component": "recharge",
    "computationType": "normal",
    "period": "annual",
    "verificationStatus": 1,
    "view": "admin",
    "year": "2024-2025"
}

os.makedirs("output", exist_ok=True)

# -------------------
# Step 1: COUNTRY (India level)
# -------------------
country_payload = base_payload.copy()
country_payload.update({
    "locname": "INDIA",
    "loctype": "COUNTRY",
    "locuuid": "ffce954d-24e1-494b-ba7e-0931d8ad6085",
    "parentuuid": "ffce954d-24e1-494b-ba7e-0931d8ad6085",
    "stateuuid": None
})

resp = requests.post(url, json=country_payload)
resp.raise_for_status()
india_data = resp.json()

# Save raw JSON
with open("output/india.json", "w", encoding="utf-8") as f:
    json.dump(india_data, f, indent=2)

# Format into CSV
states_rows = []
for entry in india_data:
    if entry.get("loctype") == "STATE":
        states_rows.append({
            "state": entry.get("locationName"),
            "area_total": entry.get("area", {}).get("total", {}).get("totalArea"),
            "rainfall_total": entry.get("rainfall", {}).get("total"),
            "recharge_total": entry.get("rechargeData", {}).get("total", {}).get("total"),
            "draft_total": entry.get("draftData", {}).get("total", {}).get("total"),
            "stage_of_extraction": entry.get("stageOfExtraction", {}).get("total"),
            "category": entry.get("category"),
            "uuid": entry.get("locuuid")
        })

pd.DataFrame(states_rows).to_csv("output/india.csv", index=False)
print(f"✅ Saved {len(states_rows)} states into india.csv and india.json")

# -------------------
# Step 2: DISTRICTS (per state)
# -------------------
for s in states_rows:
    state_name = s["state"].replace(" ", "_")
    state_uuid = s["uuid"]

    state_payload = base_payload.copy()
    state_payload.update({
        "locname": s["state"],
        "loctype": "STATE",
        "locuuid": state_uuid,
        "stateuuid": state_uuid,
        "parentuuid": country_payload["locuuid"]
    })

    r = requests.post(url, json=state_payload)
    if r.status_code != 200:
        print(f"❌ Failed for {s['state']}")
        continue

    state_data = r.json()

    # Save raw JSON
    with open(f"output/{state_name}.json", "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=2)

    # Format into CSV
    district_rows = []
    for entry in state_data:
        if entry.get("loctype") == "DISTRICT":
            district_rows.append({
                "state": s["state"],
                "district": entry.get("locationName"),
                "area_total": entry.get("area", {}).get("total", {}).get("totalArea"),
                "rainfall_total": entry.get("rainfall", {}).get("total"),
                "recharge_total": entry.get("rechargeData", {}).get("total", {}).get("total"),
                "draft_total": entry.get("draftData", {}).get("total", {}).get("total"),
                "stage_of_extraction": entry.get("stageOfExtraction", {}).get("total"),
                "category": entry.get("category")
            })

    if district_rows:
        pd.DataFrame(district_rows).to_csv(f"output/{state_name}.csv", index=False)
        print(f"✅ Saved {len(district_rows)} districts for {s['state']}")
    else:
        print(f"⚠ No districts found for {s['state']}")

    time.sleep(1.5)  # polite delay
