import os
import sys
import json
import time
import subprocess
from pathlib import Path

# Ensure we can import said_lam from the current directory
sys.path.append(str(Path.cwd()))

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def test_comprehensive_workflow():
    try:
        from said_lam import LAM
        
        print("=== SAID-LAM COMPREHENSIVE TEST SUITE ===")
        
        # 1. TEST: TIER LIMITS & SCA UNLOCK
        print("\n[SECTION 1: TIER & SCA VERIFICATION]")
        model = LAM("SAIDResearch/SAID-LAM-v1")
        print(f"Initial State: Tier={model.tier}, MaxTokens={model.max_tokens}")
        
        if model.tier == "FREE" and model.max_tokens == 12000:
            print("✅ Initial FREE tier limits verified.")
        else:
            print("❌ FREE tier limits mismatch.")

        # Register for BETA
        print("Registering for BETA...")
        model.register_beta("comprehensive_test@example.com")
        print(f"Post-Registration: Tier={model.tier}, MaxTokens={model.max_tokens}")
        
        if model.tier == "BETA" and model.max_tokens == 32000:
            print("✅ BETA tier (32K tokens) verified.")
        else:
            print("❌ BETA tier upgrade failed.")

        # Test SCA (Indexing + Search)
        print("Verifying SCA (Indexing/Search)...")
        model.index("sca_doc", "Crystalline Attention enables deterministic recall at scale.")
        model.build_index()
        results = model.search("Crystalline Attention")
        print(f"Search results: {results}")
        if len(results) > 0 and results[0][0] == "sca_doc":
            print("✅ SCA Functionality verified.")
        else:
            print("❌ SCA Search failed.")

        # 2. TEST: REVOCATION BEHAVIOR
        print("\n[SECTION 2: REVOCATION TEST]")
        
        # Find the license key we just created
        license_path = Path.home() / ".lam" / "lam_license.json"
        if not license_path.exists():
            print("❌ Could not find license file for revocation test.")
            return

        with open(license_path, "r") as f:
            lic_data = json.load(f)
        
        license_key = lic_data["license_key"]
        print(f"Acting on License Key: {license_key}")

        # Simulate Admin Revocation via Wrangler
        print(f"Simulating Revocation for {license_key}...")
        kv_data = run_command(f"wrangler kv key get --binding=KV --remote beta:{license_key}")
        if not kv_data:
             print("❌ Failed to fetch KV data for revocation simulation.")
             return
        
        data = json.loads(kv_data)
        data["revoked"] = True
        
        # Use a temporary file to pipe the JSON back to wrangler
        tmp_kv = "/tmp/revoked_kv.json"
        with open(tmp_kv, "w") as f:
            json.dump(data, f)
        
        run_command(f"wrangler kv key put --binding=KV --remote beta:{license_key} --path={tmp_kv}")
        print("Admin set 'revoked: true' in Cloudflare KV.")

        # Verify Revocation on next check
        print("Re-validating with Revocation active...")
        # Since the engine caches the tier for the session, we recreate the model to force a check
        # Or we call the internal validation directly if we could, but a new model is cleaner test
        model_revoked = LAM("SAIDResearch/SAID-LAM-v1")
        print(f"Tier after Revocation: {model_revoked.tier}, MaxTokens={model_revoked.max_tokens}")
        
        if model_revoked.tier == "FREE":
            print("✅ Revocation logic verified: downgraded to FREE.")
        else:
            print("❌ Revocation logic failed: still in BETA.")

        # 3. TEST: RENEWAL BEHAVIOR
        print("\n[SECTION 3: RENEWAL TEST]")
        print("Simulating Renewal (re-registering a revoked/expired key)...")
        # In our worker, register_beta() will now reset 'revoked' to false and extend expiry if found
        model_revoked.register_beta("comprehensive_test@example.com")
        print(f"Tier after Renewal Call: {model_revoked.tier}, MaxTokens={model_revoked.max_tokens}")
        
        if model_revoked.tier == "BETA":
            print("✅ Renewal logic verified: upgraded back to BETA.")
        else:
            print("❌ Renewal logic failed.")

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_comprehensive_workflow()
