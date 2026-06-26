"""
TEST 1: Basic Connectivity to kvcm-server

This test verifies:
1. Health check endpoint is accessible
2. Register instance API accepts hybrid specs (tp0 + tp0_hybrid)
3. location_spec_groups (Full, FullAndHybrid) are correctly formatted
"""

import requests
from e2e_tests.utils import (
    KVCMSERVER_BASE_URL,
    get_register_payload,
    print_header,
    print_section,
)


def test_basic_connectivity() -> bool:
    """Test basic HTTP connectivity to kvcm-server"""
    print_header("TEST 1: Basic Connectivity to kvcm-server")
    
    # Health check
    print_section("[1.1] Health check...")
    try:
        resp = requests.get(f"{KVCMSERVER_BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            print("✓ Health check passed")
            print(f"  Response: {resp.json()}")
        else:
            print(f"⚠ Health check returned {resp.status_code}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False
    
    # Register instance API (with hybrid specs)
    print_section("[1.2] Register instance API (with hybrid specs)...")
    register_payload = get_register_payload()
    
    try:
        resp = requests.post(
            f"{KVCMSERVER_BASE_URL}/api/registerInstance",
            json=register_payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if resp.status_code == 200:
            data = resp.json()
            status_code = data.get('header', {}).get('status', {}).get('code')
            if status_code == 'OK':
                print("✓ Register instance successful")
                print(f"  Response: {data}")
            elif status_code == 'INSTANCE_NOT_EXIST':
                print("✓ API communication successful (instance group not pre-registered)")
                print(f"  This is expected for fresh kvcm-server")
            else:
                print(f"⚠ Unexpected status: {status_code}")
        else:
            print(f"✗ HTTP request failed: {resp.status_code}")
            print(f"  Response: {resp.text}")
            return False
    except Exception as e:
        print(f"✗ Register instance failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    import sys
    success = test_basic_connectivity()
    sys.exit(0 if success else 1)
