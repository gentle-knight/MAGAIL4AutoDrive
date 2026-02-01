"""
Thin wrapper: forwards to scripts/visualize_trained_policy.py --policy_type bc.
Use: python visualize_bc.py [--model_path models/bc/policy_best.pt] [other args...]
Or call directly: python scripts/visualize_trained_policy.py --policy_type bc --model_path models/bc/policy_best.pt
"""
import subprocess
import sys
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(script_dir, "scripts", "visualize_trained_policy.py")
    cmd = [sys.executable, script, "--policy_type", "bc"] + sys.argv[1:]
    sys.exit(subprocess.run(cmd).returncode)

if __name__ == "__main__":
    main()
