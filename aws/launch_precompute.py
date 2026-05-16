"""
Launches 4 EC2 instances in parallel for Stokes precomputation.
Each instance gets its own INSTANCE_INDEX (0-3) injected into user data.

Usage:
    python aws/launch_precompute.py \
        --key-name YOUR_KEY_PAIR_NAME \
        --security-group sg-xxxxxxxxxxxxxxxxx \
        --iam-role YOUR_S3_ROLE_NAME

Find these in AWS console:
    Key pair:       EC2 -> Key Pairs -> name (without .pem)
    Security group: EC2 -> Security Groups -> sg-xxx
    IAM role:       EC2 -> Launch instance -> IAM instance profile
"""

import argparse
import base64
import re
import boto3
from pathlib import Path


INSTANCE_TYPE = "c5.2xlarge"
AMI_ID        = "ami-0951a43515d1f167b"
REGION        = "eu-west-2"
VOLUME_GB     = 200
N_INSTANCES   = 4


def make_user_data(template: str, index: int) -> str:
    """Replace INSTANCE_INDEX=0 with the correct index."""
    return re.sub(r"INSTANCE_INDEX=\d+", f"INSTANCE_INDEX={index}", template)


def launch_instance(ec2, user_data: str, index: int, args) -> dict:
    response = ec2.run_instances(
        ImageId      = AMI_ID,
        InstanceType = INSTANCE_TYPE,
        MinCount     = 1,
        MaxCount     = 1,
        KeyName      = args.key_name,
        SecurityGroupIds = [args.security_group],
        IamInstanceProfile = {"Name": args.iam_role},
        UserData     = base64.b64encode(user_data.encode()).decode(),
        BlockDeviceMappings = [{
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": VOLUME_GB,
                "VolumeType": "gp3",
                "DeleteOnTermination": True,
            },
        }],
        TagSpecifications = [{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name",    "Value": f"polaris-precompute-{index}"},
                {"Key": "Project", "Value": "POLARIS"},
                {"Key": "Index",   "Value": str(index)},
            ],
        }],
    )
    instance = response["Instances"][0]
    return {
        "index":       index,
        "instance_id": instance["InstanceId"],
        "state":       instance["State"]["Name"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-name",       required=True, help="EC2 key pair name (without .pem)")
    parser.add_argument("--security-group", required=True, help="Security group ID (sg-xxx)")
    parser.add_argument("--iam-role",       required=True, help="IAM instance profile name")
    parser.add_argument("--dry-run",        action="store_true", help="Print user data only, don't launch")
    args = parser.parse_args()

    script_path = Path(__file__).parent / "user_data_precompute.sh"
    template    = script_path.read_text()

    if args.dry_run:
        for i in range(N_INSTANCES):
            print(f"\n{'='*60}")
            print(f"Instance {i} user data (first 20 lines):")
            lines = make_user_data(template, i).splitlines()[:20]
            print("\n".join(lines))
        return

    ec2 = boto3.client("ec2", region_name=REGION)

    print(f"Launching {N_INSTANCES} x {INSTANCE_TYPE} instances in {REGION}...")
    print(f"AMI: {AMI_ID} | Storage: {VOLUME_GB} GB gp3\n")

    launched = []
    for i in range(N_INSTANCES):
        user_data = make_user_data(template, i)
        result    = launch_instance(ec2, user_data, i, args)
        launched.append(result)
        print(f"  Instance {i}: {result['instance_id']} ({result['state']})")

    print(f"\nAll {N_INSTANCES} instances launched.")
    print("\nMonitor logs (SSH into each and run):")
    print("  tail -f /var/log/polaris_precompute.log")
    print("\nInstance IDs:")
    for r in launched:
        print(f"  Index {r['index']}: {r['instance_id']}")


if __name__ == "__main__":
    main()
