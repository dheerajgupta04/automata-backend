import json

def generate_output(validation_report, bpm):
    output = {
        "validation": validation_report,
        "heart_rate_bpm": bpm
    }
    print(json.dumps(output, indent=4))