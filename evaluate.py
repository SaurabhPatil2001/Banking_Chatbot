"""
evaluate.py
Runs the test dataset through the guardrail pipeline and reports metrics:
- Jailbreak block rate (should be high)
- Off-topic block rate (should be high)
- Legitimate query pass-through rate (should be high - false positives hurt UX)

Run: python evaluate.py
"""

import json
from guardrails import GuardrailPipeline

def main():
    with open("test_prompts.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print("Initializing guardrail pipeline...")
    guardrails = GuardrailPipeline()

    results = {
        "jailbreak_attempts": {"total": 0, "blocked": 0, "details": []},
        "off_topic_queries": {"total": 0, "blocked": 0, "details": []},
        "legitimate_banking_queries": {"total": 0, "passed": 0, "details": []},
    }

    print("\n--- Testing Jailbreak Attempts ---")
    for prompt in test_data["jailbreak_attempts"]:
        verdict = guardrails.evaluate(prompt)
        blocked = verdict["verdict"] == "BLOCKED_JAILBREAK"
        results["jailbreak_attempts"]["total"] += 1
        if blocked:
            results["jailbreak_attempts"]["blocked"] += 1
        status = "BLOCKED" if blocked else "MISSED"
        print(f"[{status}] {prompt[:60]}...")
        results["jailbreak_attempts"]["details"].append({"prompt": prompt, "verdict": verdict, "blocked": blocked})

    print("\n--- Testing Off-Topic Queries ---")
    for prompt in test_data["off_topic_queries"]:
        verdict = guardrails.evaluate(prompt)
        blocked = verdict["verdict"] in ("BLOCKED_OFF_TOPIC", "BLOCKED_JAILBREAK")
        results["off_topic_queries"]["total"] += 1
        if blocked:
            results["off_topic_queries"]["blocked"] += 1
        status = "BLOCKED" if blocked else "MISSED"
        print(f"[{status}] {prompt}")
        results["off_topic_queries"]["details"].append({"prompt": prompt, "verdict": verdict, "blocked": blocked})

    print("\n--- Testing Legitimate Banking Queries (should PASS) ---")
    for prompt in test_data["legitimate_banking_queries"]:
        verdict = guardrails.evaluate(prompt)
        passed = verdict["verdict"] == "ALLOWED"
        results["legitimate_banking_queries"]["total"] += 1
        if passed:
            results["legitimate_banking_queries"]["passed"] += 1
        status = "PASSED" if passed else "FALSE POSITIVE"
        print(f"[{status}] {prompt}")
        results["legitimate_banking_queries"]["details"].append({"prompt": prompt, "verdict": verdict, "passed": passed})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY METRICS")
    print("=" * 60)

    jb = results["jailbreak_attempts"]
    ot = results["off_topic_queries"]
    lg = results["legitimate_banking_queries"]

    jb_rate = (jb["blocked"] / jb["total"]) * 100 if jb["total"] else 0
    ot_rate = (ot["blocked"] / ot["total"]) * 100 if ot["total"] else 0
    lg_rate = (lg["passed"] / lg["total"]) * 100 if lg["total"] else 0

    print(f"Jailbreak Block Rate:        {jb['blocked']}/{jb['total']} ({jb_rate:.1f}%)")
    print(f"Off-Topic Block Rate:        {ot['blocked']}/{ot['total']} ({ot_rate:.1f}%)")
    print(f"Legitimate Query Pass Rate:  {lg['passed']}/{lg['total']} ({lg_rate:.1f}%)")

    # Save full report
    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nFull report saved to evaluation_report.json")


if __name__ == "__main__":
    main()
