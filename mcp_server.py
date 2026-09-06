"""
MCP Server for Kahneman-Tversky KTO Utility Loss Skill.
"""

import json
import sys
from client import KTOLossEvaluator

KTO = KTOLossEvaluator()


def handle_request(req: dict) -> dict:
    method = req.get("method")
    params = req.get("params", {})

    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "evaluate_kto_sample",
                    "description": "Evaluate KTO loss on unpaired binary feedback (thumbs up / down)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "policy_logprob": {"type": "number"},
                            "ref_logprob": {"type": "number"},
                            "is_desirable": {"type": "boolean"}
                        },
                        "required": ["policy_logprob", "ref_logprob", "is_desirable"]
                    }
                }
            ]
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "evaluate_kto_sample":
            res = KTO.evaluate_sample(
                args["policy_logprob"],
                args["ref_logprob"],
                args["is_desirable"]
            )
            return {"content": [{"type": "text", "text": json.dumps(res)}]}

        return {"error": f"Unknown tool: {tool_name}"}

    return {"error": f"Unknown method: {method}"}


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            resp["id"] = req.get("id")
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
