import re


def extract_ground_truth(sample):
    """
    Extract last assistant tool call.
    """

    for msg in reversed(sample["messages"]):
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            tc = msg["tool_calls"][0]

            return {
                "tool_name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"]
            }

    return None


def build_inference_messages(sample):
    """
    Remove final assistant tool call.
    """

    msgs = sample["messages"][:]

    for i in range(len(msgs) - 1, -1, -1):
        msg = msgs[i]

        if msg.get("role") == "assistant" and "tool_calls" in msg:
            del msgs[i]
            break

    return msgs


def parse_prediction(text):

    match = re.search(
        r"call:([^{]+)\{(.*?)\}",
        text,
        re.DOTALL
    )

    if not match:
        return None, {}

    tool_name = match.group(1).strip()

    args_block = match.group(2)

    args = {}

    arg_matches = re.findall(
        r"(\w+):<escape>(.*?)<escape>",
        args_block
    )

    for k, v in arg_matches:
        args[k] = v

    return tool_name, args