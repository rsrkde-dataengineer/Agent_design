"""Command-line entry point.

python -m helpdesk "Where is my order A-1001?"     one question
python -m helpdesk                                 chat on one thread until Ctrl-D
"""

import argparse
import logging
import sys

from helpdesk.bootstrap import build_harness
from helpdesk.core.context import RequestContext
from helpdesk.settings import Settings


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    harness = build_harness(Settings())
    context = RequestContext(customer_id=args.customer, customer_name=args.name)

    if args.message:
        try:
            result = harness.run(args.agent, args.message, context, thread_id=args.thread)
        except Exception as exc:
            sys.exit(f"Run failed: {exc}")
        print(f"{result.answer}\n\n[{result.status} | thread {result.thread_id}]")
        return

    thread_id = args.thread
    print(f"Talking to the {args.agent} agent as {args.name}. Press Ctrl-D to stop.")
    while True:
        try:
            message = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        try:
            result = harness.run(args.agent, message, context, thread_id=thread_id)
        except Exception as exc:
            print(f"\n[run failed: {exc}]")
            continue
        thread_id = result.thread_id
        print(f"\n{args.agent}> {result.answer}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="helpdesk", description="Talk to the helpdesk agents.")
    parser.add_argument("message", nargs="?", help="ask one question and exit")
    parser.add_argument("--agent", default="supervisor", help="agent to talk to")
    parser.add_argument("--thread", help="continue an earlier conversation")
    parser.add_argument(
        "--customer", default="c-100", help="customer id (demo data has c-100, c-200)"
    )
    parser.add_argument("--name", default="Sam", help="customer's name")
    parser.add_argument("-v", "--verbose", action="store_true", help="log model and tool calls")
    return parser.parse_args()


if __name__ == "__main__":
    main()
