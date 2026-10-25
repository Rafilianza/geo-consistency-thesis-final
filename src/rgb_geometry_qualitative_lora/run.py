"""Run one LoRA arm or write its combined report after the first study."""

import sys

from rgb_geometry_qualitative.run import main as run_policy
from . import protocol


def write_report(args, config):
    from . import reporting
    return reporting.write_report(args.output_root, config, args.reference_root)


def main(argv=None):
    return run_policy(argv, study_protocol=protocol, report_writer=write_report)


if __name__ == "__main__":
    sys.exit(main())
