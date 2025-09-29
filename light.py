#!/usr/bin/env python3
import sys
from control import tcControl

def main():
    if len(sys.argv) != 2 or sys.argv[1].lower() not in ("on", "off"):
        print("Usage: light_control.py [on|off]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    tc = tcControl()

    if cmd == "on":
        tc.light_on()
        print("[INFO] Light turned ON")
    elif cmd == "off":
        tc.light_off()
        print("[INFO] Light turned OFF")

if __name__ == "__main__":
    main()
