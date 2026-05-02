#!/usr/bin/env python3

from control import tcControl


def main():
    controller = tcControl()
    controller.feed_reel_off()
    try:
        controller.feed_reel_on()
        print("Feed reel is ON. Press Enter or Ctrl+C to stop.")
        try:
            input()
        except EOFError:
            pass
        except KeyboardInterrupt:
            print()
    finally:
        controller.feed_reel_off()
        controller.clean_up()
        print("Feed reel is OFF.")


if __name__ == "__main__":
    main()