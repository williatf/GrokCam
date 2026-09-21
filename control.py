import time

from mcp23s17 import MCP23S17


class tcControl:
    def __init__(self):
        self._device = MCP23S17()
        self._closed = False
        for pin in range(11):
            self._device.configure_output(pin)
        self.STEPPER_PINS = [101, 102, 100]  # step, dir, enable
        self.STEPPER_PINS2 = [104, 103, 105]
        self.REEL_PINS = [106, 107]
        self.FEED_REEL_PIN = self.REEL_PINS[0]
        self.TAKEUP_REEL_PIN = self.REEL_PINS[1]
        self.LED_PIN = 108
        self.SHUTTER_PINS = [109, 110]
        self.feed_steps_taken = 0
        self.takeup_steps_taken = 0
        self.takeup_pulse_sequence = 0
        self._adaptive_takeup_enabled = False
        self._takeup_frame_counter = 0
        self._takeup_interval_frames = None
        self._last_takeup_telemetry = {
            'takeup_active': False,
            'takeup_pulse_started': False,
            'takeup_pulse_sequence': 0,
            'takeup_command': None,
            'takeup_pulse_duration': None,
            'takeup_motor_steps': 0,
            'takeup_motor_direction': None,
            'takeup_timestamp': None,
            'takeup_interval_frames': None,
            'takeup_interval_before': None,
            'takeup_interval_after': None,
        }
        self._digital_write(self.LED_PIN, 0)
        self._digital_write(self.STEPPER_PINS[2], 0) #enable
        self._digital_write(self.STEPPER_PINS2[2], 0) #enable
        self._digital_write(self.STEPPER_PINS[1], 1) #direction forward
        self._digital_write(self.STEPPER_PINS2[1], 1) #direction forward

        self.PUSHER_RATIO = 0.98 # push ~2% less than pull
        self.FEED_INTERVAL = 5000
        self.TAKEUP_INTERVAL = 2500
        self.FEED_PULSE_DURATION = 0.1
        self.TAKEUP_PULSE_DURATION = 0.2
        self.ADVANCE_SETTLE_DELAY = 0.01
        self.POST_TAKEUP_SETTLE_DELAY = 0.01

    def _digital_write(self, pin, value):
        if self._closed:
            raise RuntimeError("controller is closed")
        self._device.write(int(pin) - 100, value)

    def light_on(self):
        self._digital_write(self.LED_PIN, 1)

    def light_off(self):
        self._digital_write(self.LED_PIN, 0)

    def set_reel_state(self, pin, enabled):
        self._digital_write(pin, 1 if enabled else 0)

    def feed_reel_on(self):
        self.set_reel_state(self.FEED_REEL_PIN, True)

    def feed_reel_off(self):
        self.set_reel_state(self.FEED_REEL_PIN, False)

    def takeup_reel_on(self):
        self.set_reel_state(self.TAKEUP_REEL_PIN, True)

    def takeup_reel_off(self):
        self.set_reel_state(self.TAKEUP_REEL_PIN, False)

    def get_last_takeup_telemetry(self):
        """Return telemetry for the most recent advance's take-up action."""
        return dict(self._last_takeup_telemetry)

    def begin_takeup_capture(self, interval_frames=12):
        """Use capture-local frame cadence; manual movement remains step-based."""
        self._adaptive_takeup_enabled = True
        self._takeup_frame_counter = 0
        self._takeup_interval_frames = max(1, int(interval_frames))

    def set_takeup_interval_frames(self, interval_frames):
        if not self._adaptive_takeup_enabled:
            raise RuntimeError('adaptive take-up capture is not active')
        self._takeup_interval_frames = max(1, int(interval_frames))

    def end_takeup_capture(self):
        self._adaptive_takeup_enabled = False
        self._takeup_frame_counter = 0
        self._takeup_interval_frames = None

    def steps_forward(self, steps=1):
        # Puller is master, always moves
        # Pusher moves according to PUSHER_RATIO
        pusher_counter = 0.0

        for _ in range(steps):
            # Decide if pusher should move this step
            pusher_counter += self.PUSHER_RATIO
            pusher_step = pusher_counter >= 1.0

            # --- STEP HIGH ---
            self._digital_write(self.STEPPER_PINS2[0], 1)  # puller step
            if pusher_step:
                self._digital_write(self.STEPPER_PINS[0], 1)  # pusher step
                pusher_counter -= 1.0

            time.sleep(0.000001)

            # --- STEP LOW ---
            self._digital_write(self.STEPPER_PINS2[0], 0)
            if pusher_step:
                self._digital_write(self.STEPPER_PINS[0], 0)

        feed_pulses, takeup_pulses = self._schedule_reel_pulses(steps)
        self._run_deferred_reel_pulses(steps, feed_pulses, takeup_pulses)

    def steps_back(self, steps=1):
        self._digital_write(self.STEPPER_PINS[1], 0) #direction backwards
        self._digital_write(self.STEPPER_PINS2[1], 0)
        pusher_counter = 0.0

        for _ in range(steps):
            pusher_counter += self.PUSHER_RATIO
            pusher_step = pusher_counter >= 1.0

            self._digital_write(self.STEPPER_PINS2[0], 1)
            if pusher_step:
                self._digital_write(self.STEPPER_PINS[0], 1)
                pusher_counter -= 1.0

            time.sleep(0.000001)

            self._digital_write(self.STEPPER_PINS2[0], 0)
            if pusher_step:
                self._digital_write(self.STEPPER_PINS[0], 0)

        self._digital_write(self.STEPPER_PINS[1], 1) #direction back to foward
        self._digital_write(self.STEPPER_PINS2[1], 1)
        feed_pulses, takeup_pulses = self._schedule_reel_pulses(steps)
        self._run_deferred_reel_pulses(steps, feed_pulses, takeup_pulses)

    def _schedule_reel_pulses(self, advance_steps):
        self.feed_steps_taken += advance_steps
        feed_pulses = self.feed_steps_taken // self.FEED_INTERVAL
        self.feed_steps_taken = self.feed_steps_taken % self.FEED_INTERVAL

        if self._adaptive_takeup_enabled:
            self._takeup_frame_counter += 1
            takeup_pulses = int(
                self._takeup_frame_counter >= self._takeup_interval_frames
            )
            if takeup_pulses:
                self._takeup_frame_counter = 0
            return int(feed_pulses), takeup_pulses

        self.takeup_steps_taken += advance_steps
        takeup_pulses = self.takeup_steps_taken // self.TAKEUP_INTERVAL
        self.takeup_steps_taken = self.takeup_steps_taken % self.TAKEUP_INTERVAL

        return int(feed_pulses), int(takeup_pulses)

    def _run_deferred_reel_pulses(self, advance_steps, feed_pulses, takeup_pulses):
        interval = self._takeup_interval_frames if self._adaptive_takeup_enabled else None
        self._last_takeup_telemetry = {
            'takeup_active': bool(takeup_pulses > 0),
            'takeup_pulse_started': False,
            'takeup_pulse_sequence': int(self.takeup_pulse_sequence),
            'takeup_command': None,
            'takeup_pulse_duration': None,
            'takeup_motor_steps': 0,
            'takeup_motor_direction': None,
            'takeup_timestamp': None,
            'takeup_interval_frames': interval,
            'takeup_interval_before': interval if takeup_pulses else None,
            'takeup_interval_after': interval if takeup_pulses else None,
        }
        print(f"[APP] Advance complete: steps={advance_steps}")
        time.sleep(self.ADVANCE_SETTLE_DELAY)
        if feed_pulses <= 0 and takeup_pulses <= 0:
            return

        if feed_pulses > 0:
            print(f"[APP] Running feed reel pulses: count={feed_pulses}, steps={feed_pulses * self.FEED_INTERVAL}")
            for _ in range(feed_pulses):
                self.pulse_reel(self.FEED_REEL_PIN, self.FEED_PULSE_DURATION)

        if takeup_pulses > 0:
            print(f"[APP] Running take-up reel pulses: count={takeup_pulses}, steps={takeup_pulses * self.TAKEUP_INTERVAL}")
            for _ in range(takeup_pulses):
                started = time.monotonic_ns()
                self.takeup_pulse_sequence += 1
                self.pulse_reel(self.TAKEUP_REEL_PIN, self.TAKEUP_PULSE_DURATION)
                self._last_takeup_telemetry.update({
                    'takeup_pulse_started': True,
                    'takeup_pulse_sequence': int(self.takeup_pulse_sequence),
                    'takeup_command': {
                        'pulse_count': int(takeup_pulses),
                        'steps': int(takeup_pulses * self.TAKEUP_INTERVAL),
                    },
                    'takeup_pulse_duration': (
                        time.monotonic_ns() - started
                    ) / 1_000_000_000.0,
                    'takeup_motor_steps': int(self.TAKEUP_INTERVAL),
                    'takeup_timestamp': int(started),
                })

        print("[APP] Deferred reel pulses complete")
        time.sleep(self.POST_TAKEUP_SETTLE_DELAY)

    def pulse_reel(self, pin, duration):
        self.set_reel_state(pin, True)
        time.sleep(duration)
        self.set_reel_state(pin, False)

    def rewind(self):
        pin = self.TAKEUP_REEL_PIN
        self.set_reel_state(pin, True)
        time.sleep(1)
        self.set_reel_state(pin, False)

    def clean_up(self):
        if self._closed:
            return
        self.light_off()
        self.feed_reel_off()
        self.takeup_reel_off()
        self._digital_write(self.STEPPER_PINS[2], 0)
        self._device.close()
        self._closed = True
